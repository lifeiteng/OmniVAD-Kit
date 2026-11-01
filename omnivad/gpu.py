"""Optional batched PyTorch backend for the bundled non-streaming VAD.

PyTorch is imported only by this module. The standard native backend is unchanged.
The supported ncnn graph is checked before loading its existing weights; no model
conversion, retraining, downloads, or automatic CUDA device selection is required.
"""

import hashlib
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ._binding import default_model_dir


class _Weights:
    def __init__(self, data):
        self.data, self.position = data, 0

    def read(self, count):
        if self.position + 4 > len(self.data):
            raise ValueError("Truncated ncnn weight header")
        tag = struct.unpack_from("<I", self.data, self.position)[0]
        self.position += 4
        if tag == 0x01306B47:
            dtype = "<f2"
        elif tag == 0:
            dtype = "<f4"
        else:
            raise ValueError(f"Unsupported ncnn weight tag: {tag:#x}")
        size = count * np.dtype(dtype).itemsize
        if self.position + size > len(self.data):
            raise ValueError("Truncated ncnn weight data")
        result = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.position).astype(np.float32)
        self.position += (size + 3) // 4 * 4
        return torch.from_numpy(result)


def _load_bundle(path):
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise ValueError("Truncated OmniVAD bundle")
    magic, version, nparam, nbin, nmeans, nistd = struct.unpack_from("<4s5I", data)
    if magic != b"OVAD" or version != 1 or len(data) != 24 + nparam + nbin + nmeans + nistd:
        raise ValueError("Invalid OmniVAD bundle")
    param = data[24 : 24 + nparam]
    # Deliberately fail closed for other models and future graph revisions.
    expected = "9e361f9af4d9f94c879c75a91ba594fe612100dddd6d6fa657a748ff70e34704"
    if hashlib.sha256(param).hexdigest() != expected:
        raise ValueError("GPU backend supports only the bundled non-stream VAD graph")
    weights = _Weights(data[24 + nparam : 24 + nparam + nbin])
    linear, memory = [], []
    for line in param.decode().splitlines()[2:]:
        fields = line.split()
        nb, nt = int(fields[2]), int(fields[3])
        args = dict(v.split("=", 1) for v in fields[4 + nb + nt :])
        if fields[0] == "Gemm":
            n, k = int(args["8"]), int(args["9"])
            matrix = weights.read(n * k).reshape(n, k)
            bias = weights.read(n) if args["10"] == "4" else torch.zeros(n)
            linear.append((matrix, bias))
        elif fields[0] == "ConvolutionDepthWise1D":
            memory.append(weights.read(int(args["6"])).reshape(128, 1, 20))
    if weights.position != len(weights.data) or len(linear) != 18 or len(memory) != 16:
        raise ValueError("Unexpected non-stream VAD weight layout")
    offset = 24 + nparam + nbin
    means = np.frombuffer(data, dtype="<f4", count=nmeans // 4, offset=offset).copy()
    istd = np.frombuffer(data, dtype="<f4", count=nistd // 4, offset=offset + nmeans).copy()
    if means.shape != (80,) or istd.shape != (80,):
        raise ValueError("Expected 80-dimensional VAD CMVN")
    return linear, memory, torch.from_numpy(means), torch.from_numpy(istd)


class OmniVADGPU:
    """Batch non-stream VAD using the existing bundle on an explicit device.

    Inputs are finite float32 mono arrays at 16 kHz. FP32 is intentional: VAD
    thresholds are sensitive to rounding. Variable lengths are masked at every
    DFSMN block so padding cannot become speech or leak through future context.
    """

    def __init__(self, model_path=None, *, device="cuda:0", fallback_margin=1e-4, **postprocessing):
        self.device = torch.device(device)
        defaults = dict(
            threshold=0.4,
            smooth_window_size=5,
            min_speech_frames=20,
            max_speech_frames=3000,
            min_silence_frames=20,
            merge_silence_frames=0,
            extend_speech_frames=0,
        )
        if set(postprocessing) - set(defaults):
            raise ValueError("Unknown VAD postprocessing option")
        self.config = {**defaults, **postprocessing}
        if not 0 <= self.config["threshold"] <= 1 or not np.isfinite(self.config["threshold"]):
            raise ValueError("Threshold must be finite and in [0, 1]")
        for key, value in self.config.items():
            if key != "threshold" and (type(value) is not int or value < 0):
                raise ValueError("Frame configuration must be nonnegative integers")
        if not np.isfinite(fallback_margin) or fallback_margin < 0:
            raise ValueError("Fallback margin must be finite and nonnegative")
        self.fallback_margin = fallback_margin
        self.fallback_count = 0
        self._cpu = None
        self.model_path = Path(model_path or Path(default_model_dir()) / "vad.omnivad")
        linear, memory, means, istd = _load_bundle(self.model_path)
        self.linear = [(w.to(self.device), b.to(self.device)) for w, b in linear]
        self.memory = [w.to(self.device) for w in memory]
        self.means, self.istd = means.to(self.device), istd.to(self.device)
        # Match native/frontend/fbank.h, including its Povey window and mel bins.
        frame = np.arange(400, dtype=np.float64)
        self.window = torch.tensor(
            (0.5 - 0.5 * np.cos(2 * np.pi * frame / 399)) ** 0.85, dtype=torch.float32, device=self.device
        )
        mel_low = np.float32(1127) * np.log(np.float32(1) + np.float32(20) / np.float32(700))
        mel_high = np.float32(1127) * np.log(np.float32(1) + np.float32(16000 / 2) / np.float32(700))
        delta = np.float32((mel_high - mel_low) / np.float32(81))
        frequencies = np.arange(256, dtype=np.float32) * np.float32(16000 / 512)
        mel = np.float32(1127) * np.log(np.float32(1) + frequencies / np.float32(700))
        filters = []
        for i in range(80):
            left = np.float32(mel_low + np.float32(i) * delta)
            center = np.float32(mel_low + np.float32(i + 1) * delta)
            right = np.float32(mel_low + np.float32(i + 2) * delta)
            filters.append(np.maximum(0, np.minimum((mel - left) / (center - left), (right - mel) / (right - center))))
        self.filters = torch.tensor(np.array(filters).T, device=self.device)

    @torch.inference_mode()
    def features(self, audios):
        if not audios:
            raise ValueError("An empty VAD batch is not supported")
        lengths = []
        for audio in audios:
            if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
                raise ValueError("Expected float32 mono arrays")
            if len(audio) < 400 or not np.isfinite(audio).all():
                raise ValueError("Expected finite audio with at least one 25 ms frame")
            lengths.append(1 + (len(audio) - 400) // 160)
        samples = np.zeros((len(audios), max(map(len, audios))), dtype=np.float32)
        for i, audio in enumerate(audios):
            samples[i, : len(audio)] = audio
        frames = (torch.from_numpy(samples).to(self.device) * 32768).unfold(1, 400, 160)
        frames = frames - frames.mean(-1, keepdim=True)
        # The native frontend stores the previous frame's last pre-emphasized
        # sample, rather than the previous waveform sample. Preserve that rule.
        previous = F.pad(frames[:, :-1, -1] - 0.97 * frames[:, :-1, -2], (1, 0))
        emphasized = torch.cat(
            (frames[:, :, :1] - 0.97 * previous[:, :, None], frames[:, :, 1:] - 0.97 * frames[:, :, :-1]), dim=-1
        )
        spectrum = torch.fft.rfft(emphasized * self.window, n=512)[..., :256]
        power = spectrum.real.square() + spectrum.imag.square()
        features = ((power @ self.filters).clamp_min(1e-20).log() - self.means) * self.istd
        return features, torch.tensor(lengths, device=self.device)

    @torch.inference_mode()
    def infer_features(self, features, lengths):
        mask = (torch.arange(features.shape[1], device=self.device)[None, :] < lengths[:, None])[:, :, None]

        def linear(x, i):
            return F.linear(x, *self.linear[i])

        x = F.relu(linear(F.relu(linear(features, 0)), 1)) * mask
        state = None
        for block in range(8):
            if block:
                x = linear(F.relu(linear(state, 2 * block)), 2 * block + 1) * mask
            channels = x.transpose(1, 2)
            past = F.conv1d(F.pad(channels, (19, 0)), self.memory[2 * block], groups=128)
            future = F.conv1d(F.pad(channels, (0, 20)), self.memory[2 * block + 1], groups=128)[..., 1:]
            current = x + past.transpose(1, 2) + future.transpose(1, 2)
            state = (current if state is None else current + state) * mask
        probabilities = torch.sigmoid(linear(F.relu(linear(state, 16)), 17).squeeze(-1) + 0.04882)
        return probabilities

    def detect_probs_batch(self, audios):
        features, lengths = self.features(audios)
        output = self.infer_features(features, lengths).cpu().numpy()
        return [np.array(row[: int(length)], copy=True) for row, length in zip(output, lengths.cpu().tolist())]

    def detect_batch(self, audios, sample_rate=16000):
        if sample_rate != 16000:
            raise ValueError("GPU VAD accepts only 16 kHz audio")
        probabilities = self.detect_probs_batch(audios)
        results = []
        for audio, probs in zip(audios, probabilities):
            timestamps, smooth = _segments(probs, len(audio), self.config)
            # Threshold-near clips use the existing native path. Long-clip split
            # points depend on argmin and also conservatively use native output.
            borderline = bool(np.any(np.abs(smooth - self.config["threshold"]) <= self.fallback_margin))
            long_clip = self.config["max_speech_frames"] > 0 and len(probs) > self.config["max_speech_frames"]
            if borderline or long_clip:
                if self._cpu is None:
                    from .vad import OmniVAD

                    self._cpu = OmniVAD(model_path=str(self.model_path), **self.config)
                timestamps = self._cpu.detect(audio, sample_rate)["timestamps"]
                self.fallback_count += 1
            results.append(dict(duration=round(len(audio) / sample_rate, 3), timestamps=timestamps))
        return results

    def detect(self, audio, sample_rate=16000):
        return self.detect_batch([audio], sample_rate)[0]


def _segments(probabilities, samples, config):
    """Match native postprocess_pipeline, including float32 accumulation order."""
    raw = np.asarray(probabilities, dtype=np.float32)
    n = len(raw)
    window = config["smooth_window_size"]
    smooth = raw.copy()
    if window > 1:
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(raw, (window - 1, 0)), window)
        # Accumulate left-to-right, exactly as native code; pairwise reduction
        # can change decisions close to the configured speech threshold.
        totals = np.add.accumulate(windows, axis=1, dtype=np.float32)[:, -1]
        smooth = totals / np.minimum(np.arange(1, n + 1), window).astype(np.float32)
    binary = smooth >= np.float32(config["threshold"])
    decisions = np.zeros(n, dtype=bool)
    minimum, silence = config["min_speech_frames"], config["min_silence_frames"]
    if minimum <= 0 and silence <= 0:
        decisions[:] = binary
    else:
        state, start, quiet = 0, -1, -1
        for t, speech in enumerate(binary):
            if state == 0 and speech:
                state, start = 1, t
            elif state == 1:
                if speech and t - start >= minimum:
                    state = 2
                    decisions[start:t] = True
                elif not speech:
                    state, start = 0, -1
            elif state == 2 and not speech:
                state, quiet = 3, t
            elif state == 3:
                if not speech and t - quiet >= silence:
                    state, start = 0, -1
                elif speech:
                    state, quiet = 2, -1
            decisions[t] = state in (2, 3)
    before = decisions.copy()
    for t in range(1, n):
        if not before[t - 1] and before[t]:
            decisions[max(0, t - window) : t] = True
    before = decisions.copy()
    quiet = -1
    for t in range(1, n):
        if before[t - 1] and not before[t] and quiet < 0:
            quiet = t
        elif not before[t - 1] and before[t] and quiet >= 0:
            if t - quiet < config["merge_silence_frames"]:
                decisions[quiet:t] = True
            quiet = -1
    extension = config["extend_speech_frames"]
    if extension > 0:
        before = decisions.copy()
        for t in np.flatnonzero(before):
            decisions[max(0, t - extension) : min(n, t + extension + 1)] = True

    def spans(values):
        changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))

    maximum = config["max_speech_frames"]
    if maximum > 0:
        for start, end in spans(decisions):
            local = 0
            while end - start - local > maximum:
                lo = start + local + maximum // 2
                hi = min(end, start + local + maximum)
                cut = lo + int(np.argmin(raw[lo:hi]))
                decisions[cut] = False
                local = cut - start + 1
    result = []
    for start, end in spans(decisions):
        a = np.float32(start) * np.float32(0.01)
        b = np.float32(end) * np.float32(0.01)
        if end == n:
            b = min(np.float32(b + np.float32(0.025)), np.float32(samples) / np.float32(16000))
        result.append((round(float(a), 3), round(float(b), 3)))
    return result, smooth
