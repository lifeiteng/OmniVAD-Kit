"""Optional PyTorch backend parity and padding regression checks."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")
from omnivad import OmniVAD
from omnivad.gpu import OmniVADGPU

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def backend():
    import torch

    torch.set_num_threads(1)
    return OmniVADGPU(device="cuda:0" if torch.cuda.is_available() else "cpu")


def test_variable_length_batch_matches_native_and_individual_inference(backend):
    audios = [
        sf.read(DATA / (name + ".wav"), dtype="float32")[0]
        for name in ["hello_en", "hello_zh", "en_medium", "zh_medium", "event"]
    ]
    reference = OmniVAD()
    batch = backend.detect_probs_batch(audios)
    for audio, actual in zip(audios, batch):
        expected = reference.detect_probs(audio)
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-4)
        np.testing.assert_allclose(actual, backend.detect_probs_batch([audio])[0], atol=5e-5, rtol=5e-4)
    assert backend.detect_batch(audios) == [reference.detect(audio) for audio in audios]


@pytest.mark.parametrize("samples", [560, 16000])
def test_silence_and_frame_boundaries_match_native(backend, samples):
    audio = np.zeros(samples, dtype=np.float32)
    np.testing.assert_allclose(
        backend.detect_probs_batch([audio])[0], OmniVAD().detect_probs(audio), atol=5e-5, rtol=5e-4
    )
    assert backend.detect(audio) == OmniVAD().detect(audio)


@pytest.mark.parametrize(
    "config",
    [
        dict(smooth_window_size=1, min_speech_frames=0, min_silence_frames=0),
        dict(threshold=0.6, merge_silence_frames=10, extend_speech_frames=7),
        dict(max_speech_frames=150),
    ],
)
def test_custom_postprocessing_matches_native(config):
    model = OmniVADGPU(device="cuda:0" if __import__("torch").cuda.is_available() else "cpu", **config)
    audio = sf.read(DATA / "en_medium.wav", dtype="float32")[0]
    assert model.detect(audio) == OmniVAD(**config).detect(audio)


def test_borderline_uses_native_fallback():
    model = OmniVADGPU(device="cuda:0" if __import__("torch").cuda.is_available() else "cpu", fallback_margin=1)
    audio = sf.read(DATA / "hello_zh.wav", dtype="float32")[0]
    assert model.detect(audio) == OmniVAD().detect(audio)
    assert model.fallback_count == 1


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(399, dtype=np.float32),
        np.zeros(400, dtype=np.float64),
        np.full(400, np.nan, dtype=np.float32),
        np.zeros((400, 2), dtype=np.float32),
    ],
)
def test_rejects_invalid_audio(backend, audio):
    with pytest.raises(ValueError):
        backend.detect(audio)


def test_rejects_other_model_graph():
    with pytest.raises(ValueError, match="non-stream VAD graph"):
        OmniVADGPU(model_path=DATA.parent.parent / "models/aed.omnivad", device="cpu")


@pytest.mark.parametrize("samples", [400, 401, 559])
def test_one_frame_inputs_are_supported(backend, samples):
    # Native ncnn Crop cannot allocate its zero-length future-context branch.
    # The GPU implementation explicitly pads that branch instead.
    audio = np.zeros(samples, dtype=np.float32)
    probabilities = backend.detect_probs_batch([audio])[0]
    assert probabilities.shape == (1,) and np.isfinite(probabilities).all()
    assert backend.detect(audio)["timestamps"] == []
