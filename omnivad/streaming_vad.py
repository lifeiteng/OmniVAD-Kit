"""High-level streaming VAD: audio chunks in, completed segments out.

Wraps :class:`OmniStreamVAD` (model inference) and :class:`OmniStreamSegmenter`
(segment state machine) into a single object so callers don't have to glue
them together manually.

Typical usage::

    from omnivad import OmniStreamingVAD
    import numpy as np

    vad = OmniStreamingVAD()                 # default config
    pcm = np.fromfile("speech.pcm", dtype=np.int16)
    chunk_size = 160                          # 10ms @ 16kHz

    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        for start, end in vad.process(chunk):
            print(f"speech: {start:.2f}s -> {end:.2f}s")

    # End-of-stream: flush trailing in-progress segment (if any)
    for start, end in vad.flush():
        print(f"speech (tail): {start:.2f}s -> {end:.2f}s")

If you also need per-frame confidence, use OmniStreamVAD + OmniStreamSegmenter
directly — that's the supported way to share the inference output between
multiple consumers without doing it twice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from omnivad.stream_segmenter import OmniStreamSegmenter
from omnivad.stream_vad import OmniStreamVAD


class OmniStreamingVAD:
    """Audio-in → segments-out helper that owns a stream VAD + segmenter pair."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        threshold: float = 0.5,
        smooth_window_size: int = 5,
        min_speech_secs: float = 0.20,
        min_silence_secs: float = 0.20,
        max_chunk_secs: float = 30.0,
    ):
        # Note: stream VAD uses its own threshold for is_speech but the
        # segmenter re-decides based on raw confidence + smoothing, so the
        # VAD threshold here mainly affects the StreamResult.is_speech bit
        # we don't use in this convenience wrapper.
        self._vad = OmniStreamVAD(model_path, threshold=threshold)
        self._segmenter = OmniStreamSegmenter(
            threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_secs=min_speech_secs,
            min_silence_secs=min_silence_secs,
            max_chunk_secs=max_chunk_secs,
        )
        self._total_samples_seen: int = 0

    # --------------------------------------------------------------------- #
    #  Streaming                                                             #
    # --------------------------------------------------------------------- #

    # Stream VAD's hop length: each omni_stream_vad_process() call advances
    # by exactly one frame. Feeding larger chunks in one call would silently
    # drop intermediate frames, so we split internally.
    _STREAM_VAD_HOP = 160  # 10ms @ 16kHz

    def process(self, pcm_chunk: np.ndarray) -> List[Tuple[float, float]]:
        """Feed one PCM chunk of arbitrary length (int16 or float32).
        Returns 0+ completed segments emitted during this call."""
        chunk = np.asarray(pcm_chunk)
        self._total_samples_seen += len(chunk)

        out: List[Tuple[float, float]] = []
        hop = self._STREAM_VAD_HOP
        for offset in range(0, len(chunk), hop):
            sub = chunk[offset : offset + hop]
            result = self._vad.process(sub)
            if result is not None:
                out.extend(self._segmenter.process_frame(result.confidence))
        return out

    def flush(self) -> List[Tuple[float, float]]:
        """Emit any in-progress segment at end-of-stream."""
        return self._segmenter.flush(self._total_samples_seen)

    # --------------------------------------------------------------------- #
    #  State queries                                                         #
    # --------------------------------------------------------------------- #

    @property
    def is_in_speech(self) -> bool:
        return self._segmenter.is_in_speech

    @property
    def active_start(self) -> Optional[float]:
        return self._segmenter.active_start

    @property
    def total_samples_seen(self) -> int:
        return self._total_samples_seen

    # --------------------------------------------------------------------- #
    #  Lifecycle                                                             #
    # --------------------------------------------------------------------- #

    def reset(self) -> None:
        """Reset both the VAD model state and the segmenter state machine."""
        self._vad.reset()
        self._segmenter.reset()
        self._total_samples_seen = 0

    def close(self) -> None:
        if self._segmenter:
            self._segmenter.close()
        if self._vad:
            self._vad.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
