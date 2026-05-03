"""
OmniVAD — Cross-platform Voice Activity Detection and Audio Event Detection.

Based on FireRedVAD (Xiaohongshu), using DFSMN models (~2MB, ~588K params).
Supports 100+ languages.

Quick Start
-----------
>>> from omnivad import OmniVAD
>>> vad = OmniVAD()
>>> result = vad.detect("audio.wav")
>>> print(result["timestamps"])
[(0.44, 1.82), (3.10, 5.60)]
"""

from omnivad.aed import OmniAED
from omnivad.chunking import ChunkResult, default_chunk_config, merge_chunks
from omnivad.stream_vad import OmniStreamVAD
from omnivad.vad import OmniVAD

__version__ = "0.2.8"
__all__ = [
    "OmniVAD",
    "OmniStreamVAD",
    "OmniAED",
    "merge_chunks",
    "ChunkResult",
    "default_chunk_config",
]
