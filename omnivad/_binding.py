"""Low-level ctypes bindings for libomnivad."""

import ctypes
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Library & model discovery                                                   #
# --------------------------------------------------------------------------- #

_LIB_NAMES = {
    "darwin": ["libomnivad.dylib"],
    "win32": ["omnivad.dll"],
    "linux": ["libomnivad.so"],
}


def _find_library() -> str:
    pkg_dir = Path(__file__).parent
    for name in _LIB_NAMES.get(sys.platform, ["libomnivad.so"]):
        path = pkg_dir / name
        if path.exists():
            return str(path)
    raise RuntimeError(
        f"Cannot find omnivad native library in {pkg_dir}. "
        "Make sure the package was installed correctly (pip install omnivad)."
    )


def default_model_dir() -> str:
    pkg_dir = Path(__file__).parent
    # Installed wheel
    d = pkg_dir / "models"
    if d.is_dir():
        return str(d)
    # Development (repo root)
    d = pkg_dir.parent / "models"
    if d.is_dir():
        return str(d)
    raise RuntimeError("Cannot find model directory")


# --------------------------------------------------------------------------- #
#  Error check helper                                                          #
# --------------------------------------------------------------------------- #

OMNI_ERR_NULL_HANDLE = -1
OMNI_ERR_NULL_POINTER = -2
OMNI_ERR_LOAD_BUNDLE = -3
OMNI_ERR_LOAD_PARAM = -4
OMNI_ERR_LOAD_MODEL = -5
OMNI_ERR_LOAD_CMVN = -6
OMNI_ERR_NO_FRAMES = -7
OMNI_ERR_INFERENCE = -8
OMNI_ERR_OUT_OF_MEMORY = -9
OMNI_ERR_INVALID_ARG = -10


def _check(ret: int) -> None:
    if ret != 0:
        msg = _lib.omni_error_string(ret)
        raise RuntimeError(f"OmniVAD error ({ret}): {msg.decode()}")


# --------------------------------------------------------------------------- #
#  Struct definitions (must match native/include/omnivad.h exactly)            #
# --------------------------------------------------------------------------- #


class OmniSegment(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
    ]


class OmniAedSegment(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
        ("cls", ctypes.c_int),
        ("confidence", ctypes.c_float),
    ]


class OmniStreamVadResult(ctypes.Structure):
    _fields_ = [
        ("confidence", ctypes.c_float),
        ("is_speech", ctypes.c_bool),
        ("frame_offset", ctypes.c_int),
    ]


class OmniPostConfig(ctypes.Structure):
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("smooth_window_size", ctypes.c_int),
        ("min_speech_frames", ctypes.c_int),
        ("min_silence_frames", ctypes.c_int),
        ("max_speech_frames", ctypes.c_int),
        ("merge_silence_frames", ctypes.c_int),
        ("extend_speech_frames", ctypes.c_int),
    ]


class OmniAedPostConfig(ctypes.Structure):
    _fields_ = [
        ("speech", OmniPostConfig),
        ("singing", OmniPostConfig),
        ("music", OmniPostConfig),
    ]


class OmniChunk(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_float),
        ("end", ctypes.c_float),
        ("seg_start_idx", ctypes.c_int),
        ("seg_count", ctypes.c_int),
    ]


# OmniChunkMode enum values (must match native/include/omnivad.h)
OMNI_CHUNK_GREEDY = 0
OMNI_CHUNK_LONGEST_GAP = 1


class OmniChunkConfig(ctypes.Structure):
    _fields_ = [
        ("max_chunk_secs", ctypes.c_float),
        ("max_gap_secs", ctypes.c_float),
        ("pad_onset_secs", ctypes.c_float),
        ("pad_offset_secs", ctypes.c_float),
        ("min_speech_secs", ctypes.c_float),
        ("min_silence_secs", ctypes.c_float),
        ("mode", ctypes.c_int),
    ]


# --------------------------------------------------------------------------- #
#  Load native library & declare function signatures                           #
# --------------------------------------------------------------------------- #

_lib = ctypes.CDLL(_find_library())

# -- Utility --
_lib.omni_error_string.argtypes = [ctypes.c_int]
_lib.omni_error_string.restype = ctypes.c_char_p

_lib.omni_free.argtypes = [ctypes.c_void_p]
_lib.omni_free.restype = None

_lib.omni_post_config_default.argtypes = []
_lib.omni_post_config_default.restype = OmniPostConfig

_lib.omni_aed_post_config_default.argtypes = []
_lib.omni_aed_post_config_default.restype = OmniAedPostConfig

# -- Stream VAD --
_OmniStreamVadHandle = ctypes.c_void_p

_lib.omni_stream_vad_create.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
_lib.omni_stream_vad_create.restype = _OmniStreamVadHandle

_lib.omni_stream_vad_clone.argtypes = [_OmniStreamVadHandle, ctypes.POINTER(ctypes.c_int)]
_lib.omni_stream_vad_clone.restype = _OmniStreamVadHandle

_lib.omni_stream_vad_process.argtypes = [
    _OmniStreamVadHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(OmniStreamVadResult),
]
_lib.omni_stream_vad_process.restype = ctypes.c_int

_lib.omni_stream_vad_detect_full.argtypes = [
    _OmniStreamVadHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_vad_detect_full.restype = ctypes.c_int

_lib.omni_stream_vad_detect_full_int16.argtypes = [
    _OmniStreamVadHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_vad_detect_full_int16.restype = ctypes.c_int

_lib.omni_stream_vad_reset.argtypes = [_OmniStreamVadHandle]
_lib.omni_stream_vad_reset.restype = None

_lib.omni_stream_vad_get_frame_offset.argtypes = [_OmniStreamVadHandle]
_lib.omni_stream_vad_get_frame_offset.restype = ctypes.c_int

_lib.omni_stream_vad_destroy.argtypes = [_OmniStreamVadHandle]
_lib.omni_stream_vad_destroy.restype = None

# -- VAD --
_OmniVadHandle = ctypes.c_void_p

_lib.omni_vad_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
_lib.omni_vad_create.restype = _OmniVadHandle

# float [-1,1] input (default)
_lib.omni_vad_detect.argtypes = [
    _OmniVadHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(OmniPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_detect.restype = ctypes.c_int

# int16 PCM input
_lib.omni_vad_detect_int16.argtypes = [
    _OmniVadHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(OmniPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_detect_int16.restype = ctypes.c_int

# per-frame probs, float [-1,1] input
_lib.omni_vad_detect_probs.argtypes = [
    _OmniVadHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_detect_probs.restype = ctypes.c_int

# per-frame probs, int16 input
_lib.omni_vad_detect_probs_int16.argtypes = [
    _OmniVadHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_detect_probs_int16.restype = ctypes.c_int

_lib.omni_vad_destroy.argtypes = [_OmniVadHandle]
_lib.omni_vad_destroy.restype = None

# -- AED --
_OmniAedHandle = ctypes.c_void_p

_lib.omni_aed_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
_lib.omni_aed_create.restype = _OmniAedHandle

_lib.omni_aed_detect.argtypes = [
    _OmniAedHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(OmniAedPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniAedSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_detect.restype = ctypes.c_int

_lib.omni_aed_detect_int16.argtypes = [
    _OmniAedHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(OmniAedPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniAedSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_detect_int16.restype = ctypes.c_int

_lib.omni_aed_detect_probs.argtypes = [
    _OmniAedHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_detect_probs.restype = ctypes.c_int

_lib.omni_aed_detect_probs_int16.argtypes = [
    _OmniAedHandle,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_detect_probs_int16.restype = ctypes.c_int

_lib.omni_aed_destroy.argtypes = [_OmniAedHandle]
_lib.omni_aed_destroy.restype = None

# -- Chunking (pure-algorithm) --
_lib.omni_chunk_config_default.argtypes = []
_lib.omni_chunk_config_default.restype = OmniChunkConfig

_lib.omni_merge_chunks.argtypes = [
    ctypes.POINTER(OmniSegment),
    ctypes.c_int,
    ctypes.POINTER(OmniChunkConfig),
    ctypes.POINTER(ctypes.POINTER(OmniChunk)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_merge_chunks.restype = ctypes.c_int

# -- Stream Segmenter (pure-algorithm, no model) --
_OmniStreamSegmenterHandle = ctypes.c_void_p

_lib.omni_stream_segmenter_create.argtypes = [
    ctypes.POINTER(OmniPostConfig),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_segmenter_create.restype = _OmniStreamSegmenterHandle

_lib.omni_stream_segmenter_process_frame.argtypes = [
    _OmniStreamSegmenterHandle,
    ctypes.c_float,
    ctypes.POINTER(ctypes.POINTER(OmniSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_segmenter_process_frame.restype = ctypes.c_int

_lib.omni_stream_segmenter_process_probs.argtypes = [
    _OmniStreamSegmenterHandle,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(OmniSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_segmenter_process_probs.restype = ctypes.c_int

_lib.omni_stream_segmenter_flush.argtypes = [
    _OmniStreamSegmenterHandle,
    ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(OmniSegment)),
    ctypes.POINTER(ctypes.c_int),
]
_lib.omni_stream_segmenter_flush.restype = ctypes.c_int

_lib.omni_stream_segmenter_is_in_speech.argtypes = [_OmniStreamSegmenterHandle]
_lib.omni_stream_segmenter_is_in_speech.restype = ctypes.c_bool

_lib.omni_stream_segmenter_get_active_start.argtypes = [_OmniStreamSegmenterHandle]
_lib.omni_stream_segmenter_get_active_start.restype = ctypes.c_float

_lib.omni_stream_segmenter_reset.argtypes = [_OmniStreamSegmenterHandle]
_lib.omni_stream_segmenter_reset.restype = None

_lib.omni_stream_segmenter_destroy.argtypes = [_OmniStreamSegmenterHandle]
_lib.omni_stream_segmenter_destroy.restype = None
