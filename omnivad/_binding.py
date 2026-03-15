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

OMNI_ERR_NO_FRAMES = -6


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


class OmniVadStreamResult(ctypes.Structure):
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
_OmniVadHandle = ctypes.c_void_p

_lib.omni_vad_stream_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float,
]
_lib.omni_vad_stream_create.restype = _OmniVadHandle

_lib.omni_vad_stream_create_from_bundle.argtypes = [ctypes.c_char_p, ctypes.c_float]
_lib.omni_vad_stream_create_from_bundle.restype = _OmniVadHandle

_lib.omni_vad_stream_process.argtypes = [
    _OmniVadHandle, ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.POINTER(OmniVadStreamResult),
]
_lib.omni_vad_stream_process.restype = ctypes.c_int

_lib.omni_vad_stream_detect_full.argtypes = [
    _OmniVadHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_stream_detect_full.restype = ctypes.c_int

_lib.omni_vad_stream_reset.argtypes = [_OmniVadHandle]
_lib.omni_vad_stream_reset.restype = None

_lib.omni_vad_stream_get_frame_offset.argtypes = [_OmniVadHandle]
_lib.omni_vad_stream_get_frame_offset.restype = ctypes.c_int

_lib.omni_vad_stream_destroy.argtypes = [_OmniVadHandle]
_lib.omni_vad_stream_destroy.restype = None

# -- Non-stream VAD --
_OmniVadNonStreamHandle = ctypes.c_void_p

_lib.omni_vad_nonstream_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
]
_lib.omni_vad_nonstream_create.restype = _OmniVadNonStreamHandle

_lib.omni_vad_nonstream_create_from_bundle.argtypes = [ctypes.c_char_p]
_lib.omni_vad_nonstream_create_from_bundle.restype = _OmniVadNonStreamHandle

_lib.omni_vad_nonstream_process.argtypes = [
    _OmniVadNonStreamHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(OmniPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniSegment)), ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_nonstream_process.restype = ctypes.c_int

_lib.omni_vad_nonstream_process_raw.argtypes = [
    _OmniVadNonStreamHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int),
]
_lib.omni_vad_nonstream_process_raw.restype = ctypes.c_int

_lib.omni_vad_nonstream_destroy.argtypes = [_OmniVadNonStreamHandle]
_lib.omni_vad_nonstream_destroy.restype = None

# -- Non-stream AED --
_OmniAedNonStreamHandle = ctypes.c_void_p

_lib.omni_aed_nonstream_create.argtypes = [
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
]
_lib.omni_aed_nonstream_create.restype = _OmniAedNonStreamHandle

_lib.omni_aed_nonstream_create_from_bundle.argtypes = [ctypes.c_char_p]
_lib.omni_aed_nonstream_create_from_bundle.restype = _OmniAedNonStreamHandle

_lib.omni_aed_nonstream_process.argtypes = [
    _OmniAedNonStreamHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(OmniAedPostConfig),
    ctypes.POINTER(ctypes.POINTER(OmniAedSegment)), ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_nonstream_process.restype = ctypes.c_int

_lib.omni_aed_nonstream_process_raw.argtypes = [
    _OmniAedNonStreamHandle, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int),
]
_lib.omni_aed_nonstream_process_raw.restype = ctypes.c_int

_lib.omni_aed_nonstream_destroy.argtypes = [_OmniAedNonStreamHandle]
_lib.omni_aed_nonstream_destroy.restype = None
