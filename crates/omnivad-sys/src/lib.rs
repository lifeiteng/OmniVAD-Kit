#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_float, c_int, c_uchar, c_void};

pub const OMNI_OK: c_int = 0;
pub const OMNI_ERR_NULL_HANDLE: c_int = -1;
pub const OMNI_ERR_NULL_POINTER: c_int = -2;
pub const OMNI_ERR_LOAD_BUNDLE: c_int = -3;
pub const OMNI_ERR_LOAD_PARAM: c_int = -4;
pub const OMNI_ERR_LOAD_MODEL: c_int = -5;
pub const OMNI_ERR_LOAD_CMVN: c_int = -6;
pub const OMNI_ERR_NO_FRAMES: c_int = -7;
pub const OMNI_ERR_INFERENCE: c_int = -8;
pub const OMNI_ERR_OUT_OF_MEMORY: c_int = -9;
pub const OMNI_ERR_INVALID_ARG: c_int = -10;

pub const OMNI_AED_SPEECH: c_int = 0;
pub const OMNI_AED_SINGING: c_int = 1;
pub const OMNI_AED_MUSIC: c_int = 2;

pub const OMNI_AED_EVENT_SILENCE: c_int = 0;
pub const OMNI_AED_EVENT_SPEECH: c_int = 1;
pub const OMNI_AED_EVENT_SINGING: c_int = 2;
pub const OMNI_AED_EVENT_MUSIC: c_int = 3;
pub const OMNI_AED_EVENT_MIXED: c_int = 4;

pub const OMNI_CHUNK_GREEDY: c_int = 0;
pub const OMNI_CHUNK_LONGEST_GAP: c_int = 1;

pub const OMNI_AED_KIND_MASK_SPEECH: u32 = 1 << 0;
pub const OMNI_AED_KIND_MASK_SINGING: u32 = 1 << 1;
pub const OMNI_AED_KIND_MASK_MUSIC: u32 = 1 << 2;

pub enum OmniVadCtx {}
pub type OmniVadHandle = *mut OmniVadCtx;

pub enum OmniStreamVadCtx {}
pub type OmniStreamVadHandle = *mut OmniStreamVadCtx;

pub enum OmniAedCtx {}
pub type OmniAedHandle = *mut OmniAedCtx;

pub enum OmniAedOverlapSegmenterCtx {}
pub type OmniAedOverlapSegmenterHandle = *mut OmniAedOverlapSegmenterCtx;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniSegment {
    pub start: c_float,
    pub end: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniAedSegment {
    pub start: c_float,
    pub end: c_float,
    pub cls: c_int,
    pub confidence: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniPostConfig {
    pub threshold: c_float,
    pub smooth_window_size: c_int,
    pub min_speech_frames: c_int,
    pub min_silence_frames: c_int,
    pub max_speech_frames: c_int,
    pub merge_silence_frames: c_int,
    pub extend_speech_frames: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniStreamVadConfig {
    pub threshold: c_float,
    pub smooth_window_size: c_int,
    pub pad_start_frame: c_int,
    pub min_speech_frame: c_int,
    pub max_speech_frame: c_int,
    pub min_silence_frame: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniStreamVadResult {
    pub confidence: c_float,
    pub smoothed_prob: c_float,
    pub is_speech: c_uchar,
    pub is_speech_start: c_uchar,
    pub is_speech_end: c_uchar,
    pub frame_idx: c_int,
    pub speech_start_frame: c_int,
    pub speech_end_frame: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniAedPostConfig {
    pub speech: OmniPostConfig,
    pub singing: OmniPostConfig,
    pub music: OmniPostConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniChunk {
    pub start: c_float,
    pub end: c_float,
    pub seg_start_idx: c_int,
    pub seg_count: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniChunkConfig {
    pub max_chunk_secs: c_float,
    pub max_gap_secs: c_float,
    pub pad_onset_secs: c_float,
    pub pad_offset_secs: c_float,
    pub min_speech_secs: c_float,
    pub min_silence_secs: c_float,
    pub mode: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniAedOverlapConfig {
    pub hop_ms: c_int,
    pub overlap_ms: c_int,
    pub edge_guard_ms: c_int,
    pub hard_split_pause_ms: c_int,
    pub max_chunk_ms: c_int,
    pub min_speech_ms: c_int,
    pub merge_gap_ms: c_int,
    pub music_gap_tolerance_ms: c_int,
    pub pad_start_ms: c_int,
    pub pad_end_ms: c_int,
    pub speech_threshold: c_float,
    pub singing_threshold: c_float,
    pub music_threshold: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniAedOnlineEvent {
    pub start: c_float,
    pub end: c_float,
    pub primary_kind: c_int,
    pub kind_mask: u32,
    pub speech_confidence: c_float,
    pub singing_confidence: c_float,
    pub music_confidence: c_float,
    pub confidence: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OmniAedOnlineSegment {
    pub start: c_float,
    pub end: c_float,
    pub event_start_idx: c_int,
    pub event_count: c_int,
}

extern "C" {
    pub fn omni_error_string(code: c_int) -> *const c_char;
    pub fn omni_free(ptr: *mut c_void);

    pub fn omni_post_config_default() -> OmniPostConfig;

    pub fn omni_vad_create(bundle_path: *const c_char, out_error: *mut c_int) -> OmniVadHandle;
    pub fn omni_vad_create_from_buffer(
        data: *const c_void,
        size: c_int,
        out_error: *mut c_int,
    ) -> OmniVadHandle;
    pub fn omni_vad_detect(
        handle: OmniVadHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        config: *const OmniPostConfig,
        out_segments: *mut *mut OmniSegment,
        out_count: *mut c_int,
    ) -> c_int;
    pub fn omni_vad_detect_int16(
        handle: OmniVadHandle,
        audio_data: *const i16,
        num_samples: c_int,
        config: *const OmniPostConfig,
        out_segments: *mut *mut OmniSegment,
        out_count: *mut c_int,
    ) -> c_int;
    pub fn omni_vad_detect_probs(
        handle: OmniVadHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_vad_detect_probs_int16(
        handle: OmniVadHandle,
        audio_data: *const i16,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_vad_destroy(handle: OmniVadHandle);

    pub fn omni_stream_vad_config_default() -> OmniStreamVadConfig;
    pub fn omni_stream_vad_create(
        bundle_path: *const c_char,
        config: *const OmniStreamVadConfig,
        out_error: *mut c_int,
    ) -> OmniStreamVadHandle;
    pub fn omni_stream_vad_create_from_buffer(
        data: *const c_void,
        size: c_int,
        config: *const OmniStreamVadConfig,
        out_error: *mut c_int,
    ) -> OmniStreamVadHandle;
    pub fn omni_stream_vad_clone(
        handle: OmniStreamVadHandle,
        out_error: *mut c_int,
    ) -> OmniStreamVadHandle;
    pub fn omni_stream_vad_process(
        handle: OmniStreamVadHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        result: *mut OmniStreamVadResult,
    ) -> c_int;
    pub fn omni_stream_vad_process_int16(
        handle: OmniStreamVadHandle,
        audio_data: *const i16,
        num_samples: c_int,
        result: *mut OmniStreamVadResult,
    ) -> c_int;
    pub fn omni_stream_vad_detect_full(
        handle: OmniStreamVadHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_stream_vad_detect_full_int16(
        handle: OmniStreamVadHandle,
        audio_data: *const i16,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_stream_vad_reset(handle: OmniStreamVadHandle);
    pub fn omni_stream_vad_get_frame_offset(handle: OmniStreamVadHandle) -> c_int;
    pub fn omni_stream_vad_destroy(handle: OmniStreamVadHandle);

    pub fn omni_aed_post_config_default() -> OmniAedPostConfig;
    pub fn omni_aed_create(bundle_path: *const c_char, out_error: *mut c_int) -> OmniAedHandle;
    pub fn omni_aed_create_from_buffer(
        data: *const c_void,
        size: c_int,
        out_error: *mut c_int,
    ) -> OmniAedHandle;
    pub fn omni_aed_detect(
        handle: OmniAedHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        config: *const OmniAedPostConfig,
        out_segments: *mut *mut OmniAedSegment,
        out_count: *mut c_int,
    ) -> c_int;
    pub fn omni_aed_detect_int16(
        handle: OmniAedHandle,
        audio_data: *const i16,
        num_samples: c_int,
        config: *const OmniAedPostConfig,
        out_segments: *mut *mut OmniAedSegment,
        out_count: *mut c_int,
    ) -> c_int;
    pub fn omni_aed_detect_probs(
        handle: OmniAedHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_aed_detect_probs_int16(
        handle: OmniAedHandle,
        audio_data: *const i16,
        num_samples: c_int,
        out_probs: *mut *mut c_float,
        out_frames: *mut c_int,
    ) -> c_int;
    pub fn omni_aed_destroy(handle: OmniAedHandle);

    pub fn omni_aed_overlap_config_default() -> OmniAedOverlapConfig;

    pub fn omni_aed_overlap_segmenter_create(
        bundle_path: *const c_char,
        config: *const OmniAedOverlapConfig,
        out_error: *mut c_int,
    ) -> OmniAedOverlapSegmenterHandle;

    pub fn omni_aed_overlap_segmenter_create_from_buffer(
        data: *const c_void,
        size: c_int,
        config: *const OmniAedOverlapConfig,
        out_error: *mut c_int,
    ) -> OmniAedOverlapSegmenterHandle;

    pub fn omni_aed_overlap_segmenter_clone(
        handle: OmniAedOverlapSegmenterHandle,
        out_error: *mut c_int,
    ) -> OmniAedOverlapSegmenterHandle;

    pub fn omni_aed_overlap_segmenter_ingest(
        handle: OmniAedOverlapSegmenterHandle,
        audio_data: *const c_float,
        num_samples: c_int,
        out_segments: *mut *mut OmniAedOnlineSegment,
        out_segment_count: *mut c_int,
        out_events: *mut *mut OmniAedOnlineEvent,
        out_event_count: *mut c_int,
    ) -> c_int;

    pub fn omni_aed_overlap_segmenter_ingest_int16(
        handle: OmniAedOverlapSegmenterHandle,
        audio_data: *const i16,
        num_samples: c_int,
        out_segments: *mut *mut OmniAedOnlineSegment,
        out_segment_count: *mut c_int,
        out_events: *mut *mut OmniAedOnlineEvent,
        out_event_count: *mut c_int,
    ) -> c_int;

    pub fn omni_aed_overlap_segmenter_flush(
        handle: OmniAedOverlapSegmenterHandle,
        out_segments: *mut *mut OmniAedOnlineSegment,
        out_segment_count: *mut c_int,
        out_events: *mut *mut OmniAedOnlineEvent,
        out_event_count: *mut c_int,
    ) -> c_int;

    pub fn omni_aed_overlap_segmenter_reset(handle: OmniAedOverlapSegmenterHandle);
    pub fn omni_aed_overlap_segmenter_destroy(handle: OmniAedOverlapSegmenterHandle);

    pub fn omni_chunk_config_default() -> OmniChunkConfig;
    pub fn omni_merge_chunks(
        segments: *const OmniSegment,
        num_segments: c_int,
        config: *const OmniChunkConfig,
        out_chunks: *mut *mut OmniChunk,
        out_count: *mut c_int,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn struct_layout_matches_c_abi() {
        assert_eq!(size_of::<OmniSegment>(), 8);
        assert_eq!(align_of::<OmniSegment>(), 4);
        assert_eq!(size_of::<OmniAedSegment>(), 16);
        assert_eq!(align_of::<OmniAedSegment>(), 4);
        assert_eq!(size_of::<OmniPostConfig>(), 28);
        assert_eq!(align_of::<OmniPostConfig>(), 4);
        assert_eq!(size_of::<OmniAedPostConfig>(), 84);
        assert_eq!(align_of::<OmniAedPostConfig>(), 4);
        assert_eq!(size_of::<OmniStreamVadConfig>(), 24);
        assert_eq!(align_of::<OmniStreamVadConfig>(), 4);
        assert_eq!(size_of::<OmniStreamVadResult>(), 24);
        assert_eq!(align_of::<OmniStreamVadResult>(), 4);
        assert_eq!(size_of::<OmniChunk>(), 16);
        assert_eq!(align_of::<OmniChunk>(), 4);
        assert_eq!(size_of::<OmniChunkConfig>(), 28);
        assert_eq!(align_of::<OmniChunkConfig>(), 4);
        assert_eq!(size_of::<OmniAedOverlapConfig>(), 52);
        assert_eq!(align_of::<OmniAedOverlapConfig>(), 4);
        assert_eq!(size_of::<OmniAedOnlineEvent>(), 32);
        assert_eq!(align_of::<OmniAedOnlineEvent>(), 4);
        assert_eq!(size_of::<OmniAedOnlineSegment>(), 16);
        assert_eq!(align_of::<OmniAedOnlineSegment>(), 4);
    }
}
