#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_float, c_int, c_void};

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

pub const OMNI_AED_EVENT_SILENCE: c_int = 0;
pub const OMNI_AED_EVENT_SPEECH: c_int = 1;
pub const OMNI_AED_EVENT_SINGING: c_int = 2;
pub const OMNI_AED_EVENT_MUSIC: c_int = 3;
pub const OMNI_AED_EVENT_MIXED: c_int = 4;

pub const OMNI_AED_KIND_MASK_SPEECH: u32 = 1 << 0;
pub const OMNI_AED_KIND_MASK_SINGING: u32 = 1 << 1;
pub const OMNI_AED_KIND_MASK_MUSIC: u32 = 1 << 2;

pub enum OmniAedOverlapSegmenterCtx {}
pub type OmniAedOverlapSegmenterHandle = *mut OmniAedOverlapSegmenterCtx;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn aed_overlap_struct_layout_matches_c_abi() {
        assert_eq!(size_of::<OmniAedOverlapConfig>(), 52);
        assert_eq!(align_of::<OmniAedOverlapConfig>(), 4);
        assert_eq!(size_of::<OmniAedOnlineEvent>(), 32);
        assert_eq!(align_of::<OmniAedOnlineEvent>(), 4);
        assert_eq!(size_of::<OmniAedOnlineSegment>(), 16);
        assert_eq!(align_of::<OmniAedOnlineSegment>(), 4);
    }
}
