use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use omnivad_sys as sys;

use crate::error::{cstring_from_path, len_to_c_int, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AedOverlapConfig {
    pub hop_ms: i32,
    pub overlap_ms: i32,
    pub edge_guard_ms: i32,
    pub hard_split_pause_ms: i32,
    pub max_chunk_ms: i32,
    pub min_speech_ms: i32,
    pub merge_gap_ms: i32,
    pub music_gap_tolerance_ms: i32,
    pub pad_start_ms: i32,
    pub pad_end_ms: i32,
    pub speech_threshold: f32,
    pub singing_threshold: f32,
    pub music_threshold: f32,
}

impl Default for AedOverlapConfig {
    fn default() -> Self {
        unsafe { sys::omni_aed_overlap_config_default() }.into()
    }
}

impl From<sys::OmniAedOverlapConfig> for AedOverlapConfig {
    fn from(value: sys::OmniAedOverlapConfig) -> Self {
        Self {
            hop_ms: value.hop_ms,
            overlap_ms: value.overlap_ms,
            edge_guard_ms: value.edge_guard_ms,
            hard_split_pause_ms: value.hard_split_pause_ms,
            max_chunk_ms: value.max_chunk_ms,
            min_speech_ms: value.min_speech_ms,
            merge_gap_ms: value.merge_gap_ms,
            music_gap_tolerance_ms: value.music_gap_tolerance_ms,
            pad_start_ms: value.pad_start_ms,
            pad_end_ms: value.pad_end_ms,
            speech_threshold: value.speech_threshold,
            singing_threshold: value.singing_threshold,
            music_threshold: value.music_threshold,
        }
    }
}

impl From<AedOverlapConfig> for sys::OmniAedOverlapConfig {
    fn from(value: AedOverlapConfig) -> Self {
        Self {
            hop_ms: value.hop_ms,
            overlap_ms: value.overlap_ms,
            edge_guard_ms: value.edge_guard_ms,
            hard_split_pause_ms: value.hard_split_pause_ms,
            max_chunk_ms: value.max_chunk_ms,
            min_speech_ms: value.min_speech_ms,
            merge_gap_ms: value.merge_gap_ms,
            music_gap_tolerance_ms: value.music_gap_tolerance_ms,
            pad_start_ms: value.pad_start_ms,
            pad_end_ms: value.pad_end_ms,
            speech_threshold: value.speech_threshold,
            singing_threshold: value.singing_threshold,
            music_threshold: value.music_threshold,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AedEventKind {
    Silence,
    Speech,
    Singing,
    Music,
    Mixed,
    Unknown(i32),
}

impl From<i32> for AedEventKind {
    fn from(value: i32) -> Self {
        match value {
            sys::OMNI_AED_EVENT_SILENCE => Self::Silence,
            sys::OMNI_AED_EVENT_SPEECH => Self::Speech,
            sys::OMNI_AED_EVENT_SINGING => Self::Singing,
            sys::OMNI_AED_EVENT_MUSIC => Self::Music,
            sys::OMNI_AED_EVENT_MIXED => Self::Mixed,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AedKindMask(pub u32);

impl AedKindMask {
    pub const SPEECH: Self = Self(sys::OMNI_AED_KIND_MASK_SPEECH);
    pub const SINGING: Self = Self(sys::OMNI_AED_KIND_MASK_SINGING);
    pub const MUSIC: Self = Self(sys::OMNI_AED_KIND_MASK_MUSIC);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn contains_any(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn is_transcribable(self) -> bool {
        self.contains_any(Self(Self::SPEECH.0 | Self::SINGING.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AedOnlineEvent {
    pub start: f32,
    pub end: f32,
    pub primary_kind: AedEventKind,
    pub kind_mask: AedKindMask,
    pub speech_confidence: f32,
    pub singing_confidence: f32,
    pub music_confidence: f32,
    pub confidence: f32,
}

impl From<sys::OmniAedOnlineEvent> for AedOnlineEvent {
    fn from(value: sys::OmniAedOnlineEvent) -> Self {
        Self {
            start: value.start,
            end: value.end,
            primary_kind: AedEventKind::from(value.primary_kind),
            kind_mask: AedKindMask(value.kind_mask),
            speech_confidence: value.speech_confidence,
            singing_confidence: value.singing_confidence,
            music_confidence: value.music_confidence,
            confidence: value.confidence,
        }
    }
}

impl AedOnlineEvent {
    pub fn is_transcribable(&self) -> bool {
        self.kind_mask.is_transcribable()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AedOnlineSegment {
    pub start: f32,
    pub end: f32,
    pub event_start_idx: i32,
    pub event_count: i32,
}

impl From<sys::OmniAedOnlineSegment> for AedOnlineSegment {
    fn from(value: sys::OmniAedOnlineSegment) -> Self {
        Self {
            start: value.start,
            end: value.end,
            event_start_idx: value.event_start_idx,
            event_count: value.event_count,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AedOverlapResult {
    pub segments: Vec<AedOnlineSegment>,
    pub events: Vec<AedOnlineEvent>,
}

pub struct AedOverlapSegmenter {
    handle: NonNull<sys::OmniAedOverlapSegmenterCtx>,
}

unsafe impl Send for AedOverlapSegmenter {}

impl AedOverlapSegmenter {
    pub fn from_bundle_path(path: impl AsRef<Path>, config: AedOverlapConfig) -> Result<Self> {
        let c_path = cstring_from_path(path)?;
        let cfg = sys::OmniAedOverlapConfig::from(config);
        let mut err = sys::OMNI_OK;
        let handle =
            unsafe { sys::omni_aed_overlap_segmenter_create(c_path.as_ptr(), &cfg, &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn from_bundle_bytes(data: &[u8], config: AedOverlapConfig) -> Result<Self> {
        let data_len = len_to_c_int(data.len(), "bundle data")?;
        let cfg = sys::OmniAedOverlapConfig::from(config);
        let mut err = sys::OMNI_OK;
        let handle = unsafe {
            sys::omni_aed_overlap_segmenter_create_from_buffer(
                data.as_ptr().cast::<c_void>(),
                data_len,
                &cfg,
                &mut err,
            )
        };
        Self::from_raw_handle(handle, err)
    }

    pub fn try_clone(&self) -> Result<Self> {
        let mut err = sys::OMNI_OK;
        let handle =
            unsafe { sys::omni_aed_overlap_segmenter_clone(self.handle.as_ptr(), &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn ingest_i16(&mut self, audio: &[i16]) -> Result<AedOverlapResult> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out = RawOverlapOutput::default();
        let ret = unsafe {
            sys::omni_aed_overlap_segmenter_ingest_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out.segments,
                &mut out.segment_count,
                &mut out.events,
                &mut out.event_count,
            )
        };
        collect_result(ret, out)
    }

    pub fn ingest_f32(&mut self, audio: &[f32]) -> Result<AedOverlapResult> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out = RawOverlapOutput::default();
        let ret = unsafe {
            sys::omni_aed_overlap_segmenter_ingest(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out.segments,
                &mut out.segment_count,
                &mut out.events,
                &mut out.event_count,
            )
        };
        collect_result(ret, out)
    }

    pub fn flush(&mut self) -> Result<AedOverlapResult> {
        let mut out = RawOverlapOutput::default();
        let ret = unsafe {
            sys::omni_aed_overlap_segmenter_flush(
                self.handle.as_ptr(),
                &mut out.segments,
                &mut out.segment_count,
                &mut out.events,
                &mut out.event_count,
            )
        };
        collect_result(ret, out)
    }

    pub fn reset(&mut self) {
        unsafe { sys::omni_aed_overlap_segmenter_reset(self.handle.as_ptr()) };
    }

    fn from_raw_handle(handle: sys::OmniAedOverlapSegmenterHandle, err: c_int) -> Result<Self> {
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or_else(|| Error::from_code(err))
    }
}

impl Drop for AedOverlapSegmenter {
    fn drop(&mut self) {
        unsafe { sys::omni_aed_overlap_segmenter_destroy(self.handle.as_ptr()) };
    }
}

#[derive(Default)]
struct RawOverlapOutput {
    segments: *mut sys::OmniAedOnlineSegment,
    segment_count: c_int,
    events: *mut sys::OmniAedOnlineEvent,
    event_count: c_int,
}

fn collect_result(ret: c_int, out: RawOverlapOutput) -> Result<AedOverlapResult> {
    let _guard = RawOverlapOutputGuard {
        segments: out.segments,
        events: out.events,
    };
    if ret != sys::OMNI_OK {
        return Err(Error::from_code(ret));
    }
    if out.segment_count < 0 || out.event_count < 0 {
        return Err(Error::invalid_argument("native output count is negative"));
    }

    let segments = copy_segments(out.segments, out.segment_count as usize)?;
    let events = copy_events(out.events, out.event_count as usize)?;
    Ok(AedOverlapResult { segments, events })
}

fn copy_segments(
    ptr: *const sys::OmniAedOnlineSegment,
    count: usize,
) -> Result<Vec<AedOnlineSegment>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(
            "native returned null segments with non-zero count",
        ));
    }
    let values = unsafe { slice::from_raw_parts(ptr, count) };
    Ok(values.iter().copied().map(AedOnlineSegment::from).collect())
}

fn copy_events(ptr: *const sys::OmniAedOnlineEvent, count: usize) -> Result<Vec<AedOnlineEvent>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(
            "native returned null events with non-zero count",
        ));
    }
    let values = unsafe { slice::from_raw_parts(ptr, count) };
    Ok(values.iter().copied().map(AedOnlineEvent::from).collect())
}

struct RawOverlapOutputGuard {
    segments: *mut sys::OmniAedOnlineSegment,
    events: *mut sys::OmniAedOnlineEvent,
}

impl Drop for RawOverlapOutputGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.segments.is_null() {
                sys::omni_free(self.segments.cast::<c_void>());
            }
            if !self.events.is_null() {
                sys::omni_free(self.events.cast::<c_void>());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_native_defaults() {
        let cfg = AedOverlapConfig::default();
        assert_eq!(cfg.hop_ms, 2000);
        assert_eq!(cfg.overlap_ms, 250);
        assert_eq!(cfg.hard_split_pause_ms, 2000);
        assert_eq!(cfg.max_chunk_ms, 60_000);
        assert_eq!(cfg.speech_threshold, 0.5);
    }

    #[test]
    fn event_kind_mapping_preserves_unknown_values() {
        assert_eq!(
            AedEventKind::from(sys::OMNI_AED_EVENT_SPEECH),
            AedEventKind::Speech
        );
        assert_eq!(AedEventKind::from(99), AedEventKind::Unknown(99));
    }

    #[test]
    fn kind_mask_contains_classes() {
        let mask = AedKindMask(AedKindMask::SPEECH.0 | AedKindMask::MUSIC.0);
        assert!(mask.contains(AedKindMask::SPEECH));
        assert!(mask.contains(AedKindMask::MUSIC));
        assert!(!mask.contains(AedKindMask::SINGING));
    }

    #[test]
    fn transcribable_masks_include_speech_and_singing() {
        assert!(AedKindMask::SPEECH.is_transcribable());
        assert!(AedKindMask::SINGING.is_transcribable());
        assert!(AedKindMask(AedKindMask::SPEECH.0 | AedKindMask::MUSIC.0).is_transcribable());
        assert!(AedKindMask(AedKindMask::SINGING.0 | AedKindMask::MUSIC.0).is_transcribable());
        assert!(!AedKindMask::MUSIC.is_transcribable());
        assert!(!AedKindMask(0).is_transcribable());
    }

    #[test]
    fn online_event_transcribable_uses_mask_not_primary_kind() {
        let event = AedOnlineEvent {
            start: 0.0,
            end: 1.0,
            primary_kind: AedEventKind::Mixed,
            kind_mask: AedKindMask(AedKindMask::SINGING.0 | AedKindMask::MUSIC.0),
            speech_confidence: 0.0,
            singing_confidence: 0.8,
            music_confidence: 0.9,
            confidence: 0.9,
        };

        assert!(event.is_transcribable());
    }
}
