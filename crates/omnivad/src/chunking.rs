use std::os::raw::c_int;

use omnivad_sys as sys;

use crate::common::{collect_native_array, Segment};
use crate::error::{len_to_c_int, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    Greedy,
    LongestGap,
    Unknown(i32),
}

impl From<i32> for ChunkMode {
    fn from(value: i32) -> Self {
        match value {
            sys::OMNI_CHUNK_GREEDY => Self::Greedy,
            sys::OMNI_CHUNK_LONGEST_GAP => Self::LongestGap,
            other => Self::Unknown(other),
        }
    }
}

impl From<ChunkMode> for i32 {
    fn from(value: ChunkMode) -> Self {
        match value {
            ChunkMode::Greedy => sys::OMNI_CHUNK_GREEDY,
            ChunkMode::LongestGap => sys::OMNI_CHUNK_LONGEST_GAP,
            ChunkMode::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkConfig {
    pub max_chunk_secs: f32,
    pub max_gap_secs: f32,
    pub pad_onset_secs: f32,
    pub pad_offset_secs: f32,
    pub min_speech_secs: f32,
    pub min_silence_secs: f32,
    pub mode: ChunkMode,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        unsafe { sys::omni_chunk_config_default() }.into()
    }
}

impl From<sys::OmniChunkConfig> for ChunkConfig {
    fn from(value: sys::OmniChunkConfig) -> Self {
        Self {
            max_chunk_secs: value.max_chunk_secs,
            max_gap_secs: value.max_gap_secs,
            pad_onset_secs: value.pad_onset_secs,
            pad_offset_secs: value.pad_offset_secs,
            min_speech_secs: value.min_speech_secs,
            min_silence_secs: value.min_silence_secs,
            mode: ChunkMode::from(value.mode),
        }
    }
}

impl From<ChunkConfig> for sys::OmniChunkConfig {
    fn from(value: ChunkConfig) -> Self {
        Self {
            max_chunk_secs: value.max_chunk_secs,
            max_gap_secs: value.max_gap_secs,
            pad_onset_secs: value.pad_onset_secs,
            pad_offset_secs: value.pad_offset_secs,
            min_speech_secs: value.min_speech_secs,
            min_silence_secs: value.min_silence_secs,
            mode: value.mode.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    pub start: f32,
    pub end: f32,
    pub segment_start_index: i32,
    pub segment_count: i32,
}

impl From<sys::OmniChunk> for Chunk {
    fn from(value: sys::OmniChunk) -> Self {
        Self {
            start: value.start,
            end: value.end,
            segment_start_index: value.seg_start_idx,
            segment_count: value.seg_count,
        }
    }
}

pub fn merge_chunks(segments: &[Segment], config: ChunkConfig) -> Result<Vec<Chunk>> {
    let num_segments = len_to_c_int(segments.len(), "segments")?;
    let native_segments: Vec<sys::OmniSegment> = segments.iter().copied().map(Into::into).collect();
    let segments_ptr = if native_segments.is_empty() {
        std::ptr::null()
    } else {
        native_segments.as_ptr()
    };
    let cfg = sys::OmniChunkConfig::from(config);
    let mut out_chunks = std::ptr::null_mut();
    let mut out_count: c_int = 0;
    let ret = unsafe {
        sys::omni_merge_chunks(
            segments_ptr,
            num_segments,
            &cfg,
            &mut out_chunks,
            &mut out_count,
        )
    };
    collect_native_array(ret, out_chunks, out_count, "chunks", Chunk::from)
}
