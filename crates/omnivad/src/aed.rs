use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use omnivad_sys as sys;

use crate::common::{collect_native_array, NativeArrayGuard, PostConfig};
use crate::error::{cstring_from_path, len_to_c_int, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AedClass {
    Speech,
    Singing,
    Music,
    Unknown(i32),
}

impl From<i32> for AedClass {
    fn from(value: i32) -> Self {
        match value {
            sys::OMNI_AED_SPEECH => Self::Speech,
            sys::OMNI_AED_SINGING => Self::Singing,
            sys::OMNI_AED_MUSIC => Self::Music,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AedSegment {
    pub start: f32,
    pub end: f32,
    pub class: AedClass,
    pub confidence: f32,
}

impl From<sys::OmniAedSegment> for AedSegment {
    fn from(value: sys::OmniAedSegment) -> Self {
        Self {
            start: value.start,
            end: value.end,
            class: AedClass::from(value.cls),
            confidence: value.confidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AedPostConfig {
    pub speech: PostConfig,
    pub singing: PostConfig,
    pub music: PostConfig,
}

impl Default for AedPostConfig {
    fn default() -> Self {
        unsafe { sys::omni_aed_post_config_default() }.into()
    }
}

impl From<sys::OmniAedPostConfig> for AedPostConfig {
    fn from(value: sys::OmniAedPostConfig) -> Self {
        Self {
            speech: value.speech.into(),
            singing: value.singing.into(),
            music: value.music.into(),
        }
    }
}

impl From<AedPostConfig> for sys::OmniAedPostConfig {
    fn from(value: AedPostConfig) -> Self {
        Self {
            speech: value.speech.into(),
            singing: value.singing.into(),
            music: value.music.into(),
        }
    }
}

pub struct Aed {
    handle: NonNull<sys::OmniAedCtx>,
}

unsafe impl Send for Aed {}

impl Aed {
    pub fn from_bundle_path(path: impl AsRef<Path>) -> Result<Self> {
        let c_path = cstring_from_path(path)?;
        let mut err = sys::OMNI_OK;
        let handle = unsafe { sys::omni_aed_create(c_path.as_ptr(), &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn from_bundle_bytes(data: &[u8]) -> Result<Self> {
        let data_len = len_to_c_int(data.len(), "bundle data")?;
        let mut err = sys::OMNI_OK;
        let handle = unsafe {
            sys::omni_aed_create_from_buffer(data.as_ptr().cast::<c_void>(), data_len, &mut err)
        };
        Self::from_raw_handle(handle, err)
    }

    pub fn detect_i16(&self, audio: &[i16], config: AedPostConfig) -> Result<Vec<AedSegment>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let cfg = sys::OmniAedPostConfig::from(config);
        let mut out_segments = std::ptr::null_mut();
        let mut out_count: c_int = 0;
        let ret = unsafe {
            sys::omni_aed_detect_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &cfg,
                &mut out_segments,
                &mut out_count,
            )
        };
        collect_native_array(ret, out_segments, out_count, "segments", AedSegment::from)
    }

    pub fn detect_f32(&self, audio: &[f32], config: AedPostConfig) -> Result<Vec<AedSegment>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let cfg = sys::OmniAedPostConfig::from(config);
        let mut out_segments = std::ptr::null_mut();
        let mut out_count: c_int = 0;
        let ret = unsafe {
            sys::omni_aed_detect(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &cfg,
                &mut out_segments,
                &mut out_count,
            )
        };
        collect_native_array(ret, out_segments, out_count, "segments", AedSegment::from)
    }

    pub fn detect_probs_i16(&self, audio: &[i16]) -> Result<Vec<[f32; 3]>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_aed_detect_probs_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_prob_triplets(ret, out_probs, out_frames)
    }

    pub fn detect_probs_f32(&self, audio: &[f32]) -> Result<Vec<[f32; 3]>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_aed_detect_probs(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_prob_triplets(ret, out_probs, out_frames)
    }

    fn from_raw_handle(handle: sys::OmniAedHandle, err: c_int) -> Result<Self> {
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or_else(|| Error::from_code(err))
    }
}

impl Drop for Aed {
    fn drop(&mut self) {
        unsafe { sys::omni_aed_destroy(self.handle.as_ptr()) };
    }
}

fn collect_prob_triplets(ret: c_int, ptr: *mut f32, frames: c_int) -> Result<Vec<[f32; 3]>> {
    let _guard = NativeArrayGuard { ptr };
    crate::common::check_status(ret)?;
    if frames < 0 {
        return Err(Error::invalid_argument(
            "native returned negative probability frame count",
        ));
    }
    let frames = frames as usize;
    if frames == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(
            "native returned null probabilities with non-zero frame count",
        ));
    }
    let total = frames
        .checked_mul(3)
        .ok_or_else(|| Error::invalid_argument("probability frame count overflowed"))?;
    let values = unsafe { slice::from_raw_parts(ptr, total) };
    Ok(values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}
