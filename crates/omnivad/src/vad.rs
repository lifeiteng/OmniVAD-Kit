use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

use omnivad_sys as sys;

use crate::common::{collect_native_array, collect_native_f32_array, PostConfig, Segment};
use crate::error::{cstring_from_path, len_to_c_int, Error, Result};

pub struct Vad {
    handle: NonNull<sys::OmniVadCtx>,
}

unsafe impl Send for Vad {}

impl Vad {
    pub fn from_bundle_path(path: impl AsRef<Path>) -> Result<Self> {
        let c_path = cstring_from_path(path)?;
        let mut err = sys::OMNI_OK;
        let handle = unsafe { sys::omni_vad_create(c_path.as_ptr(), &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn from_bundle_bytes(data: &[u8]) -> Result<Self> {
        let data_len = len_to_c_int(data.len(), "bundle data")?;
        let mut err = sys::OMNI_OK;
        let handle = unsafe {
            sys::omni_vad_create_from_buffer(data.as_ptr().cast::<c_void>(), data_len, &mut err)
        };
        Self::from_raw_handle(handle, err)
    }

    pub fn detect_i16(&self, audio: &[i16], config: PostConfig) -> Result<Vec<Segment>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let cfg = sys::OmniPostConfig::from(config);
        let mut out_segments = std::ptr::null_mut();
        let mut out_count: c_int = 0;
        let ret = unsafe {
            sys::omni_vad_detect_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &cfg,
                &mut out_segments,
                &mut out_count,
            )
        };
        collect_native_array(ret, out_segments, out_count, "segments", Segment::from)
    }

    pub fn detect_f32(&self, audio: &[f32], config: PostConfig) -> Result<Vec<Segment>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let cfg = sys::OmniPostConfig::from(config);
        let mut out_segments = std::ptr::null_mut();
        let mut out_count: c_int = 0;
        let ret = unsafe {
            sys::omni_vad_detect(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &cfg,
                &mut out_segments,
                &mut out_count,
            )
        };
        collect_native_array(ret, out_segments, out_count, "segments", Segment::from)
    }

    pub fn detect_probs_i16(&self, audio: &[i16]) -> Result<Vec<f32>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_vad_detect_probs_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_native_f32_array(ret, out_probs, out_frames, "probabilities")
    }

    pub fn detect_probs_f32(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_vad_detect_probs(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_native_f32_array(ret, out_probs, out_frames, "probabilities")
    }

    fn from_raw_handle(handle: sys::OmniVadHandle, err: c_int) -> Result<Self> {
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or_else(|| Error::from_code(err))
    }
}

impl Drop for Vad {
    fn drop(&mut self) {
        unsafe { sys::omni_vad_destroy(self.handle.as_ptr()) };
    }
}
