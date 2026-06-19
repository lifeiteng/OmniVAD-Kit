use std::mem::MaybeUninit;
use std::os::raw::{c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

use omnivad_sys as sys;

use crate::common::collect_native_f32_array;
use crate::error::{cstring_from_path, len_to_c_int, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamVadConfig {
    pub threshold: f32,
    pub smooth_window_size: i32,
    pub pad_start_frame: i32,
    pub min_speech_frame: i32,
    pub max_speech_frame: i32,
    pub min_silence_frame: i32,
}

impl Default for StreamVadConfig {
    fn default() -> Self {
        unsafe { sys::omni_stream_vad_config_default() }.into()
    }
}

impl From<sys::OmniStreamVadConfig> for StreamVadConfig {
    fn from(value: sys::OmniStreamVadConfig) -> Self {
        Self {
            threshold: value.threshold,
            smooth_window_size: value.smooth_window_size,
            pad_start_frame: value.pad_start_frame,
            min_speech_frame: value.min_speech_frame,
            max_speech_frame: value.max_speech_frame,
            min_silence_frame: value.min_silence_frame,
        }
    }
}

impl From<StreamVadConfig> for sys::OmniStreamVadConfig {
    fn from(value: StreamVadConfig) -> Self {
        Self {
            threshold: value.threshold,
            smooth_window_size: value.smooth_window_size,
            pad_start_frame: value.pad_start_frame,
            min_speech_frame: value.min_speech_frame,
            max_speech_frame: value.max_speech_frame,
            min_silence_frame: value.min_silence_frame,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamVadFrame {
    pub confidence: f32,
    pub smoothed_prob: f32,
    pub is_speech: bool,
    pub is_speech_start: bool,
    pub is_speech_end: bool,
    pub frame_idx: i32,
    pub speech_start_frame: i32,
    pub speech_end_frame: i32,
}

impl From<sys::OmniStreamVadResult> for StreamVadFrame {
    fn from(value: sys::OmniStreamVadResult) -> Self {
        Self {
            confidence: value.confidence,
            smoothed_prob: value.smoothed_prob,
            is_speech: value.is_speech != 0,
            is_speech_start: value.is_speech_start != 0,
            is_speech_end: value.is_speech_end != 0,
            frame_idx: value.frame_idx,
            speech_start_frame: value.speech_start_frame,
            speech_end_frame: value.speech_end_frame,
        }
    }
}

pub struct StreamVad {
    handle: NonNull<sys::OmniStreamVadCtx>,
}

unsafe impl Send for StreamVad {}

impl StreamVad {
    pub fn from_bundle_path(path: impl AsRef<Path>, config: StreamVadConfig) -> Result<Self> {
        let c_path = cstring_from_path(path)?;
        let cfg = sys::OmniStreamVadConfig::from(config);
        let mut err = sys::OMNI_OK;
        let handle = unsafe { sys::omni_stream_vad_create(c_path.as_ptr(), &cfg, &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn from_bundle_bytes(data: &[u8], config: StreamVadConfig) -> Result<Self> {
        let data_len = len_to_c_int(data.len(), "bundle data")?;
        let cfg = sys::OmniStreamVadConfig::from(config);
        let mut err = sys::OMNI_OK;
        let handle = unsafe {
            sys::omni_stream_vad_create_from_buffer(
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
        let handle = unsafe { sys::omni_stream_vad_clone(self.handle.as_ptr(), &mut err) };
        Self::from_raw_handle(handle, err)
    }

    pub fn process_i16(&mut self, audio: &[i16]) -> Result<Option<StreamVadFrame>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut result = MaybeUninit::<sys::OmniStreamVadResult>::uninit();
        let ret = unsafe {
            sys::omni_stream_vad_process_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                result.as_mut_ptr(),
            )
        };
        collect_process_result(ret, result)
    }

    pub fn process_f32(&mut self, audio: &[f32]) -> Result<Option<StreamVadFrame>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut result = MaybeUninit::<sys::OmniStreamVadResult>::uninit();
        let ret = unsafe {
            sys::omni_stream_vad_process(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                result.as_mut_ptr(),
            )
        };
        collect_process_result(ret, result)
    }

    pub fn detect_full_i16(&self, audio: &[i16]) -> Result<Vec<f32>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_stream_vad_detect_full_int16(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_native_f32_array(ret, out_probs, out_frames, "probabilities")
    }

    pub fn detect_full_f32(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let num_samples = len_to_c_int(audio.len(), "audio")?;
        let mut out_probs = std::ptr::null_mut();
        let mut out_frames: c_int = 0;
        let ret = unsafe {
            sys::omni_stream_vad_detect_full(
                self.handle.as_ptr(),
                audio.as_ptr(),
                num_samples,
                &mut out_probs,
                &mut out_frames,
            )
        };
        collect_native_f32_array(ret, out_probs, out_frames, "probabilities")
    }

    pub fn reset(&mut self) {
        unsafe { sys::omni_stream_vad_reset(self.handle.as_ptr()) };
    }

    pub fn frame_offset(&self) -> i32 {
        unsafe { sys::omni_stream_vad_get_frame_offset(self.handle.as_ptr()) }
    }

    fn from_raw_handle(handle: sys::OmniStreamVadHandle, err: c_int) -> Result<Self> {
        NonNull::new(handle)
            .map(|handle| Self { handle })
            .ok_or_else(|| Error::from_code(err))
    }
}

impl Drop for StreamVad {
    fn drop(&mut self) {
        unsafe { sys::omni_stream_vad_destroy(self.handle.as_ptr()) };
    }
}

fn collect_process_result(
    ret: c_int,
    result: MaybeUninit<sys::OmniStreamVadResult>,
) -> Result<Option<StreamVadFrame>> {
    match ret {
        sys::OMNI_OK => Ok(Some(unsafe { result.assume_init() }.into())),
        sys::OMNI_ERR_NO_FRAMES => Ok(None),
        other => Err(Error::from_code(other)),
    }
}
