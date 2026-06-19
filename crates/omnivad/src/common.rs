use std::os::raw::{c_int, c_void};
use std::slice;

use omnivad_sys as sys;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
}

impl From<sys::OmniSegment> for Segment {
    fn from(value: sys::OmniSegment) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl From<Segment> for sys::OmniSegment {
    fn from(value: Segment) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostConfig {
    pub threshold: f32,
    pub smooth_window_size: i32,
    pub min_speech_frames: i32,
    pub min_silence_frames: i32,
    pub max_speech_frames: i32,
    pub merge_silence_frames: i32,
    pub extend_speech_frames: i32,
}

impl Default for PostConfig {
    fn default() -> Self {
        unsafe { sys::omni_post_config_default() }.into()
    }
}

impl From<sys::OmniPostConfig> for PostConfig {
    fn from(value: sys::OmniPostConfig) -> Self {
        Self {
            threshold: value.threshold,
            smooth_window_size: value.smooth_window_size,
            min_speech_frames: value.min_speech_frames,
            min_silence_frames: value.min_silence_frames,
            max_speech_frames: value.max_speech_frames,
            merge_silence_frames: value.merge_silence_frames,
            extend_speech_frames: value.extend_speech_frames,
        }
    }
}

impl From<PostConfig> for sys::OmniPostConfig {
    fn from(value: PostConfig) -> Self {
        Self {
            threshold: value.threshold,
            smooth_window_size: value.smooth_window_size,
            min_speech_frames: value.min_speech_frames,
            min_silence_frames: value.min_silence_frames,
            max_speech_frames: value.max_speech_frames,
            merge_silence_frames: value.merge_silence_frames,
            extend_speech_frames: value.extend_speech_frames,
        }
    }
}

pub(crate) fn check_status(ret: c_int) -> Result<()> {
    if ret == sys::OMNI_OK {
        Ok(())
    } else {
        Err(Error::from_code(ret))
    }
}

pub(crate) fn collect_native_array<T, U>(
    ret: c_int,
    ptr: *mut T,
    count: c_int,
    label: &str,
    map: impl Fn(T) -> U,
) -> Result<Vec<U>>
where
    T: Copy,
{
    let _guard = NativeArrayGuard { ptr };
    check_status(ret)?;
    if count < 0 {
        return Err(Error::invalid_argument(format!(
            "native returned negative {label} count"
        )));
    }
    let count = count as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(Error::invalid_argument(format!(
            "native returned null {label} with non-zero count"
        )));
    }
    let values = unsafe { slice::from_raw_parts(ptr, count) };
    Ok(values.iter().copied().map(map).collect())
}

pub(crate) fn collect_native_f32_array(
    ret: c_int,
    ptr: *mut f32,
    count: c_int,
    label: &str,
) -> Result<Vec<f32>> {
    collect_native_array(ret, ptr, count, label, |value| value)
}

pub(crate) struct NativeArrayGuard<T> {
    pub ptr: *mut T,
}

impl<T> Drop for NativeArrayGuard<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::omni_free(self.ptr.cast::<c_void>());
            }
        }
    }
}
