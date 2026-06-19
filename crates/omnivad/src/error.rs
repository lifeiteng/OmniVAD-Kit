use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_int;
use std::path::Path;

use omnivad_sys as sys;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: c_int,
    pub message: String,
}

impl Error {
    pub(crate) fn from_code(code: c_int) -> Self {
        let message = unsafe {
            let ptr = sys::omni_error_string(code);
            if ptr.is_null() {
                format!("OmniVAD error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Self { code, message }
    }

    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            code: sys::OMNI_ERR_INVALID_ARG,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OmniVAD error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

pub(crate) fn cstring_from_path(path: impl AsRef<Path>) -> Result<CString> {
    path_to_cstring(path.as_ref())
        .map_err(|_| Error::invalid_argument("bundle path contains an interior NUL byte"))
}

#[cfg(unix)]
fn path_to_cstring(path: &Path) -> std::result::Result<CString, std::ffi::NulError> {
    use std::os::unix::ffi::OsStrExt;

    CString::new(path.as_os_str().as_bytes())
}

#[cfg(not(unix))]
fn path_to_cstring(path: &Path) -> std::result::Result<CString, std::ffi::NulError> {
    let path = path.to_string_lossy();
    CString::new(path.as_bytes())
}

pub(crate) fn len_to_c_int(len: usize, label: &str) -> Result<c_int> {
    if len > c_int::MAX as usize {
        return Err(Error::invalid_argument(format!(
            "{label} is too large for C ABI"
        )));
    }
    Ok(len as c_int)
}
