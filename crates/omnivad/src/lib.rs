//! Safe Rust bindings for OmniVAD.
//!
//! The current Rust API exposes the AED overlap segmenter. It is intended for
//! downstream applications that need chunk-level pseudo-streaming event
//! detection while keeping the native C ABI behind a safe ownership boundary.
//!
//! The native `libomnivad` shared library must be available to the linker and
//! runtime loader. During local development, set `OMNIVAD_LIB_DIR` for linking
//! and add the same directory to the platform runtime library path when running
//! tests or binaries.
//!
//! ```no_run
//! use omnivad::{AedOverlapConfig, AedOverlapSegmenter};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut segmenter =
//!     AedOverlapSegmenter::from_bundle_path("models/aed.omnivad", AedOverlapConfig::default())?;
//! let result = segmenter.ingest_i16(&vec![0; 16_000])?;
//! let final_result = segmenter.flush()?;
//! assert!(result.segments.is_empty() || final_result.segments.is_empty());
//! # Ok(())
//! # }
//! ```

pub mod aed_overlap;

pub use aed_overlap::{
    AedEventKind, AedKindMask, AedOnlineEvent, AedOnlineSegment, AedOverlapConfig,
    AedOverlapResult, AedOverlapSegmenter, Error, Result,
};
