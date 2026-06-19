//! Safe Rust bindings for OmniVAD.
//!
//! The Rust API exposes safe wrappers for the native VAD, streaming VAD, AED,
//! AED overlap segmenter, and chunking APIs while keeping the C ABI behind an
//! owned Rust boundary.
//!
//! The native `libomnivad` shared library must be available to the linker and
//! runtime loader. During local development, set `OMNIVAD_LIB_DIR` for linking
//! and add the same directory to the platform runtime library path when running
//! tests or binaries.
//!
//! ```no_run
//! use omnivad::{PostConfig, Vad};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let vad = Vad::from_bundle_path("models/vad.omnivad")?;
//! let segments = vad.detect_i16(&vec![0; 16_000], PostConfig::default())?;
//! assert!(segments.is_empty());
//! # Ok(())
//! # }
//! ```

pub mod aed;
pub mod aed_overlap;
pub mod chunking;
pub mod stream_vad;
pub mod vad;

mod common;
mod error;

pub use aed::{Aed, AedClass, AedPostConfig, AedSegment};
pub use aed_overlap::{
    AedEventKind, AedKindMask, AedOnlineEvent, AedOnlineSegment, AedOverlapConfig,
    AedOverlapResult, AedOverlapSegmenter,
};
pub use chunking::{merge_chunks, Chunk, ChunkConfig, ChunkMode};
pub use common::{PostConfig, Segment};
pub use error::{Error, Result};
pub use stream_vad::{StreamVad, StreamVadConfig, StreamVadFrame};
pub use vad::Vad;

pub type OmniVad = Vad;
pub type OmniStreamVad = StreamVad;
pub type OmniAed = Aed;
