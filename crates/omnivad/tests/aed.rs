use std::path::{Path, PathBuf};

use omnivad::{Aed, AedClass, AedPostConfig};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate should be inside repo/crates/omnivad")
        .to_path_buf()
}

fn aed_model_path() -> PathBuf {
    repo_root().join("models/aed.omnivad")
}

#[test]
fn creates_from_bundle_path_and_detects_silence() {
    let aed = Aed::from_bundle_path(aed_model_path()).expect("create AED");
    let segments = aed
        .detect_i16(&vec![0; 16_000], AedPostConfig::default())
        .expect("detect one second of silence");
    assert!(segments.is_empty());
}

#[test]
fn creates_from_bundle_bytes() {
    let bytes = std::fs::read(aed_model_path()).expect("read AED model bundle");
    let aed = Aed::from_bundle_bytes(&bytes).expect("create AED from in-memory model bundle");
    let probs = aed
        .detect_probs_f32(&vec![0.0; 600])
        .expect("detect probabilities");
    assert!(!probs.is_empty());
    assert!(probs.iter().all(|frame| frame.len() == 3));
}

#[test]
fn rejects_sub_frame_audio() {
    let aed = Aed::from_bundle_path(aed_model_path()).expect("create AED");
    let error = aed
        .detect_f32(&vec![0.0; 120], AedPostConfig::default())
        .expect_err("sub-frame audio should fail");
    assert_ne!(error.code, omnivad_sys::OMNI_OK);
}

#[test]
fn class_mapping_preserves_unknown_values() {
    assert_eq!(AedClass::from(0), AedClass::Speech);
    assert_eq!(AedClass::from(1), AedClass::Singing);
    assert_eq!(AedClass::from(2), AedClass::Music);
    assert_eq!(AedClass::from(42), AedClass::Unknown(42));
}
