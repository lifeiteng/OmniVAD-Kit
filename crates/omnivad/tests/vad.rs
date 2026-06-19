use std::path::{Path, PathBuf};

use omnivad::{PostConfig, Vad};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate should be inside repo/crates/omnivad")
        .to_path_buf()
}

fn vad_model_path() -> PathBuf {
    repo_root().join("models/vad.omnivad")
}

#[test]
fn creates_from_bundle_path_and_detects_silence() {
    let vad = Vad::from_bundle_path(vad_model_path()).expect("create VAD");
    let segments = vad
        .detect_i16(&vec![0; 16_000], PostConfig::default())
        .expect("detect one second of silence");
    assert!(segments.is_empty());
}

#[test]
fn creates_from_bundle_bytes() {
    let bytes = std::fs::read(vad_model_path()).expect("read VAD model bundle");
    let vad = Vad::from_bundle_bytes(&bytes).expect("create VAD from in-memory model bundle");
    let probs = vad
        .detect_probs_f32(&vec![0.0; 600])
        .expect("detect probabilities");
    assert!(!probs.is_empty());
}

#[test]
fn rejects_sub_frame_audio() {
    let vad = Vad::from_bundle_path(vad_model_path()).expect("create VAD");
    let error = vad
        .detect_f32(&vec![0.0; 120], PostConfig::default())
        .expect_err("sub-frame audio should fail");
    assert_ne!(error.code, omnivad_sys::OMNI_OK);
}

#[test]
fn missing_bundle_reports_load_error() {
    let error = match Vad::from_bundle_path(repo_root().join("models/missing-vad.omnivad")) {
        Ok(_) => panic!("missing bundle should fail"),
        Err(error) => error,
    };
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_LOAD_BUNDLE);
}
