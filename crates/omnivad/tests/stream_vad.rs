use std::path::{Path, PathBuf};

use omnivad::{StreamVad, StreamVadConfig};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate should be inside repo/crates/omnivad")
        .to_path_buf()
}

fn stream_vad_model_path() -> PathBuf {
    repo_root().join("models/stream-vad.omnivad")
}

#[test]
fn tiny_chunks_buffer_without_error() {
    let mut vad = StreamVad::from_bundle_path(stream_vad_model_path(), StreamVadConfig::default())
        .expect("create Stream VAD");

    for samples in [0_usize, 1, 10, 100, 159] {
        let result = vad
            .process_i16(&vec![0; samples])
            .expect("tiny chunk should only buffer");
        assert!(result.is_none());
    }
}

#[test]
fn exact_chunks_eventually_emit_frames() {
    let mut vad = StreamVad::from_bundle_path(stream_vad_model_path(), StreamVadConfig::default())
        .expect("create Stream VAD");
    let chunk = vec![0; 160];
    let mut result = None;

    for _ in 0..20 {
        result = vad.process_i16(&chunk).expect("process exact chunk");
    }

    let frame = result.expect("stream VAD should emit after enough context");
    assert!(frame.frame_idx > 0);
    assert_eq!(vad.frame_offset(), frame.frame_idx);
}

#[test]
fn clone_has_fresh_runtime_state() {
    let mut original =
        StreamVad::from_bundle_path(stream_vad_model_path(), StreamVadConfig::default())
            .expect("create Stream VAD");
    for _ in 0..20 {
        let _ = original.process_i16(&vec![0; 160]).expect("process chunk");
    }
    assert!(original.frame_offset() > 0);

    let cloned = original.try_clone().expect("clone Stream VAD");
    assert_eq!(cloned.frame_offset(), 0);
}

#[test]
fn detect_full_returns_probabilities_for_silence() {
    let vad = StreamVad::from_bundle_path(stream_vad_model_path(), StreamVadConfig::default())
        .expect("create Stream VAD");
    let probs = vad
        .detect_full_f32(&vec![0.0; 16_000])
        .expect("detect full probabilities");
    assert!(!probs.is_empty());
}

#[test]
fn invalid_config_is_rejected() {
    let config = StreamVadConfig {
        threshold: 1.5,
        ..StreamVadConfig::default()
    };
    let error = match StreamVad::from_bundle_path(stream_vad_model_path(), config) {
        Ok(_) => panic!("invalid config should fail"),
        Err(error) => error,
    };
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_INVALID_ARG);
}
