use std::path::{Path, PathBuf};

use omnivad::{AedEventKind, AedKindMask, AedOverlapConfig, AedOverlapSegmenter, Error};

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

fn fast_config() -> AedOverlapConfig {
    AedOverlapConfig {
        hop_ms: 500,
        overlap_ms: 100,
        hard_split_pause_ms: 200,
        max_chunk_ms: 2_000,
        ..AedOverlapConfig::default()
    }
}

#[test]
fn creates_from_bundle_path_and_processes_silence() {
    let mut segmenter = AedOverlapSegmenter::from_bundle_path(aed_model_path(), fast_config())
        .expect("create AED overlap segmenter");

    let first = segmenter
        .ingest_i16(&vec![0; 1_600])
        .expect("ingest silence");
    assert!(first.segments.is_empty());
    assert!(first.events.is_empty());

    let final_result = segmenter.flush().expect("flush silence");
    assert!(final_result.segments.is_empty());
    assert!(final_result.events.is_empty());
}

#[test]
fn creates_from_bundle_bytes() {
    let bytes = std::fs::read(aed_model_path()).expect("read AED model bundle");
    let mut segmenter = AedOverlapSegmenter::from_bundle_bytes(&bytes, fast_config())
        .expect("create from in-memory model bundle");
    let result = segmenter
        .ingest_f32(&vec![0.0; 8_000])
        .expect("ingest f32 silence");
    assert!(result.segments.is_empty());
    assert!(result.events.is_empty());
}

#[test]
fn clone_has_fresh_state() {
    let mut original = AedOverlapSegmenter::from_bundle_path(aed_model_path(), fast_config())
        .expect("create AED overlap segmenter");
    let mut cloned = original.try_clone().expect("clone segmenter");

    assert!(original
        .ingest_i16(&vec![0; 800])
        .unwrap()
        .segments
        .is_empty());
    assert!(cloned
        .ingest_i16(&vec![0; 800])
        .unwrap()
        .segments
        .is_empty());
    assert!(original.flush().unwrap().segments.is_empty());
    assert!(cloned.flush().unwrap().segments.is_empty());
}

#[test]
fn invalid_config_is_rejected() {
    let config = AedOverlapConfig {
        hop_ms: 500,
        overlap_ms: 500,
        ..AedOverlapConfig::default()
    };
    let error = match AedOverlapSegmenter::from_bundle_path(aed_model_path(), config) {
        Ok(_) => panic!("invalid config should fail"),
        Err(error) => error,
    };
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_INVALID_ARG);
}

#[test]
fn ingest_after_flush_is_rejected() {
    let mut segmenter = AedOverlapSegmenter::from_bundle_path(aed_model_path(), fast_config())
        .expect("create AED overlap segmenter");
    segmenter.flush().expect("initial flush");

    let error = segmenter
        .ingest_i16(&vec![0; 1_600])
        .expect_err("ingest after flush should fail");
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_INVALID_ARG);
}

#[test]
fn missing_bundle_reports_load_error() {
    let error = match AedOverlapSegmenter::from_bundle_path(
        repo_root().join("models/missing-aed.omnivad"),
        AedOverlapConfig::default(),
    ) {
        Ok(_) => panic!("missing bundle should fail"),
        Err(error) => error,
    };
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_LOAD_BUNDLE);
}

#[test]
fn event_kind_and_mask_helpers_are_stable() {
    assert_eq!(AedEventKind::from(1), AedEventKind::Speech);
    assert_eq!(AedEventKind::from(42), AedEventKind::Unknown(42));

    let mask = AedKindMask(AedKindMask::SPEECH.0 | AedKindMask::MUSIC.0);
    assert!(mask.contains(AedKindMask::SPEECH));
    assert!(mask.contains(AedKindMask::MUSIC));
    assert!(!mask.contains(AedKindMask::SINGING));
}

#[test]
fn native_default_matches_safe_default() {
    let native = unsafe { omnivad_sys::omni_aed_overlap_config_default() };
    let safe = AedOverlapConfig::default();
    assert_eq!(native.hop_ms, safe.hop_ms);
    assert_eq!(native.overlap_ms, safe.overlap_ms);
    assert_eq!(native.edge_guard_ms, safe.edge_guard_ms);
    assert_eq!(native.hard_split_pause_ms, safe.hard_split_pause_ms);
    assert_eq!(native.max_chunk_ms, safe.max_chunk_ms);
    assert_eq!(native.min_speech_ms, safe.min_speech_ms);
    assert_eq!(native.merge_gap_ms, safe.merge_gap_ms);
    assert_eq!(native.music_gap_tolerance_ms, safe.music_gap_tolerance_ms);
    assert_eq!(native.pad_start_ms, safe.pad_start_ms);
    assert_eq!(native.pad_end_ms, safe.pad_end_ms);
    assert_eq!(native.speech_threshold, safe.speech_threshold);
    assert_eq!(native.singing_threshold, safe.singing_threshold);
    assert_eq!(native.music_threshold, safe.music_threshold);
}

#[test]
fn error_display_includes_code_and_message() {
    let error = Error {
        code: omnivad_sys::OMNI_ERR_INVALID_ARG,
        message: "invalid argument".to_string(),
    };
    let rendered = error.to_string();
    assert!(rendered.contains("-10"));
    assert!(rendered.contains("invalid argument"));
}
