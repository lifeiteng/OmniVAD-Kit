use omnivad::{merge_chunks, ChunkConfig, ChunkMode, Segment};

#[test]
fn default_config_matches_native_defaults() {
    let config = ChunkConfig::default();
    assert_eq!(config.max_chunk_secs, 30.0);
    assert!(config.max_gap_secs.is_infinite());
    assert_eq!(config.pad_onset_secs, 0.04);
    assert_eq!(config.pad_offset_secs, 0.04);
    assert_eq!(config.min_speech_secs, 0.0);
    assert_eq!(config.min_silence_secs, 0.20);
    assert_eq!(config.mode, ChunkMode::Greedy);
}

#[test]
fn empty_input_returns_no_chunks() {
    let chunks = merge_chunks(&[], ChunkConfig::default()).expect("merge empty input");
    assert!(chunks.is_empty());
}

#[test]
fn greedy_mode_packs_until_max_duration() {
    let config = ChunkConfig {
        max_chunk_secs: 10.0,
        max_gap_secs: f32::INFINITY,
        pad_onset_secs: 0.0,
        pad_offset_secs: 0.0,
        min_speech_secs: 0.0,
        min_silence_secs: 0.0,
        mode: ChunkMode::Greedy,
    };
    let segments = [
        Segment {
            start: 0.0,
            end: 4.0,
        },
        Segment {
            start: 4.5,
            end: 8.0,
        },
        Segment {
            start: 8.2,
            end: 11.0,
        },
    ];

    let chunks = merge_chunks(&segments, config).expect("merge chunks");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].start, 0.0);
    assert_eq!(chunks[0].end, 8.0);
    assert_eq!(chunks[0].segment_start_index, 0);
    assert_eq!(chunks[0].segment_count, 2);
    assert_eq!(chunks[1].start, 8.2);
    assert_eq!(chunks[1].end, 11.0);
}

#[test]
fn longest_gap_mode_splits_on_large_internal_gap() {
    let config = ChunkConfig {
        max_chunk_secs: 30.0,
        max_gap_secs: 5.0,
        pad_onset_secs: 0.0,
        pad_offset_secs: 0.0,
        min_speech_secs: 0.0,
        min_silence_secs: 0.0,
        mode: ChunkMode::LongestGap,
    };
    let segments = [
        Segment {
            start: 0.0,
            end: 3.0,
        },
        Segment {
            start: 4.0,
            end: 6.0,
        },
        Segment {
            start: 20.0,
            end: 22.0,
        },
    ];

    let chunks = merge_chunks(&segments, config).expect("merge chunks");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].start, 0.0);
    assert_eq!(chunks[0].end, 6.0);
    assert_eq!(chunks[1].start, 20.0);
    assert_eq!(chunks[1].end, 22.0);
}

#[test]
fn invalid_config_is_rejected() {
    let config = ChunkConfig {
        max_chunk_secs: 0.0,
        ..ChunkConfig::default()
    };
    let error = merge_chunks(
        &[Segment {
            start: 0.0,
            end: 1.0,
        }],
        config,
    )
    .expect_err("invalid chunk config should fail");
    assert_eq!(error.code, omnivad_sys::OMNI_ERR_INVALID_ARG);
}
