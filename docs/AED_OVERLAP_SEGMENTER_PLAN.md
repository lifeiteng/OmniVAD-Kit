# AED Overlap Segmenter Plan

Date: 2026-06-18

This document plans a reusable AED-first pseudo-streaming segmenter for
OmniVAD-Kit. The goal is to ship the segmentation logic as a standalone
OmniVAD-Kit capability that can be reused by any downstream application.

## 1. Problem Statement

OmniVAD-Kit currently exposes three model modes:

- `vad.omnivad`: whole-audio speech segmentation.
- `stream-vad.omnivad`: true frame-by-frame speech VAD with model cache.
- `aed.omnivad`: whole-audio audio event detection with speech, singing, and
  music classes.

The AED model is not a true streaming model. It exposes whole-window event and
probability APIs, but no stateful `process()` API. For file import pipelines,
we still need incremental VAD-like output so downstream consumers can process
confirmed chunks before the whole file is decoded.

The segmenter should convert AED whole-window inference into stable,
monotonic, chunk-level pseudo-streaming events.

## 2. Non-Goals

- Do not replace the true `stream-vad.omnivad` model.
- Do not hide user-controlled split settings behind hardcoded caps.
- Do not implement downstream application UI or pipeline logic in OmniVAD-Kit.
- Do not treat AED as a simple binary VAD too early.
- Do not rely on whole-file rescans for long audio.

## 3. Proposed Design

### 3.1 Naming

Use the name `AedOverlapSegmenter`.

Avoid names like `BatchVAD`, `LookaheadVad`, or `StreamAED`. The algorithm is
not batch VAD, and it is not true streaming AED. It is overlapped chunk AED
inference with monotonic event commitment.

### 3.2 Default Runtime Shape

Initial candidate values:

```text
hop_ms = 2000
overlap_ms = 250
aed_input_ms = 2250
frame_shift_ms = 10
edge_guard_ms = 0-100
```

AED windows:

```text
[0ms, 2250ms]
[2000ms, 4250ms]
[4000ms, 6250ms]
...
```

The overlap provides warm-up context for the AED model and stabilizes chunk
boundaries. The final default must be selected by comparing windowed AED
probabilities against full-audio AED probabilities on real fixtures. It must
not change the semantics of user split settings such as max silence or max
chunk length.

### 3.3 Internal Pipeline

```text
PCM chunks
  -> AedWindowRunner
  -> ProbMerger
  -> EventExtractor
  -> EventMerger
  -> TranscribablePolicy
  -> SegmentStateMachine
  -> confirmed transcribable chunks
```

Responsibilities:

- `AedWindowRunner`: runs `aed.detect_probs` on bounded `hop + overlap`
  windows.
- `ProbMerger`: merges overlapped frame probabilities with edge-aware weights.
- `EventExtractor`: preserves typed AED events instead of collapsing into a
  binary speech mask immediately.
- `EventMerger`: removes short spikes, merges short gaps, and resolves adjacent
  speech, singing, music, and mixed regions.
- `TranscribablePolicy`: decides which event types are eligible for downstream
  transcription.
- `SegmentStateMachine`: applies final split rules using user-configured pause
  and max chunk values.

### 3.4 Event Model

```text
AedFrameProb
  time_ms: int64
  speech: float
  singing: float
  music: float
  weight: float
  coverage: int

AedEvent
  start_ms: int64
  end_ms: int64
  primary_kind: Speech | Singing | Music | Mixed | Silence
  kind_mask: bitmask(Speech, Singing, Music)
  speech_confidence: float
  singing_confidence: float
  music_confidence: float
  confidence: float
  coverage: int

AedSegment
  start_ms: int64
  end_ms: int64
  source_event_start_idx: int
  source_event_count: int
```

Event policy defaults:

- `Speech`: transcribable.
- `Singing`: transcribable by default, but labeled.
- `Mixed`: transcribable if speech or singing exceeds threshold. The original
  per-class probabilities must remain available.
- `Music`: not transcribable unless mixed with speech or singing.
- `Silence`: never transcribable; only drives pause splitting.

`primary_kind` is a presentation field. The state machine must retain
`kind_mask` and the three class confidences so policy changes do not require
rerunning AED. Mixed event detection is threshold based:

```text
speech_active = speech >= speech_threshold
singing_active = singing >= singing_threshold
music_active = music >= music_threshold
kind_mask = active class bitmask
primary_kind = Mixed if more than one class is active
```

The host may choose whether singing-only and mixed music regions are
transcribable.

### 3.5 Overlap Merge

Each AED probability frame is written into a global probability timeline.
Frames in the center of a chunk receive higher weight than frames near chunk
edges.

```text
merged_prob[t] = sum(prob_i[t] * weight_i[t]) / sum(weight_i[t])
```

Initial weighting:

```text
edge_weight = 0.35
center_weight = 1.0
```

For `hop_ms=2000` and `overlap_ms=250`, only boundary frames are covered by
two chunks. This keeps cost low, but it is only acceptable if the short-window
alignment gate shows that 250ms covers enough AED context. If it does not,
increase overlap before changing caller-level split settings.

### 3.6 Monotonic Commit

The segmenter must never rewrite output it has already emitted.

```text
consume_from = committed_until_ms
consume_to = min(available_prob_until_ms, latest_window_start_ms - edge_guard_ms)
```

Only frames in `[consume_from, consume_to]` may update the event state machine.
Frames before `committed_until_ms` are discarded. Overlap can only affect
uncommitted boundary frames.

Do not commit into the trailing overlap of the latest window. For the default
windows:

```text
[0ms, 2250ms]
[2000ms, 4250ms]
```

The first ingest may not commit `2000-2250ms`, because the second window still
has to contribute probabilities for that overlap. A valid implementation can
also express this invariant as:

```text
consume_to <= latest_window_end_ms - overlap_ms - edge_guard_ms
```

Tests must include a case where the second window disagrees with the first
window in the overlap region; the first window must not have committed that
region early.

### 3.7 User Configuration Semantics

The segmenter should expose post-processing settings, but the caller owns
user-facing values.

Required config:

```text
hard_split_pause_ms
max_chunk_ms
min_speech_ms
merge_gap_ms
music_gap_tolerance_ms
pad_start_ms
pad_end_ms
```

Rules:

- `hard_split_pause_ms` must be applied as-is.
- `max_chunk_ms` must be applied as-is.
- `overlap_ms` is a model-stability setting, not a hidden pause extension.
- `edge_guard_ms` must remain small and documented.
- `hop_ms`, `overlap_ms`, and `edge_guard_ms` must be non-negative 10ms-grid
  values, and `0 <= overlap_ms < hop_ms`.
- The expected first-segment confirmation latency is approximately
  `hop_ms + hard_split_pause_ms + edge_guard_ms`.

## 4. API Plan

### 4.1 Native C API

Native API is required for C/C++ consumers and language bindings.

Add opaque segmenter handle:

```c
typedef struct OmniAedOverlapSegmenterCtx* OmniAedOverlapSegmenterHandle;
```

Add config:

```c
typedef struct {
    int hop_ms;
    int overlap_ms;
    int edge_guard_ms;
    int hard_split_pause_ms;
    int max_chunk_ms;
    int min_speech_ms;
    int merge_gap_ms;
    int music_gap_tolerance_ms;
    int pad_start_ms;
    int pad_end_ms;
    float speech_threshold;
    float singing_threshold;
    float music_threshold;
} OmniAedOverlapConfig;
```

Add output:

```c
typedef enum {
    OMNI_AED_EVENT_SILENCE = 0,
    OMNI_AED_EVENT_SPEECH = 1,
    OMNI_AED_EVENT_SINGING = 2,
    OMNI_AED_EVENT_MUSIC = 3,
    OMNI_AED_EVENT_MIXED = 4,
} OmniAedEventKind;

typedef struct {
    float start;
    float end;
    OmniAedEventKind primary_kind;
    uint32_t kind_mask;
    float speech_confidence;
    float singing_confidence;
    float music_confidence;
    float confidence;
} OmniAedOnlineEvent;

typedef struct {
    float start;
    float end;
    int event_start_idx;
    int event_count;
} OmniAedOnlineSegment;
```

Output timestamps use seconds, matching existing `OmniSegment` and
`OmniAedSegment`. Internal state may use integer milliseconds or frame indices,
but public ABI must not mix units. `event_start_idx` and `event_count` refer
only to the `out_events` array returned by the same call.

Add functions:

```c
OMNIVAD_API OmniAedOverlapConfig omni_aed_overlap_config_default(void);

OMNIVAD_API OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_create(
    const char* bundle_path,
    const OmniAedOverlapConfig* config,
    int* out_error
);

OMNIVAD_API OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_create_from_buffer(
    const void* data,
    int size,
    const OmniAedOverlapConfig* config,
    int* out_error
);

OMNIVAD_API OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_clone(
    OmniAedOverlapSegmenterHandle handle,
    int* out_error
);

OMNIVAD_API int omni_aed_overlap_segmenter_ingest(
    OmniAedOverlapSegmenterHandle handle,
    const float* audio_data,
    int num_samples,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count
);

OMNIVAD_API int omni_aed_overlap_segmenter_ingest_int16(
    OmniAedOverlapSegmenterHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count
);

OMNIVAD_API int omni_aed_overlap_segmenter_flush(
    OmniAedOverlapSegmenterHandle handle,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count
);

OMNIVAD_API void omni_aed_overlap_segmenter_reset(
    OmniAedOverlapSegmenterHandle handle
);

OMNIVAD_API void omni_aed_overlap_segmenter_destroy(
    OmniAedOverlapSegmenterHandle handle
);
```

Notes:

- Use `omni_free` for returned arrays, matching existing APIs.
- Keep the ABI additive. Do not change existing structs or functions.
- The segmenter should accept arbitrary PCM chunk sizes.
- The segmenter should internally buffer until it has enough samples to run the
  next AED window.
- If ingest receives too few samples to run a new window, return `OMNI_OK` with
  zero output counts.
- When a returned count is zero, the corresponding output pointer must be NULL.
- `event_start_idx` and `event_count` are local to the `out_events` array
  returned by the same call. Hosts that need a global event timeline must append
  returned events themselves.
- `flush()` must define tail behavior explicitly: run the final partial window
  if it has enough frames for AED inference, clamp padding to real duration, and
  emit any pending transcribable segment that satisfies `min_speech_ms`.

### 4.2 Python API

Add:

```python
from omnivad import AedOverlapSegmenter

segmenter = AedOverlapSegmenter(
    hop_seconds=2.0,
    overlap_seconds=0.25,
    hard_split_pause_seconds=2.0,
    max_chunk_seconds=60.0,
)

for pcm in pcm_chunks:
    result = segmenter.ingest(pcm)
    # result.segments, result.events

final = segmenter.flush()
```

Python is the primary test harness because it can run synthetic probability
tests and real audio fixture tests quickly.

### 4.3 TypeScript API

Add after native/Python behavior is stable:

```ts
const segmenter = await AedOverlapSegmenter.create({
  hopMs: 2000,
  overlapMs: 250,
  hardSplitPauseMs: 2000,
  maxChunkMs: 60000,
});

const result = segmenter.ingest(pcm);
const final = segmenter.flush();
```

The TypeScript package should call the new WASM exports. Do not maintain a
separate TypeScript implementation of the event state machine.

## 5. Implementation Phases

### Phase 0: Pure Algorithm Prototype and Tests

Prototype and test the event pipeline on synthetic probability frames before
connecting to AED inference. The prototype is a reference harness, not the
shipping implementation.

Files:

- `omnivad/aed_overlap.py`
- `tests/test_aed_overlap_segmenter.py`

Minimum synthetic tests:

- Speech followed by silence emits after exactly `hard_split_pause_ms`.
- `max_chunk_ms` caps segment duration. When a long segment contains an
  internal non-transcribable gap before the cap, split at the longest such gap;
  fall back to a hard split only when no internal gap exists.
- Adjacent speech events with short gaps are merged.
- Pure music is skipped.
- Speech + music becomes `Mixed` and remains transcribable.
- Singing is transcribable and labeled.
- Overlap boundary predictions are merged deterministically.
- Emitted segments are strictly monotonic.
- Already committed frames cannot be changed by later overlap.
- The trailing overlap of a window is not committed until the next window has
  been merged.
- Arbitrary ingest chunk sizes produce identical output.
- Frame timestamps are aligned with the existing AED convention, including the
  25ms analysis window tail and clamp-to-duration behavior.

### Phase 1: Native Segmenter

Move the proven algorithm into native C++ and expose the C API. Native C++ is
the single authoritative implementation. Python and TypeScript must be bindings
or test harnesses, not independent implementations.

Files:

- `native/include/omnivad.h`
- `native/src/omnivad.cpp`
- `native/test/test_aed_overlap_segmenter.cpp`

Native tests:

- Config defaults and invalid config rejection.
- Empty/tiny input behavior.
- Ingest arbitrary chunk sizes.
- Flush pending tail.
- Returned arrays are allocated and freed via `omni_free`.
- Reset clears all internal state.

### Phase 2: Python Binding

Bind the native segmenter via ctypes.

Files:

- `omnivad/_binding.py`
- `omnivad/aed_overlap.py`
- `omnivad/__init__.py`

Python tests should mirror native scenarios and add fixture-level coverage.

### Phase 3: Real Audio Regression

Run with existing fixtures:

- `tests/data/DQacCB9tDaw_16K_2mins.mp3`
- `tests/data/hello_en.wav`
- `tests/data/hello_zh.wav`
- `tests/data/en_medium.wav`
- `tests/data/zh_medium.wav`
- `tests/data/event.wav`

Assertions:

- Outputs are deterministic across repeated runs.
- Outputs are identical across ingest chunk sizes, where practical.
- Segments are monotonic and non-overlapping.
- No segment exceeds `max_chunk_ms`; long segments prefer the longest internal
  non-transcribable gap before the cap over a hard boundary.
- Segment split gaps respect `hard_split_pause_ms`.
- AED event classes remain inspectable.
- Sliding-window probabilities are compared against full-audio AED
  probabilities to catch short-window context drift before downstream adoption.

The `DQacCB9tDaw_16K_2mins.mp3` fixture is the primary offline alignment gate.
Run full-audio AED once, then run the overlap segmenter on the same decoded
16kHz mono PCM using multiple ingest chunk sizes. The windowed result must stay
close to the full-audio result:

- Speech/singing/music frame probabilities should match within a small
  tolerance outside the intentionally uncommitted overlap tail.
- Event boundaries should differ only by a small frame-level tolerance.
- Transcribable segment coverage should have high IoU with the full-audio AED
  baseline.
- Any systematic start/end drift must fail the gate, because it would produce
  different downstream transcription cuts.
- Max-chunk split behavior should also use composites derived from this fixture:
  first run non-overlap AED, crop real speech/singing spans from its output,
  insert controlled pause durations, then verify overlap segmentation chooses
  the longest eligible internal pause before the cap and preserves
  transcribable coverage under hard-split fallback.
- Continuation tails created by a max-chunk split must keep their
  transcribable label even if the remaining tail is shorter than
  `min_speech_ms`; `min_speech_ms` should filter newly detected short events,
  not truncate already accepted long speech/singing.

Initial numeric thresholds:

```text
prob_mae <= 0.05
boundary_tolerance_ms <= 120
transcribable_iou >= 0.95
```

These thresholds are starting points. If the fixture shows that 250ms overlap
cannot meet them, increase overlap before accepting the segmenter default.

### Phase 4: Consumer Integration Guidance

After OmniVAD-Kit native tests pass, downstream applications can adopt the
segmenter through the C API or language bindings:

```text
PCM decode -> omni_aed_overlap_segmenter_ingest_int16 -> transcribable chunk queue
```

Consumer guidance:

- Adopt behind a feature flag until regression tests pass.
- Run A/B comparisons against the existing segmentation path before removing it.
- Keep true streaming microphone paths separate unless they have dedicated tests.
- Emit pending chunk ranges as soon as segments are confirmed.
- Preserve caller-configured pause and max chunk values.
- Use a rolling PCM buffer: retain only uncommitted tail plus downstream
  in-flight ranges.
- Define fallback behavior without keeping the full decoded file in memory. If
  fallback needs full audio, use a temporary file or a bounded re-decode path.
- Log segmenter RTF, ingest latency, emitted segment count, skipped music-only
  duration, and downstream queue latency.

### Phase 5: TypeScript Package

Expose the same behavior in `packages/omnivad` after native behavior is locked.
Use the Python/native test vectors as fixtures to prevent drift.

## 6. Testing Gates

Do not integrate into downstream applications until these pass:

```bash
pytest tests/test_aed_overlap_segmenter.py
pytest tests/test_determinism.py tests/test_edge_cases.py
pytest tests/test_aed_window_alignment.py
cmake --build native/build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
./native/build/test_all models/ tests/data/en_medium.wav
```

Manual consumer verification:

- Import a 2-minute MP3 and compare final transcript completeness with the
  existing path.
- Import a 10-minute WAV and verify memory does not grow with full duration.
- Confirm progress is monotonic and does not jump backward.
- Confirm music-only regions do not create downstream transcription jobs.

## 7. Risk List

- AED inference may be unstable on very short windows. If 2250ms windows are too
  noisy, increase overlap to 500ms before increasing the whole window. The
  default overlap must be chosen by the windowed-vs-full alignment gate.
- Long-audio RTF must be budgeted before downstream adoption. With
  `hop_ms=2000` and `overlap_ms=250`, AED inference does about 12.5% duplicate
  audio work plus per-window setup overhead.
- AED probability frame alignment must be verified against timestamps from
  full-audio AED output.
- Music-heavy speech may be mislabeled as music-only. `Mixed` policy must keep
  speech/singing when their probabilities are meaningful.
- Native ABI additions must be append-only to avoid breaking existing users.
- Downstream consumers must not change user split semantics while adopting the
  new segmenter.

## 8. Definition of Done

- `AedOverlapSegmenter` is implemented in OmniVAD-Kit native code.
- Python binding is available and covered by synthetic and fixture tests.
- The algorithm is deterministic across repeated runs.
- Ingest chunk size does not materially change emitted segment boundaries.
- Downstream consumers can use the kit API for AED overlap segmentation.
- Long-audio usage no longer requires a whole-audio rescan loop in the consumer.
