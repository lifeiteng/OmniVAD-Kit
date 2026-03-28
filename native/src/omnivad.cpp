/*
 * OmniVAD Unified C API - Implementation (ncnn backend)
 *
 * Contains:
 *   - Stream VAD with packed cache [1,1024,19]
 *   - Non-stream VAD (whole-audio inference + post-processing)
 *   - Non-stream AED (3-class: speech/singing/music + per-class post-processing)
 *
 * All models share the same 80-dim log-mel fbank frontend:
 *   sample_rate=16000, frame_length=400 (25ms), frame_shift=160 (10ms),
 *   Povey window, pre-emphasis=0.97, remove_dc_offset=true.
 */

#include "omnivad.h"
#include "frontend/fbank.h"
#include "datareader.h"
#include "net.h"

#include <algorithm>
#ifndef __EMSCRIPTEN__
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#endif
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

/* -------------------------------------------------------------------------- */
/*  Constants                                                                 */
/* -------------------------------------------------------------------------- */

static const int SAMPLE_RATE       = 16000;
static const int FEAT_DIM          = 80;
static const int FRAME_LENGTH      = 400;   /* 25ms @ 16kHz */
static const int FRAME_SHIFT       = 160;   /* 10ms @ 16kHz */
static const float FRAME_SHIFT_SEC = 0.01f; /* 10ms per frame */

/* Stream VAD packed cache dimensions */
static const int CACHE_SIZE = 1024;  /* 8 * 128 */
static const int CACHE_LEN  = 19;

/* -------------------------------------------------------------------------- */
/*  Helpers: CMVN loading                                                     */
/* -------------------------------------------------------------------------- */

static bool load_binary_vector(const char* path, std::vector<float>& vec) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int dim = (int)(size / (long)sizeof(float));
    vec.resize(dim);
    size_t read = fread(vec.data(), sizeof(float), dim, fp);
    fclose(fp);
    return (int)read == dim;
}

/* -------------------------------------------------------------------------- */
/*  Helpers: .omnivad bundle loading                                          */
/* -------------------------------------------------------------------------- */

/*
 * Bundle format: OVAD(4) + version(4) + param_size(4) + bin_size(4)
 *                + cmvn_means_size(4) + cmvn_istd_size(4)
 *                + param_data + bin_data + cmvn_means + cmvn_istd
 */
struct OmniBundle {
    std::vector<char> param_data;
    std::vector<char> bin_data;
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool has_cmvn;
};

static bool load_bundle(const char* path, OmniBundle& bundle) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[omnivad] failed to open bundle: %s\n", path);
        return false;
    }

    /* Read header (24 bytes) */
    char magic[4];
    uint32_t version, param_size, bin_size, means_size, istd_size;

    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "OVAD", 4) != 0) {
        fprintf(stderr, "[omnivad] invalid bundle magic in: %s\n", path);
        fclose(fp);
        return false;
    }
    if (fread(&version, 4, 1, fp) != 1 || version != 1) {
        fprintf(stderr, "[omnivad] unsupported bundle version %u in: %s\n", version, path);
        fclose(fp);
        return false;
    }
    if (fread(&param_size, 4, 1, fp) != 1 ||
        fread(&bin_size, 4, 1, fp) != 1 ||
        fread(&means_size, 4, 1, fp) != 1 ||
        fread(&istd_size, 4, 1, fp) != 1) {
        fprintf(stderr, "[omnivad] truncated bundle header in: %s\n", path);
        fclose(fp);
        return false;
    }

    /* Read data sections */
    bundle.param_data.resize(param_size);
    bundle.bin_data.resize(bin_size);

    if (fread(bundle.param_data.data(), 1, param_size, fp) != param_size ||
        fread(bundle.bin_data.data(), 1, bin_size, fp) != bin_size) {
        fprintf(stderr, "[omnivad] truncated bundle data in: %s\n", path);
        fclose(fp);
        return false;
    }

    /* CMVN (optional but expected) */
    bundle.has_cmvn = (means_size > 0 && istd_size > 0);
    if (bundle.has_cmvn) {
        int means_dim = means_size / sizeof(float);
        int istd_dim = istd_size / sizeof(float);
        bundle.cmvn_means.resize(means_dim);
        bundle.cmvn_istd.resize(istd_dim);
        if (fread(bundle.cmvn_means.data(), 1, means_size, fp) != means_size ||
            fread(bundle.cmvn_istd.data(), 1, istd_size, fp) != istd_size) {
            fprintf(stderr, "[omnivad] truncated CMVN data in: %s\n", path);
            fclose(fp);
            return false;
        }
    }

    fclose(fp);
    return true;
}

/* Load ncnn model from in-memory param/bin data */
static ncnn::Net* load_ncnn_from_memory(const std::vector<char>& param_data,
                                         const std::vector<char>& bin_data) {
    ncnn::Net* net = new (std::nothrow) ncnn::Net();
    if (!net) return NULL;

    if (net->load_param_mem(param_data.data()) != 0) {
        fprintf(stderr, "[omnivad] failed to load param from bundle\n");
        delete net;
        return NULL;
    }

    /* Load model binary directly from memory via DataReaderFromMemory.
     * No temp files needed — works everywhere including WASM. */
    const unsigned char* bin_ptr = (const unsigned char*)bin_data.data();
    ncnn::DataReaderFromMemory dr(bin_ptr);
    if (net->load_model(dr) != 0) {
        fprintf(stderr, "[omnivad] failed to load model bin from memory\n");
        delete net;
        return NULL;
    }
    return net;
}

static void apply_cmvn(const std::vector<float>& means,
                       const std::vector<float>& istd,
                       float* features,
                       int num_frames, int feat_dim) {
    if ((int)means.size() < feat_dim || (int)istd.size() < feat_dim) return;
    for (int t = 0; t < num_frames; ++t) {
        float* f = features + t * feat_dim;
        for (int d = 0; d < feat_dim; ++d) {
            f[d] = (f[d] - means[d]) * istd[d];
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Helpers: Post-processing                                                  */
/*  Exact port of Python VadPostprocessor from vad_postprocessor.py           */
/* -------------------------------------------------------------------------- */

struct FrameSegment {
    int start;
    int end;
};

/* VAD state machine states (matches Python VadState enum) */
enum VadState {
    VAD_SILENCE          = 0,
    VAD_POSSIBLE_SPEECH  = 1,
    VAD_SPEECH           = 2,
    VAD_POSSIBLE_SILENCE = 3,
};

/*
 * Step 1: Causal moving-average smoothing.
 * Matches Python: np.convolve(probs, kernel, mode='full')[:len(probs)]
 * with boundary fix: first window_size-1 frames use cumulative average.
 */
static void smooth_probs_causal(float* probs, int num_frames, int window_size) {
    if (window_size <= 1 || num_frames <= 0) return;

    std::vector<float> buf(num_frames);

    /* Boundary: first window_size-1 frames use cumulative average */
    int boundary = (window_size - 1 < num_frames) ? window_size - 1 : num_frames;
    for (int i = 0; i < boundary; ++i) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) sum += probs[j];
        buf[i] = sum / (float)(i + 1);
    }

    /* Main: causal moving average with window_size */
    float window_sum = 0.0f;
    for (int j = 0; j < window_size && j < num_frames; ++j) window_sum += probs[j];

    for (int i = boundary; i < num_frames; ++i) {
        /* np.convolve(probs, ones/N, 'full')[:len] is a causal filter:
         * smoothed[i] = mean(probs[i-window_size+1 .. i]) */
        if (i >= window_size) {
            window_sum = 0.0f;
            for (int j = i - window_size + 1; j <= i; ++j) window_sum += probs[j];
        }
        buf[i] = window_sum / (float)window_size;
        /* Slide window forward */
        if (i + 1 < num_frames) {
            window_sum += probs[i + 1];
            window_sum -= probs[i - window_size + 1];
        }
    }

    memcpy(probs, buf.data(), sizeof(float) * num_frames);
}

/*
 * Step 2: Apply threshold. Matches Python: (probs >= threshold).
 */
static void apply_threshold(const float* probs, int* binary, int num_frames, float threshold) {
    for (int i = 0; i < num_frames; ++i) {
        binary[i] = (probs[i] >= threshold) ? 1 : 0;
    }
}

/*
 * Step 3: 4-state machine smoothing.
 * Matches Python _smooth_preds_with_state_machine exactly.
 */
static void state_machine_smooth(const int* binary, int* decisions, int num_frames,
                                  int min_speech_frames, int min_silence_frames) {
    if (min_speech_frames <= 0 && min_silence_frames <= 0) {
        memcpy(decisions, binary, sizeof(int) * num_frames);
        return;
    }

    VadState state = VAD_SILENCE;
    int speech_start = -1;
    int silence_start = -1;

    for (int t = 0; t < num_frames; ++t) {
        int is_speech = binary[t];

        if (state == VAD_SILENCE) {
            if (is_speech) {
                state = VAD_POSSIBLE_SPEECH;
                speech_start = t;
            }
        } else if (state == VAD_POSSIBLE_SPEECH) {
            if (is_speech) {
                if (t - speech_start >= min_speech_frames) {
                    state = VAD_SPEECH;
                    /* Backfill: mark speech_start..t-1 as speech */
                    for (int j = speech_start; j < t; ++j) decisions[j] = 1;
                }
            } else {
                state = VAD_SILENCE;
                speech_start = -1;
            }
        } else if (state == VAD_SPEECH) {
            if (!is_speech) {
                state = VAD_POSSIBLE_SILENCE;
                silence_start = t;
            }
        } else if (state == VAD_POSSIBLE_SILENCE) {
            if (!is_speech) {
                if (t - silence_start >= min_silence_frames) {
                    state = VAD_SILENCE;
                    speech_start = -1;
                }
            } else {
                state = VAD_SPEECH;
                silence_start = -1;
            }
        }

        /* Current frame's decision */
        if (state == VAD_SPEECH || state == VAD_POSSIBLE_SILENCE) {
            decisions[t] = 1;
        } else {
            decisions[t] = 0;
        }
    }
}

/*
 * Step 4: Fix smooth window start.
 * Extend speech starts backward by smooth_window_size frames.
 * Matches Python _fix_smooth_window_start.
 */
static void fix_smooth_window_start(int* decisions, int num_frames, int smooth_window_size) {
    if (smooth_window_size <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    for (int t = 1; t < num_frames; ++t) {
        if (orig[t - 1] == 0 && orig[t] == 1) {
            int start = (t - smooth_window_size > 0) ? (t - smooth_window_size) : 0;
            for (int j = start; j < t; ++j) decisions[j] = 1;
        }
    }
}

/*
 * Step 5: Merge short silence segments.
 * Matches Python _merge_short_silence_segments.
 */
static void merge_short_silence(int* decisions, int num_frames, int merge_silence_frames) {
    if (merge_silence_frames <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    int silence_start = -1;
    for (int t = 0; t < num_frames; ++t) {
        if (t > 0 && orig[t - 1] == 1 && orig[t] == 0 && silence_start < 0) {
            silence_start = t;
        } else if (t > 0 && orig[t - 1] == 0 && orig[t] == 1 && silence_start >= 0) {
            int gap = t - silence_start;
            if (gap < merge_silence_frames) {
                for (int j = silence_start; j < t; ++j) decisions[j] = 1;
            }
            silence_start = -1;
        }
    }
}

/*
 * Step 6: Extend speech segments N frames before and after.
 * Matches Python _extend_speech_segments (convolution-based).
 */
static void extend_speech(int* decisions, int num_frames, int extend_frames) {
    if (extend_frames <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    for (int t = 0; t < num_frames; ++t) {
        if (orig[t] == 1) {
            int lo = (t - extend_frames > 0) ? (t - extend_frames) : 0;
            int hi = (t + extend_frames < num_frames) ? (t + extend_frames + 1) : num_frames;
            for (int j = lo; j < hi; ++j) decisions[j] = 1;
        }
    }
}

/*
 * Step 7: Split long speech segments at lowest-probability point.
 * Matches Python _split_long_speech_segments + _find_split_points.
 */
static void split_long_segments(int* decisions, int num_frames,
                                 int max_speech_frames, const float* raw_probs) {
    if (max_speech_frames <= 0) return;

    /* Find speech segments from decisions */
    int seg_start = -1;
    std::vector<std::pair<int,int>> segments;
    for (int t = 0; t < num_frames; ++t) {
        if (decisions[t] == 1 && seg_start < 0) {
            seg_start = t;
        } else if (decisions[t] == 0 && seg_start >= 0) {
            segments.push_back({seg_start, t});
            seg_start = -1;
        }
    }
    if (seg_start >= 0) segments.push_back({seg_start, num_frames});

    /* For each segment, find split points */
    for (auto& seg : segments) {
        int s = seg.first;
        int e = seg.second;
        int length = e - s;
        int local_start = 0;

        while (local_start < length) {
            if ((length - local_start) <= max_speech_frames) break;

            int window_start = local_start + max_speech_frames / 2;
            int window_end = local_start + max_speech_frames;
            if (window_end > length) window_end = length;

            /* Find min prob in window */
            int min_idx = window_start;
            float min_val = raw_probs[s + window_start];
            for (int j = window_start + 1; j < window_end; ++j) {
                if (raw_probs[s + j] < min_val) {
                    min_val = raw_probs[s + j];
                    min_idx = j;
                }
            }

            /* Insert silence at split point */
            decisions[s + min_idx] = 0;
            local_start = min_idx + 1;
        }
    }
}

/*
 * Convert frame-level decisions to time segments.
 * Matches Python decision_to_segment: uses FRAME_SHIFT_S for start,
 * and for trailing segments adds FRAME_LENGTH_S to end time.
 */
static const float FRAME_LENGTH_SEC = 0.025f;

static std::vector<FrameSegment> decisions_to_segments(
    const int* decisions, int num_frames, int min_speech_frames)
{
    std::vector<FrameSegment> segments;
    int seg_start = -1;
    for (int t = 0; t < num_frames; ++t) {
        if (decisions[t] == 1 && seg_start < 0) {
            seg_start = t;
        } else if (decisions[t] == 0 && seg_start >= 0) {
            segments.push_back({seg_start, t});
            seg_start = -1;
        }
    }
    if (seg_start >= 0) {
        segments.push_back({seg_start, num_frames});
    }
    return segments;
}

/*
 * Full post-processing pipeline matching Python VadPostprocessor.process().
 * raw_probs is the unmodified model output (needed for split_long_segments).
 */
static std::vector<FrameSegment> postprocess_pipeline(
    const float* raw_probs, int num_frames,
    float threshold, int smooth_window_size,
    int min_speech_frames, int max_speech_frames,
    int min_silence_frames, int merge_silence_frames,
    int extend_speech_frames)
{
    /* Work on a copy of probs for smoothing */
    std::vector<float> probs(raw_probs, raw_probs + num_frames);

    /* Step 1: Causal smoothing */
    smooth_probs_causal(probs.data(), num_frames, smooth_window_size);

    /* Step 2: Binary threshold */
    std::vector<int> binary(num_frames);
    apply_threshold(probs.data(), binary.data(), num_frames, threshold);

    /* Step 3: State machine smoothing */
    std::vector<int> decisions(num_frames, 0);
    state_machine_smooth(binary.data(), decisions.data(), num_frames,
                          min_speech_frames, min_silence_frames);

    /* Step 4: Fix smooth window start */
    fix_smooth_window_start(decisions.data(), num_frames, smooth_window_size);

    /* Step 5: Merge short silence */
    merge_short_silence(decisions.data(), num_frames, merge_silence_frames);

    /* Step 6: Extend speech */
    extend_speech(decisions.data(), num_frames, extend_speech_frames);

    /* Step 7: Split long segments (uses raw_probs, not smoothed) */
    split_long_segments(decisions.data(), num_frames, max_speech_frames, raw_probs);

    /* Convert to segments */
    return decisions_to_segments(decisions.data(), num_frames, min_speech_frames);
}

/* -------------------------------------------------------------------------- */
/*  Default configs                                                           */
/* -------------------------------------------------------------------------- */

extern "C" {

OmniPostConfig omni_post_config_default(void) {
    OmniPostConfig cfg;
    cfg.threshold            = 0.4f;
    cfg.smooth_window_size   = 5;
    cfg.min_speech_frames    = 20;    /* 200ms */
    cfg.min_silence_frames   = 20;    /* 200ms */
    cfg.max_speech_frames    = 2000;  /* 20s */
    cfg.merge_silence_frames = 0;     /* disabled */
    cfg.extend_speech_frames = 0;     /* disabled */
    return cfg;
}

OmniAedPostConfig omni_aed_post_config_default(void) {
    OmniAedPostConfig cfg;
    cfg.speech  = omni_post_config_default();
    cfg.singing = omni_post_config_default();
    cfg.music   = omni_post_config_default();
    /* AED classes may benefit from different defaults */
    cfg.speech.threshold  = 0.5f;
    cfg.singing.threshold = 0.5f;
    cfg.music.threshold   = 0.5f;
    return cfg;
}

const char* omni_error_string(int error_code) {
    switch (error_code) {
        case OMNI_OK:                return "success";
        case OMNI_ERR_NULL_HANDLE:   return "null handle";
        case OMNI_ERR_NULL_INPUT:    return "null input pointer";
        case OMNI_ERR_LOAD_PARAM:    return "failed to load ncnn param file";
        case OMNI_ERR_LOAD_MODEL:    return "failed to load ncnn model file";
        case OMNI_ERR_LOAD_CMVN:     return "failed to load CMVN file";
        case OMNI_ERR_NO_FRAMES:     return "no frames could be extracted from audio";
        case OMNI_ERR_INFERENCE:      return "ncnn inference failed";
        case OMNI_ERR_OUT_OF_MEMORY: return "out of memory";
        case OMNI_ERR_INVALID_ARG:   return "invalid argument";
        default:                         return "unknown error";
    }
}

void omni_free(void* ptr) {
    free(ptr);
}

/* ========================================================================== */
/*  1. Stream VAD Implementation                                              */
/* ========================================================================== */

struct OmniVadStreamCtx {
    ncnn::Net* net;

    /* Audio buffer (sliding window, keeps last 25ms) */
    std::deque<float> audio_buffer;

    /* Packed cache [w=CACHE_LEN, h=CACHE_SIZE] -> [1, 1024, 19] */
    ncnn::Mat cache_packed;

    /* Frame counter */
    int frame_offset;

    /* Speech threshold */
    float threshold;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;

    /* Fbank computer */
    vad::Fbank* fbank;
};

OmniVadHandle omni_vad_stream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd,
    float threshold)
{
    OmniVadStreamCtx* ctx = new (std::nothrow) OmniVadStreamCtx();
    if (!ctx) return NULL;

    ctx->net = NULL;
    ctx->fbank = NULL;
    ctx->frame_offset = 0;
    ctx->threshold = threshold;
    ctx->use_cmvn = false;

    /* Initialize packed cache [w=CACHE_LEN, h=CACHE_SIZE] */
    ctx->cache_packed = ncnn::Mat(CACHE_LEN, CACHE_SIZE);
    ctx->cache_packed.fill(0.0f);

    /* Load ncnn model */
    ctx->net = new (std::nothrow) ncnn::Net();
    if (!ctx->net) {
        delete ctx;
        return NULL;
    }

    if (ctx->net->load_param(model_param) != 0) {
        fprintf(stderr, "[omnivad] stream: failed to load param: %s\n", model_param);
        delete ctx->net;
        delete ctx;
        return NULL;
    }
    if (ctx->net->load_model(model_bin) != 0) {
        fprintf(stderr, "[omnivad] stream: failed to load model: %s\n", model_bin);
        delete ctx->net;
        delete ctx;
        return NULL;
    }

    /* Create fbank computer */
    ctx->fbank = new (std::nothrow) vad::Fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    if (!ctx->fbank) {
        delete ctx->net;
        delete ctx;
        return NULL;
    }

    /* Load CMVN (optional) */
    if (cmvn_means && cmvn_istd) {
        bool ok = load_binary_vector(cmvn_means, ctx->cmvn_means) &&
                  load_binary_vector(cmvn_istd,  ctx->cmvn_istd);
        if (ok && (int)ctx->cmvn_means.size() >= FEAT_DIM &&
                  (int)ctx->cmvn_istd.size()  >= FEAT_DIM) {
            ctx->use_cmvn = true;
        } else {
            fprintf(stderr, "[omnivad] stream: CMVN load failed or dimension mismatch, skipping CMVN\n");
            ctx->cmvn_means.clear();
            ctx->cmvn_istd.clear();
        }
    }

    return ctx;
}

int omni_vad_stream_process(
    OmniVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniVadStreamResult* result)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !result) return OMNI_ERR_NULL_INPUT;

    OmniVadStreamCtx* ctx = handle;

    /* Convert 16-bit PCM to float and push into buffer */
    for (int i = 0; i < num_samples; ++i) {
        ctx->audio_buffer.push_back((float)audio_data[i]);
    }

    /* Keep buffer size at most frame_length (400 samples = 25ms) */
    while ((int)ctx->audio_buffer.size() > FRAME_LENGTH) {
        ctx->audio_buffer.pop_front();
    }

    /* Not enough data for a full frame yet */
    if ((int)ctx->audio_buffer.size() < FRAME_LENGTH) {
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_OK;
    }

    /* Extract fbank features for current window */
    std::vector<float> frame_audio(ctx->audio_buffer.begin(), ctx->audio_buffer.end());
    std::vector<float> features;
    int num_frames = ctx->fbank->Compute(frame_audio, &features);

    if (num_frames == 0 || features.empty()) {
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_OK;
    }

    /* Apply CMVN to the last frame */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Take the last frame's features */
    std::vector<float> current_feat(FEAT_DIM);
    memcpy(current_feat.data(),
           features.data() + (num_frames - 1) * FEAT_DIM,
           FEAT_DIM * sizeof(float));

    ctx->frame_offset++;

    /* ncnn inference: single frame with packed cache
     *   in0 = feat [w=FEAT_DIM, h=1]
     *   in1 = cache_packed [w=CACHE_LEN, h=CACHE_SIZE]
     *   out0 = probs [1]
     *   out1 = new_cache_packed [w=CACHE_LEN, h=CACHE_SIZE]
     */
    ncnn::Mat in_feat(FEAT_DIM, 1);
    memcpy(in_feat.data, current_feat.data(), FEAT_DIM * sizeof(float));

    ncnn::Mat in_feat_clone = in_feat.clone();
    ncnn::Mat cache_clone = ctx->cache_packed.clone();

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);
    ex.input("in1", cache_clone);

    ncnn::Mat out_probs;
    int ret = ex.extract("out0", out_probs);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out0 failed: %d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_INFERENCE;
    }

    ncnn::Mat new_cache;
    ret = ex.extract("out1", new_cache);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out1 failed: %d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_INFERENCE;
    }

    /* Update cache */
    ctx->cache_packed = new_cache;

    float confidence = ((float*)out_probs.data)[0];
    result->confidence = confidence;
    result->is_speech = confidence > ctx->threshold;
    result->frame_offset = ctx->frame_offset;
    return OMNI_OK;
}

int omni_vad_stream_detect_full(
    OmniVadHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_INPUT;
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    OmniVadStreamCtx* ctx = handle;

    /* Step 1: Whole-file fbank (matches Python AudioFeat.extract) */
    std::vector<float> wave(audio_data, audio_data + num_samples);
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Step 2: Reset cache, then run model frame-by-frame with cache */
    ncnn::Mat cache_packed(CACHE_LEN, CACHE_SIZE);
    cache_packed.fill(0.0f);

    float* probs = (float*)malloc(sizeof(float) * num_frames);
    if (!probs) return OMNI_ERR_OUT_OF_MEMORY;

    for (int t = 0; t < num_frames; ++t) {
        /* Prepare single-frame input [w=FEAT_DIM, h=1] */
        ncnn::Mat in_feat(FEAT_DIM, 1);
        memcpy(in_feat.data, features.data() + t * FEAT_DIM, FEAT_DIM * sizeof(float));

        ncnn::Mat in_feat_clone = in_feat.clone();
        ncnn::Mat cache_clone = cache_packed.clone();

        ncnn::Extractor ex = ctx->net->create_extractor();
        ex.input("in0", in_feat_clone);
        ex.input("in1", cache_clone);

        ncnn::Mat out_prob_mat;
        int ret = ex.extract("out0", out_prob_mat);
        if (ret != 0) {
            free(probs);
            return OMNI_ERR_INFERENCE;
        }

        ncnn::Mat new_cache;
        ret = ex.extract("out1", new_cache);
        if (ret != 0) {
            free(probs);
            return OMNI_ERR_INFERENCE;
        }

        cache_packed = new_cache;
        probs[t] = ((float*)out_prob_mat.data)[0];
    }

    *out_probs = probs;
    *out_frames = num_frames;
    return OMNI_OK;
}

OmniVadHandle omni_vad_stream_create_from_bundle(
    const char* bundle_path,
    float threshold)
{
    OmniBundle bundle;
    if (!load_bundle(bundle_path, bundle)) return NULL;

    OmniVadStreamCtx* ctx = new (std::nothrow) OmniVadStreamCtx();
    if (!ctx) return NULL;

    ctx->fbank = NULL;
    ctx->frame_offset = 0;
    ctx->threshold = threshold;
    ctx->use_cmvn = false;
    ctx->cache_packed = ncnn::Mat(CACHE_LEN, CACHE_SIZE);
    ctx->cache_packed.fill(0.0f);

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data);
    if (!ctx->net) { delete ctx; return NULL; }

    ctx->fbank = new (std::nothrow) vad::Fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    if (!ctx->fbank) { delete ctx->net; delete ctx; return NULL; }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    }
    return ctx;
}

void omni_vad_stream_reset(OmniVadHandle handle) {
    if (!handle) return;
    handle->audio_buffer.clear();
    handle->frame_offset = 0;
    handle->cache_packed.fill(0.0f);
    if (handle->fbank) {
        handle->fbank->reset();
    }
}

int omni_vad_stream_get_frame_offset(OmniVadHandle handle) {
    if (!handle) return 0;
    return handle->frame_offset;
}

void omni_vad_stream_destroy(OmniVadHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle->fbank;
    delete handle;
}

/* ========================================================================== */
/*  2. Non-stream VAD Implementation                                          */
/* ========================================================================== */

struct OmniVadNonStreamCtx {
    ncnn::Net* net;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;
};

OmniVadNonStreamHandle omni_vad_nonstream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd)
{
    OmniVadNonStreamCtx* ctx = new (std::nothrow) OmniVadNonStreamCtx();
    if (!ctx) return NULL;

    ctx->net = NULL;
    ctx->use_cmvn = false;

    /* Load ncnn model */
    ctx->net = new (std::nothrow) ncnn::Net();
    if (!ctx->net) {
        delete ctx;
        return NULL;
    }

    if (ctx->net->load_param(model_param) != 0) {
        fprintf(stderr, "[omnivad] nonstream-vad: failed to load param: %s\n", model_param);
        delete ctx->net;
        delete ctx;
        return NULL;
    }
    if (ctx->net->load_model(model_bin) != 0) {
        fprintf(stderr, "[omnivad] nonstream-vad: failed to load model: %s\n", model_bin);
        delete ctx->net;
        delete ctx;
        return NULL;
    }

    /* Load CMVN */
    if (cmvn_means && cmvn_istd) {
        bool ok = load_binary_vector(cmvn_means, ctx->cmvn_means) &&
                  load_binary_vector(cmvn_istd,  ctx->cmvn_istd);
        if (ok && (int)ctx->cmvn_means.size() >= FEAT_DIM &&
                  (int)ctx->cmvn_istd.size()  >= FEAT_DIM) {
            ctx->use_cmvn = true;
        } else {
            fprintf(stderr, "[omnivad] nonstream-vad: CMVN load failed, skipping\n");
            ctx->cmvn_means.clear();
            ctx->cmvn_istd.clear();
        }
    }

    return ctx;
}

/*
 * Internal: compute fbank features for the whole audio, apply CMVN,
 * run ncnn inference, return raw per-frame probabilities.
 *
 * For non-stream VAD, the model takes input in0 [w=feat_dim, h=num_frames]
 * and outputs out0 with one probability per frame.
 */
static int nonstream_vad_infer(
    OmniVadNonStreamCtx* ctx,
    const float* audio_data,
    int num_samples,
    std::vector<float>& out_probs,
    int& out_num_frames)
{
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    /* Wrap audio data into a vector for the fbank API */
    std::vector<float> wave(audio_data, audio_data + num_samples);

    /* Compute fbank features */
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Prepare ncnn input: [w=FEAT_DIM, h=num_frames] */
    ncnn::Mat in_feat(FEAT_DIM, num_frames);
    memcpy(in_feat.data, features.data(), sizeof(float) * features.size());
    ncnn::Mat in_feat_clone = in_feat.clone();

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);

    ncnn::Mat ncnn_out;
    int ret = ex.extract("out0", ncnn_out);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] nonstream-vad: ncnn extract failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    /* Extract per-frame probabilities.
     * Expected output: 1D array of length num_frames. */
    int T = num_frames;
    out_probs.resize(T);

    /* ncnn may output as dims=1 w=T, or dims=2 w=1 h=T, etc. Handle both. */
    if (ncnn_out.dims == 1) {
        T = ncnn_out.w;
        out_probs.resize(T);
        const float* p = (const float*)ncnn_out.data;
        for (int i = 0; i < T; ++i) {
            out_probs[i] = p[i];
        }
    } else if (ncnn_out.dims == 2) {
        /* Could be [w=1, h=T] or [w=T, h=1] */
        if (ncnn_out.w == 1) {
            T = ncnn_out.h;
            out_probs.resize(T);
            for (int i = 0; i < T; ++i) {
                out_probs[i] = ncnn_out.row(i)[0];
            }
        } else {
            T = ncnn_out.w;
            out_probs.resize(T);
            const float* p = ncnn_out.row(0);
            for (int i = 0; i < T; ++i) {
                out_probs[i] = p[i];
            }
        }
    } else {
        /* Fallback: treat entire data as flat array */
        T = ncnn_out.total();
        out_probs.resize(T);
        const float* p = (const float*)ncnn_out.data;
        for (int i = 0; i < T; ++i) {
            out_probs[i] = p[i];
        }
    }

    out_num_frames = T;
    return OMNI_OK;
}

int omni_vad_nonstream_process(
    OmniVadNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_segments || !out_count) return OMNI_ERR_NULL_INPUT;

    OmniPostConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = omni_post_config_default();
    }

    /* Run inference */
    std::vector<float> probs;
    int num_frames = 0;
    int ret = nonstream_vad_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_segments = NULL;
        *out_count = 0;
        return ret;
    }

    /* Run full post-processing pipeline (matches Python exactly) */
    std::vector<FrameSegment> segs = postprocess_pipeline(
        probs.data(), num_frames,
        cfg.threshold, cfg.smooth_window_size,
        cfg.min_speech_frames, cfg.max_speech_frames,
        cfg.min_silence_frames, cfg.merge_silence_frames,
        cfg.extend_speech_frames);

    /* Convert to output format with proper timestamp calculation */
    int count = (int)segs.size();
    if (count == 0) {
        *out_segments = NULL;
        *out_count = 0;
        return OMNI_OK;
    }

    float wav_dur = (float)num_samples / (float)SAMPLE_RATE;

    OmniSegment* result = (OmniSegment*)malloc(sizeof(OmniSegment) * count);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < count; ++i) {
        result[i].start = segs[i].start * FRAME_SHIFT_SEC;
        /* For the last segment, match Python: end = num_frames * shift + frame_length */
        if (i == count - 1 && segs[i].end == num_frames) {
            float end_time = (float)num_frames * FRAME_SHIFT_SEC + FRAME_LENGTH_SEC;
            if (end_time > wav_dur) end_time = wav_dur;
            result[i].end = end_time;
        } else {
            result[i].end = segs[i].end * FRAME_SHIFT_SEC;
        }
    }

    *out_segments = result;
    *out_count = count;
    return OMNI_OK;
}

int omni_vad_nonstream_process_raw(
    OmniVadNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_INPUT;

    std::vector<float> probs;
    int num_frames = 0;
    int ret = nonstream_vad_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_probs = NULL;
        *out_frames = 0;
        return ret;
    }

    float* result = (float*)malloc(sizeof(float) * num_frames);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;
    memcpy(result, probs.data(), sizeof(float) * num_frames);

    *out_probs = result;
    *out_frames = num_frames;
    return OMNI_OK;
}

OmniVadNonStreamHandle omni_vad_nonstream_create_from_bundle(const char* bundle_path) {
    OmniBundle bundle;
    if (!load_bundle(bundle_path, bundle)) return NULL;

    OmniVadNonStreamCtx* ctx = new (std::nothrow) OmniVadNonStreamCtx();
    if (!ctx) return NULL;
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data);
    if (!ctx->net) { delete ctx; return NULL; }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    }
    return ctx;
}

void omni_vad_nonstream_destroy(OmniVadNonStreamHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle;
}

/* ========================================================================== */
/*  3. Non-stream AED Implementation                                          */
/* ========================================================================== */

struct OmniAedNonStreamCtx {
    ncnn::Net* net;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;
};

OmniAedNonStreamHandle omni_aed_nonstream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd)
{
    OmniAedNonStreamCtx* ctx = new (std::nothrow) OmniAedNonStreamCtx();
    if (!ctx) return NULL;

    ctx->net = NULL;
    ctx->use_cmvn = false;

    /* Load ncnn model */
    ctx->net = new (std::nothrow) ncnn::Net();
    if (!ctx->net) {
        delete ctx;
        return NULL;
    }

    if (ctx->net->load_param(model_param) != 0) {
        fprintf(stderr, "[omnivad] nonstream-aed: failed to load param: %s\n", model_param);
        delete ctx->net;
        delete ctx;
        return NULL;
    }
    if (ctx->net->load_model(model_bin) != 0) {
        fprintf(stderr, "[omnivad] nonstream-aed: failed to load model: %s\n", model_bin);
        delete ctx->net;
        delete ctx;
        return NULL;
    }

    /* Load CMVN */
    if (cmvn_means && cmvn_istd) {
        bool ok = load_binary_vector(cmvn_means, ctx->cmvn_means) &&
                  load_binary_vector(cmvn_istd,  ctx->cmvn_istd);
        if (ok && (int)ctx->cmvn_means.size() >= FEAT_DIM &&
                  (int)ctx->cmvn_istd.size()  >= FEAT_DIM) {
            ctx->use_cmvn = true;
        } else {
            fprintf(stderr, "[omnivad] nonstream-aed: CMVN load failed, skipping\n");
            ctx->cmvn_means.clear();
            ctx->cmvn_istd.clear();
        }
    }

    return ctx;
}

/*
 * Internal: run AED inference, return per-frame 3-class probabilities.
 * Output layout: probs[frame * 3 + class], where class 0=speech, 1=singing, 2=music.
 */
static int nonstream_aed_infer(
    OmniAedNonStreamCtx* ctx,
    const float* audio_data,
    int num_samples,
    std::vector<float>& out_probs,
    int& out_num_frames)
{
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    std::vector<float> wave(audio_data, audio_data + num_samples);

    /* Compute fbank features */
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Prepare ncnn input */
    ncnn::Mat in_feat(FEAT_DIM, num_frames);
    memcpy(in_feat.data, features.data(), sizeof(float) * features.size());
    ncnn::Mat in_feat_clone = in_feat.clone();

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);

    ncnn::Mat ncnn_out;
    int ret = ex.extract("out0", ncnn_out);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] nonstream-aed: ncnn extract failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    /*
     * AED model outputs 3-class probabilities per frame.
     * Possible ncnn layouts:
     *   a) dims=2, w=3, h=T  -> row(t) gives [speech, singing, music]
     *   b) dims=2, h=3, w=T  -> transposed: row(cls)[t]
     *   c) dims=1, w=T*3     -> flat array
     */
    int T = 0;
    static const int NUM_CLASSES = 3;

    if (ncnn_out.dims == 2 && ncnn_out.w == NUM_CLASSES) {
        /* Layout (a): [h=T, w=3] - most common */
        T = ncnn_out.h;
        out_probs.resize(T * NUM_CLASSES);
        for (int t = 0; t < T; ++t) {
            const float* row = ncnn_out.row(t);
            out_probs[t * NUM_CLASSES + 0] = row[0];  /* speech */
            out_probs[t * NUM_CLASSES + 1] = row[1];  /* singing */
            out_probs[t * NUM_CLASSES + 2] = row[2];  /* music */
        }
    } else if (ncnn_out.dims == 2 && ncnn_out.h == NUM_CLASSES) {
        /* Layout (b): transposed [h=3, w=T] */
        T = ncnn_out.w;
        out_probs.resize(T * NUM_CLASSES);
        const float* row0 = ncnn_out.row(0);
        const float* row1 = ncnn_out.row(1);
        const float* row2 = ncnn_out.row(2);
        for (int t = 0; t < T; ++t) {
            out_probs[t * NUM_CLASSES + 0] = row0[t];
            out_probs[t * NUM_CLASSES + 1] = row1[t];
            out_probs[t * NUM_CLASSES + 2] = row2[t];
        }
    } else if (ncnn_out.dims == 1 && ncnn_out.w % NUM_CLASSES == 0) {
        /* Layout (c): flat [T*3] */
        T = ncnn_out.w / NUM_CLASSES;
        out_probs.resize(T * NUM_CLASSES);
        const float* p = (const float*)ncnn_out.data;
        memcpy(out_probs.data(), p, sizeof(float) * T * NUM_CLASSES);
    } else {
        fprintf(stderr, "[omnivad] nonstream-aed: unexpected output dims=%d w=%d h=%d c=%d\n",
                ncnn_out.dims, ncnn_out.w, ncnn_out.h, ncnn_out.c);
        return OMNI_ERR_INFERENCE;
    }

    out_num_frames = T;
    return OMNI_OK;
}

int omni_aed_nonstream_process(
    OmniAedNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_segments || !out_count) return OMNI_ERR_NULL_INPUT;

    OmniAedPostConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = omni_aed_post_config_default();
    }

    /* Run inference */
    std::vector<float> probs;
    int num_frames = 0;
    int ret = nonstream_aed_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_segments = NULL;
        *out_count = 0;
        return ret;
    }

    static const int NUM_CLASSES = 3;
    const OmniPostConfig* class_configs[NUM_CLASSES] = {
        &cfg.speech, &cfg.singing, &cfg.music
    };
    const OmniAedClass class_ids[NUM_CLASSES] = {
        OMNI_AED_SPEECH, OMNI_AED_SINGING, OMNI_AED_MUSIC
    };

    float wav_dur = (float)num_samples / (float)SAMPLE_RATE;

    /* Process each class independently */
    std::vector<OmniAedSegment> all_segments;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        /* Extract per-frame probabilities for this class */
        std::vector<float> class_probs(num_frames);
        for (int t = 0; t < num_frames; ++t) {
            class_probs[t] = probs[t * NUM_CLASSES + c];
        }

        /* Run full post-processing pipeline */
        const OmniPostConfig* cc = class_configs[c];
        std::vector<FrameSegment> segs = postprocess_pipeline(
            class_probs.data(), num_frames,
            cc->threshold, cc->smooth_window_size,
            cc->min_speech_frames, cc->max_speech_frames,
            cc->min_silence_frames, cc->merge_silence_frames,
            cc->extend_speech_frames);

        /* Convert to output format with class label and average confidence */
        for (size_t i = 0; i < segs.size(); ++i) {
            OmniAedSegment seg;
            seg.start = segs[i].start * FRAME_SHIFT_SEC;
            /* Match Python: trailing segment end = num_frames * shift + frame_length */
            if (segs[i].end == num_frames) {
                float end_time = (float)num_frames * FRAME_SHIFT_SEC + FRAME_LENGTH_SEC;
                if (end_time > wav_dur) end_time = wav_dur;
                seg.end = end_time;
            } else {
                seg.end = segs[i].end * FRAME_SHIFT_SEC;
            }
            seg.cls   = class_ids[c];

            /* Compute average confidence over the segment */
            float sum = 0.0f;
            int count = 0;
            for (int t = segs[i].start; t < segs[i].end && t < num_frames; ++t) {
                sum += class_probs[t];
                count++;
            }
            seg.confidence = (count > 0) ? (sum / count) : 0.0f;

            all_segments.push_back(seg);
        }
    }

    /* Sort all segments by start time */
    std::sort(all_segments.begin(), all_segments.end(),
              [](const OmniAedSegment& a, const OmniAedSegment& b) {
                  if (a.start != b.start) return a.start < b.start;
                  return a.cls < b.cls;
              });

    /* Allocate and return */
    int count = (int)all_segments.size();
    if (count == 0) {
        *out_segments = NULL;
        *out_count = 0;
        return OMNI_OK;
    }

    OmniAedSegment* result = (OmniAedSegment*)malloc(sizeof(OmniAedSegment) * count);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;

    memcpy(result, all_segments.data(), sizeof(OmniAedSegment) * count);

    *out_segments = result;
    *out_count = count;
    return OMNI_OK;
}

int omni_aed_nonstream_process_raw(
    OmniAedNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_INPUT;

    std::vector<float> probs;
    int num_frames = 0;
    int ret = nonstream_aed_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_probs = NULL;
        *out_frames = 0;
        return ret;
    }

    static const int NUM_CLASSES = 3;
    int total = num_frames * NUM_CLASSES;
    float* result = (float*)malloc(sizeof(float) * total);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;
    memcpy(result, probs.data(), sizeof(float) * total);

    *out_probs = result;
    *out_frames = num_frames;
    return OMNI_OK;
}

OmniAedNonStreamHandle omni_aed_nonstream_create_from_bundle(const char* bundle_path) {
    OmniBundle bundle;
    if (!load_bundle(bundle_path, bundle)) return NULL;

    OmniAedNonStreamCtx* ctx = new (std::nothrow) OmniAedNonStreamCtx();
    if (!ctx) return NULL;
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data);
    if (!ctx->net) { delete ctx; return NULL; }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    }
    return ctx;
}

void omni_aed_nonstream_destroy(OmniAedNonStreamHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle;
}

} /* extern "C" */
