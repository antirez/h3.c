#include "h3_dit.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef H3_BENCH_LATENT_H
#define H3_BENCH_LATENT_H 32
#endif
#ifndef H3_BENCH_LATENT_W
#define H3_BENCH_LATENT_W 32
#endif
#ifndef H3_BENCH_LATENT_T
#define H3_BENCH_LATENT_T 7
#endif
#ifndef H3_BENCH_AUDIO_T
#define H3_BENCH_AUDIO_T 37
#endif

enum {
    TEXT_ROWS = 6,
    TEXT_WIDTH = 5120,
    LATENT_T = H3_BENCH_LATENT_T,
    LATENT_H = H3_BENCH_LATENT_H,
    LATENT_W = H3_BENCH_LATENT_W,
    CANVAS_H = LATENT_H * 16,
    CANVAS_W = LATENT_W * 16,
    AUDIO_T = H3_BENCH_AUDIO_T,
    VIDEO_ELEMENTS = 24 * LATENT_T * LATENT_H * LATENT_W,
    AUDIO_ELEMENTS = 32 * 2 * AUDIO_T
};

static void die(const char *message) {
    fprintf(stderr, "h3_dit_bench: %s\n", message);
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const uint8_t *octets = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= octets[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint16_t *load_text(const char *path) {
    char error[512];
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, sizeof(error))) die(error);
    const h3_st_tensor *tensor = h3_st_find(&header, "x.output");
    size_t elements = TEXT_ROWS * TEXT_WIDTH;
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != elements)
        die("prompt fixture has the wrong schema");
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values || !h3_st_read_data(&header, tensor, values,
                                     elements * sizeof(*values),
                                     error, sizeof(error))) die(error);
    h3_st_free_header(&header);
    return values;
}

static void run_command_ab(h3_dit *dit, int interval, float *video,
                           float *audio, float *video_velocity,
                           float *audio_velocity) {
    char error[512];
    setenv("H3_DIT_COMMAND_BLOCKS", "0", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating AB reference outputs");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));

    static const int split_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double split_seconds = 0.0;
    int baseline_count = 0;
    int split_count = 0;
    char interval_text[16];
    snprintf(interval_text, sizeof(interval_text), "%d", interval);
    for (size_t index = 0;
         index < sizeof(split_pattern) / sizeof(*split_pattern); index++) {
        if (split_pattern[index])
            setenv("H3_DIT_COMMAND_BLOCKS", interval_text, 1);
        else
            setenv("H3_DIT_COMMAND_BLOCKS", "0", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("command split changed DiT output bytes");
        if (split_pattern[index]) {
            split_seconds += elapsed;
            split_count++;
            printf("  AB split %d %.3fs\n", interval, elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT command AB baseline %.4fs, split-%d %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           interval, split_seconds / split_count,
           split_seconds * baseline_count /
               (baseline_seconds * split_count));
    free(video_reference);
    free(audio_reference);
}

static void run_graph_data_ab(h3_dit *dit, float *video, float *audio,
                              float *video_velocity, float *audio_velocity) {
    char error[512];
    setenv("H3_DISABLE_GRAPH_DATA_CACHE", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating graph-data AB references");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int cache_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double cached_seconds = 0.0;
    int baseline_count = 0;
    int cached_count = 0;
    for (size_t index = 0;
         index < sizeof(cache_pattern) / sizeof(*cache_pattern); index++) {
        if (cache_pattern[index]) unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
        else setenv("H3_DISABLE_GRAPH_DATA_CACHE", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("graph-data cache changed DiT output bytes");
        if (cache_pattern[index]) {
            cached_seconds += elapsed;
            cached_count++;
            printf("  AB graph-data cached %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB graph-data baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_GRAPH_DATA_CACHE");
    printf("DiT graph-data AB baseline %.4fs, cached-weights %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           cached_seconds / cached_count,
           cached_seconds * baseline_count /
               (baseline_seconds * cached_count));
    free(video_reference);
    free(audio_reference);
}

static void run_mps_command_ab(h3_dit *dit, float *video, float *audio,
                               float *video_velocity,
                               float *audio_velocity) {
    char error[512];
    setenv("H3_REUSE_MPS_COMMAND", "0", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating MPS-command AB references");
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    setenv("H3_REUSE_MPS_COMMAND", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int reuse_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double reused_seconds = 0.0;
    int baseline_count = 0;
    int reused_count = 0;
    for (size_t index = 0;
         index < sizeof(reuse_pattern) / sizeof(*reuse_pattern); index++) {
        if (reuse_pattern[index]) setenv("H3_REUSE_MPS_COMMAND", "1", 1);
        else setenv("H3_REUSE_MPS_COMMAND", "0", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_reference, video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_reference, audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("MPS-command reuse changed DiT output bytes");
        if (reuse_pattern[index]) {
            reused_seconds += elapsed;
            reused_count++;
            printf("  AB MPS command reused %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  AB MPS command baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_REUSE_MPS_COMMAND");
    printf("DiT MPS-command AB baseline %.4fs, reused %.4fs, ratio %.4f, "
           "outputs byte-identical\n", baseline_seconds / baseline_count,
           reused_seconds / reused_count,
           reused_seconds * baseline_count /
               (baseline_seconds * reused_count));
    free(video_reference);
    free(audio_reference);
}

static double relative_l2(const float *got, const float *want, size_t count,
                          double *maximum_absolute) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        double magnitude = fabs(delta);
        squared_error += delta * delta;
        squared_reference += (double)want[index] * want[index];
        if (magnitude > maximum) maximum = magnitude;
    }
    *maximum_absolute = maximum;
    return sqrt(squared_error / (squared_reference > 1e-30
                                 ? squared_reference : 1e-30));
}

static void run_qkv_ab(h3_dit *dit, float *video, float *audio,
                       float *video_velocity, float *audio_velocity) {
    char error[512];
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating cooperative QKV AB references");

    setenv("H3_DISABLE_COOP_QKV", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_COOP_QKV");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double candidate_seconds = 0.0;
    int baseline_count = 0;
    int candidate_count = 0;
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    for (size_t index = 0;
         index < sizeof(candidate_pattern) / sizeof(*candidate_pattern);
         index++) {
        if (candidate_pattern[index])
            unsetenv("H3_DISABLE_COOP_QKV");
        else
            setenv("H3_DISABLE_COOP_QKV", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (candidate_pattern[index]) {
            video_rel = relative_l2(video_velocity, video_reference,
                                    VIDEO_ELEMENTS, &video_abs);
            audio_rel = relative_l2(audio_velocity, audio_reference,
                                    AUDIO_ELEMENTS, &audio_abs);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  cooperative QKV %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", elapsed, video_rel, audio_rel);
        } else {
            if (memcmp(video_velocity, video_reference,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_velocity, audio_reference,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die("repeated scalar QKV changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  scalar QKV %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_COOP_QKV");
    printf("DiT cooperative QKV AB scalar %.4fs, cooperative %.4fs, "
           "ratio %.4f; video relL2 %.6g max %.6g; audio relL2 %.6g "
           "max %.6g\n", baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(video_reference);
    free(audio_reference);
}

static void run_qkv_cache_ab(h3_dit *dit, float *video, float *audio,
                             float *video_velocity,
                             float *audio_velocity) {
    char error[512];
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating cached QKV AB references");
    setenv("H3_DISABLE_CACHED_QKV", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_CACHED_QKV");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int cached_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double uncached_seconds = 0.0, cached_seconds = 0.0;
    int uncached_count = 0, cached_count = 0;
    for (size_t run = 0;
         run < sizeof(cached_pattern) / sizeof(*cached_pattern); run++) {
        if (cached_pattern[run])
            unsetenv("H3_DISABLE_CACHED_QKV");
        else
            setenv("H3_DISABLE_CACHED_QKV", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_velocity, video_reference,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_velocity, audio_reference,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("cached cooperative QKV changed output bytes");
        if (cached_pattern[run]) {
            cached_seconds += elapsed;
            cached_count++;
            printf("  cached cooperative QKV %.3fs\n", elapsed);
        } else {
            uncached_seconds += elapsed;
            uncached_count++;
            printf("  uncached cooperative QKV %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_CACHED_QKV");
    printf("DiT cached cooperative QKV AB uncached %.4fs, cached %.4fs, "
           "ratio %.4f; outputs byte-identical\n",
           uncached_seconds / uncached_count,
           cached_seconds / cached_count,
           cached_seconds * uncached_count /
               (uncached_seconds * cached_count));
    free(video_reference);
    free(audio_reference);
}

static void run_exact_simd_gate_adaln_ab(
                             h3_dit *dit, float *video, float *audio,
                             float *video_velocity, float *audio_velocity) {
    char error[512];
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating exact SIMD gate AdaLN AB references");
    setenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        if (candidate_pattern[run])
            unsetenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN");
        else
            setenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_velocity, video_reference,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_velocity, audio_reference,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("exact SIMD gate AdaLN changed output bytes");
        if (candidate_pattern[run]) {
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  exact SIMD gate AdaLN %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  shared-tree gate AdaLN %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN");
    printf("DiT exact SIMD gate AdaLN AB shared-tree %.4fs, SIMD %.4fs, "
           "ratio %.4f; outputs byte-identical\n",
           baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count));
    free(video_reference);
    free(audio_reference);
}

static void run_qkv_denoise_ab(h3_dit *dit, float *video, float *audio,
                               int reuse_interval) {
    char error[512];
    float *initial_video = malloc(VIDEO_ELEMENTS * sizeof(*initial_video));
    float *initial_audio = malloc(AUDIO_ELEMENTS * sizeof(*initial_audio));
    float *reference_video = malloc(VIDEO_ELEMENTS * sizeof(*reference_video));
    float *reference_audio = malloc(AUDIO_ELEMENTS * sizeof(*reference_audio));
    if (!initial_video || !initial_audio || !reference_video || !reference_audio)
        die("out of memory allocating cooperative QKV denoise AB state");
    memcpy(initial_video, video, VIDEO_ELEMENTS * sizeof(*initial_video));
    memcpy(initial_audio, audio, AUDIO_ELEMENTS * sizeof(*initial_audio));

    static const int candidate_pattern[] = {0, 1, 1, 0};
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0, have_reference = 0;
    uint64_t video_hash = 0, audio_hash = 0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        memcpy(video, initial_video, VIDEO_ELEMENTS * sizeof(*video));
        memcpy(audio, initial_audio, AUDIO_ELEMENTS * sizeof(*audio));
        if (candidate_pattern[run])
            unsetenv("H3_DISABLE_COOP_QKV");
        else
            setenv("H3_DISABLE_COOP_QKV", "1", 1);
        double start = seconds();
        if (!h3_dit_denoise_euler(dit, video, audio, reuse_interval, NULL, NULL,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (!have_reference) {
            memcpy(reference_video, video,
                   VIDEO_ELEMENTS * sizeof(*reference_video));
            memcpy(reference_audio, audio,
                   AUDIO_ELEMENTS * sizeof(*reference_audio));
            video_hash = hash_bytes(video, VIDEO_ELEMENTS * sizeof(*video));
            audio_hash = hash_bytes(audio, AUDIO_ELEMENTS * sizeof(*audio));
            have_reference = 1;
        } else if (memcmp(video, reference_video,
                          VIDEO_ELEMENTS * sizeof(*reference_video)) ||
                   memcmp(audio, reference_audio,
                          AUDIO_ELEMENTS * sizeof(*reference_audio))) {
            die("cooperative QKV changed denoised latent bytes");
        }
        if (candidate_pattern[run]) {
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  cooperative QKV denoise %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  scalar QKV denoise %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_COOP_QKV");
    printf("DiT cooperative QKV denoise AB scalar %.4fs, cooperative %.4fs, "
           "ratio %.4f; byte-identical hashes video %016llx audio %016llx\n",
           baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           (unsigned long long)video_hash, (unsigned long long)audio_hash);
    free(initial_video); free(initial_audio);
    free(reference_video); free(reference_audio);
}

typedef struct {
    int waiting;
    int step;
    int invert;
    int candidate_count;
    int scalar_count;
    double started;
    double candidate_seconds;
    double scalar_seconds;
} qkv_step_ab_state;

static void qkv_step_ab_progress(const char *phase, int completed, int total,
                                 void *opaque) {
    qkv_step_ab_state *state = opaque;
    if (strcmp(phase, "denoise")) return;
    if (!state->waiting) {
        if (completed >= total) return;
        static const int pattern[] = {0, 1, 1, 0};
        int candidate = pattern[completed % 4] ^ state->invert;
        if (candidate)
            unsetenv("H3_DISABLE_COOP_QKV");
        else
            setenv("H3_DISABLE_COOP_QKV", "1", 1);
        state->step = completed;
        state->started = seconds();
        state->waiting = 1;
        return;
    }
    if (completed != state->step + 1) return;
    double elapsed = seconds() - state->started;
    static const int pattern[] = {0, 1, 1, 0};
    int candidate = pattern[state->step % 4] ^ state->invert;
    if (candidate) {
        state->candidate_seconds += elapsed;
        state->candidate_count++;
        printf("  step %d cooperative QKV %.3fs\n", state->step, elapsed);
    } else {
        state->scalar_seconds += elapsed;
        state->scalar_count++;
        printf("  step %d scalar QKV %.3fs\n", state->step, elapsed);
    }
    state->waiting = 0;
}

static void run_qkv_step_ab(h3_dit *dit, float *video, float *audio,
                            float *video_velocity, float *audio_velocity,
                            int reuse_interval) {
    char error[512];
    setenv("H3_DISABLE_COOP_QKV", "1", 1);
    if (!h3_dit_forward(dit, 0, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    unsetenv("H3_DISABLE_COOP_QKV");
    if (!h3_dit_forward(dit, 0, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    qkv_step_ab_state state = {0};
    state.invert = getenv("H3_BENCH_QKV_INVERT") != NULL;
    setenv("H3_CPU_SAMPLER", "1", 1);
    if (!h3_dit_denoise_euler(dit, video, audio, reuse_interval,
                              qkv_step_ab_progress, &state,
                              error, sizeof(error))) die(error);
    unsetenv("H3_CPU_SAMPLER");
    unsetenv("H3_DISABLE_COOP_QKV");
    double scalar = state.scalar_seconds / state.scalar_count;
    double candidate = state.candidate_seconds / state.candidate_count;
    printf("DiT cooperative QKV per-step AB scalar %.4fs (%d), "
           "cooperative %.4fs (%d), ratio %.4f; hashes video %016llx "
           "audio %016llx\n", scalar, state.scalar_count, candidate,
           state.candidate_count, candidate / scalar,
           (unsigned long long)hash_bytes(
               video, VIDEO_ELEMENTS * sizeof(*video)),
           (unsigned long long)hash_bytes(
               audio, AUDIO_ELEMENTS * sizeof(*audio)));
}

static void run_nax_mlp_ab(h3_dit *dit, float *video, float *audio,
                           float *video_velocity, float *audio_velocity) {
    char error[512];
    int quant_ab = getenv("H3_BENCH_INT8_QUANT_AB") != NULL;
    int local_ab = getenv("H3_BENCH_INT8_GROUP_LOCAL_AB") != NULL;
    int local128_ab =
        getenv("H3_BENCH_INT8_GROUP_LOCAL128_AB") != NULL;
    int fc1local_ab = getenv("H3_BENCH_INT8_FC1_LOCAL_AB") != NULL;
    int fc1full_ab = getenv("H3_BENCH_INT8_FC1_FULL_K_AB") != NULL;
    int fc1known_ab = getenv("H3_BENCH_INT8_FC1_KNOWN_AB") != NULL;
    int group_quant128_ab =
        getenv("H3_BENCH_INT8_GROUP_QUANT128_AB") != NULL;
    int group_quant128_cached = group_quant128_ab &&
        getenv("H3_BENCH_INT8_GROUP_QUANT128_CACHED") != NULL;
    int group_quant128_cache_ab =
        getenv("H3_BENCH_INT8_GROUP_QUANT128_CACHE_AB") != NULL;
    int int8_ab = quant_ab || local_ab || local128_ab || fc1local_ab ||
        fc1full_ab ||
        fc1known_ab ||
        group_quant128_cache_ab ||
        group_quant128_ab ||
        getenv("H3_BENCH_INT8_MLP_AB") != NULL;
    const char *disable = int8_ab ? "H3_DISABLE_INT8_MLP" :
                                    "H3_DISABLE_NAX_MLP";
    const char *candidate_name = quant_ab ? "vec4 int8 quantizer" :
        local_ab ? "local grouped FC2" :
        local128_ab ? "128x128 local grouped FC2" :
        fc1local_ab ? "local FC1" :
        fc1full_ab ? "full-K FC1" :
        fc1known_ab ? "compile-time-K FC1" :
        group_quant128_cache_ab ? "register-cached grouped quantizer" :
        group_quant128_ab ? (group_quant128_cached ?
            "register-cached 128-thread grouped quantizer" :
            "128-thread grouped quantizer") :
        int8_ab ? "int8" : "NAX";
    char candidate_blocks[16] = {0};
    const char *blocks = getenv("H3_BENCH_NAX_COMMAND_BLOCKS");
    if (blocks) snprintf(candidate_blocks, sizeof(candidate_blocks), "%s",
                         blocks);
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating NAX MLP AB references");
    if (quant_ab || local_ab || local128_ab || fc1local_ab || fc1full_ab ||
        fc1known_ab ||
        group_quant128_cache_ab ||
        group_quant128_ab) {
        unsetenv("H3_INT8_FC1_LOCAL");
        if (fc1full_ab) setenv("H3_DISABLE_FC1_FULL_K", "1", 1);
        if (fc1known_ab) setenv("H3_INT8_FC1_KNOWN", "0", 1);
        else unsetenv("H3_INT8_FC1_KNOWN");
        unsetenv("H3_INT8_GROUP_QUANT_128_CACHE");
        unsetenv("H3_INT8_GROUP_QUANT_128");
        if (group_quant128_ab)
            setenv("H3_INT8_GROUP_QUANT_128", "0", 1);
        else if (group_quant128_cache_ab)
            setenv("H3_INT8_GROUP_QUANT_128", "1", 1);
        if (group_quant128_cache_ab)
            setenv("H3_INT8_GROUP_QUANT_128_CACHE", "0", 1);
        if (!fc1local_ab) {
            unsetenv("H3_INT8_VECTOR_QUANT");
            unsetenv("H3_INT8_GROUP_FC2_LOCAL");
            unsetenv("H3_INT8_GROUP_FC2_LOCAL128");
            if (local128_ab)
                setenv("H3_INT8_GROUP_FC2_LOCAL", "1", 1);
        }
    }
    else setenv(disable, "1", 1);
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (quant_ab) setenv("H3_INT8_VECTOR_QUANT", "1", 1);
    else if (local_ab) setenv("H3_INT8_GROUP_FC2_LOCAL", "1", 1);
    else if (local128_ab) {
        unsetenv("H3_INT8_GROUP_FC2_LOCAL");
        setenv("H3_INT8_GROUP_FC2_LOCAL128", "1", 1);
    }
    else if (fc1local_ab) setenv("H3_INT8_FC1_LOCAL", "1", 1);
    else if (fc1full_ab) unsetenv("H3_DISABLE_FC1_FULL_K");
    else if (fc1known_ab) setenv("H3_INT8_FC1_KNOWN", "1", 1);
    else if (group_quant128_cache_ab)
        setenv("H3_INT8_GROUP_QUANT_128_CACHE", "1", 1);
    else if (group_quant128_ab) {
        setenv("H3_INT8_GROUP_QUANT_128", "1", 1);
        if (group_quant128_cached)
            setenv("H3_INT8_GROUP_QUANT_128_CACHE", "1", 1);
    }
    else unsetenv(disable);
    if (candidate_blocks[0])
        setenv("H3_DIT_COMMAND_BLOCKS", candidate_blocks, 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    if (int8_ab && getenv("H3_BENCH_INT8_QUICK")) {
        double video_abs = 0.0, audio_abs = 0.0;
        double video_rel = relative_l2(video_velocity, video_reference,
                                       VIDEO_ELEMENTS, &video_abs);
        double audio_rel = relative_l2(audio_velocity, audio_reference,
                                       AUDIO_ELEMENTS, &audio_abs);
        printf("DiT int8 MLP quick drift video relL2 %.6g max %.6g; "
               "audio relL2 %.6g max %.6g\n",
               video_rel, video_abs, audio_rel, audio_abs);
        unsetenv(disable);
        unsetenv("H3_DIT_COMMAND_BLOCKS");
        free(video_reference);
        free(audio_reference);
        return;
    }

    static const int nax_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double nax_seconds = 0.0;
    int baseline_count = 0;
    int nax_count = 0;
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    int invert = getenv("H3_BENCH_MLP_INVERT") != NULL;
    for (size_t index = 0;
        index < sizeof(nax_pattern) / sizeof(*nax_pattern); index++) {
        int candidate_turn = nax_pattern[index] != invert;
        if (candidate_turn) {
            if (quant_ab) setenv("H3_INT8_VECTOR_QUANT", "1", 1);
            else if (local_ab)
                setenv("H3_INT8_GROUP_FC2_LOCAL", "1", 1);
            else if (local128_ab) {
                unsetenv("H3_INT8_GROUP_FC2_LOCAL");
                setenv("H3_INT8_GROUP_FC2_LOCAL128", "1", 1);
            }
            else if (fc1local_ab) setenv("H3_INT8_FC1_LOCAL", "1", 1);
            else if (fc1full_ab) unsetenv("H3_DISABLE_FC1_FULL_K");
            else if (fc1known_ab) setenv("H3_INT8_FC1_KNOWN", "1", 1);
            else if (group_quant128_cache_ab)
                setenv("H3_INT8_GROUP_QUANT_128_CACHE", "1", 1);
            else if (group_quant128_ab) {
                setenv("H3_INT8_GROUP_QUANT_128", "1", 1);
                if (group_quant128_cached)
                    setenv("H3_INT8_GROUP_QUANT_128_CACHE", "1", 1);
            }
            else unsetenv(disable);
            if (candidate_blocks[0])
                setenv("H3_DIT_COMMAND_BLOCKS", candidate_blocks, 1);
            else unsetenv("H3_DIT_COMMAND_BLOCKS");
        } else {
            if (quant_ab) unsetenv("H3_INT8_VECTOR_QUANT");
            else if (local_ab) unsetenv("H3_INT8_GROUP_FC2_LOCAL");
            else if (local128_ab) {
                unsetenv("H3_INT8_GROUP_FC2_LOCAL128");
                setenv("H3_INT8_GROUP_FC2_LOCAL", "1", 1);
            }
            else if (fc1local_ab) unsetenv("H3_INT8_FC1_LOCAL");
            else if (fc1full_ab)
                setenv("H3_DISABLE_FC1_FULL_K", "1", 1);
            else if (fc1known_ab)
                setenv("H3_INT8_FC1_KNOWN", "0", 1);
            else if (group_quant128_cache_ab)
                setenv("H3_INT8_GROUP_QUANT_128_CACHE", "0", 1);
            else if (group_quant128_ab) {
                unsetenv("H3_INT8_GROUP_QUANT_128_CACHE");
                setenv("H3_INT8_GROUP_QUANT_128", "0", 1);
            }
            else setenv(disable, "1", 1);
            unsetenv("H3_DIT_COMMAND_BLOCKS");
        }
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (candidate_turn) {
            if ((quant_ab || local_ab || local128_ab || fc1local_ab ||
                 fc1full_ab ||
                 fc1known_ab ||
                 group_quant128_cache_ab ||
                 group_quant128_ab) &&
                (memcmp(video_velocity, video_reference,
                        VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                 memcmp(audio_velocity, audio_reference,
                        AUDIO_ELEMENTS * sizeof(*audio_reference))))
                die(quant_ab ? "vec4 int8 quantizer changed output bytes" :
                    local128_ab ? "128x128 grouped FC2 changed output bytes" :
                    fc1local_ab ? "local FC1 changed output bytes" :
                    fc1full_ab ? "full-K FC1 changed output bytes" :
                    fc1known_ab ? "compile-time-K FC1 changed output bytes" :
                    group_quant128_cache_ab ?
                        "register-cached grouped quantizer changed output bytes" :
                    group_quant128_ab ?
                        "128-thread grouped quantizer changed output bytes" :
                                  "local grouped FC2 changed output bytes");
            video_rel = relative_l2(video_velocity, video_reference,
                                    VIDEO_ELEMENTS, &video_abs);
            audio_rel = relative_l2(audio_velocity, audio_reference,
                                    AUDIO_ELEMENTS, &audio_abs);
            nax_seconds += elapsed;
            nax_count++;
            printf("  %s MLP AB candidate %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", candidate_name, elapsed,
                   video_rel, audio_rel);
        } else {
            if (memcmp(video_velocity, video_reference,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_velocity, audio_reference,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die((quant_ab || local_ab || local128_ab || fc1local_ab ||
                     fc1full_ab ||
                     fc1known_ab ||
                     group_quant128_cache_ab ||
                     group_quant128_ab) ?
                               "int8 kernel baseline changed output bytes" :
                               "repeated MPSGraph MLP changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  %s MLP AB baseline %.3fs\n", candidate_name, elapsed);
        }
    }
    unsetenv(disable);
    unsetenv("H3_INT8_VECTOR_QUANT");
    unsetenv("H3_INT8_GROUP_FC2_LOCAL");
    unsetenv("H3_INT8_GROUP_FC2_LOCAL128");
    unsetenv("H3_INT8_FC1_LOCAL");
    unsetenv("H3_DISABLE_FC1_FULL_K");
    unsetenv("H3_INT8_FC1_KNOWN");
    unsetenv("H3_INT8_GROUP_QUANT_128_CACHE");
    unsetenv("H3_INT8_GROUP_QUANT_128");
    unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT %s MLP AB baseline %.4fs, candidate %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           candidate_name, baseline_seconds / baseline_count,
           nax_seconds / nax_count,
           nax_seconds * baseline_count / (baseline_seconds * nax_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(video_reference);
    free(audio_reference);
}

static void run_int8_projection_ab(h3_dit *dit, float *video, float *audio,
                                   float *video_velocity,
                                   float *audio_velocity) {
    char error[512];
    int full_k_qkv = getenv("H3_BENCH_QKV_FULL_K_AB") != NULL;
    int head_major_attention_output = !full_k_qkv &&
        getenv("H3_BENCH_HEAD_MAJOR_ATTENTION_OUT_AB") != NULL;
    int vector_gate_adaln = !head_major_attention_output &&
        getenv("H3_BENCH_VECTOR_GATE_ADALN_AB") != NULL;
    int vector_qkv_rope = !vector_gate_adaln &&
        getenv("H3_BENCH_VECTOR_QKV_ROPE_AB") != NULL;
    int known_linear = !vector_qkv_rope &&
        getenv("H3_BENCH_INT8_LINEAR_KNOWN_AB") != NULL;
    int local_qkv_scales = !known_linear &&
        getenv("H3_BENCH_QKV_LOCAL_SCALES_AB") != NULL;
    int local_linear_scales = !local_qkv_scales &&
        getenv("H3_BENCH_INT8_LOCAL_SCALES_AB") != NULL;
    int vector_qkv_rms = !local_linear_scales &&
        getenv("H3_BENCH_VECTOR_QKV_RMS_AB") != NULL;
    int fused_qkv_rope = !vector_qkv_rms &&
        getenv("H3_BENCH_FUSED_INT8_QKV_ROPE_AB") != NULL;
    int fused_inputs = !fused_qkv_rope &&
        getenv("H3_BENCH_FUSED_INT8_INPUTS_AB") != NULL;
    int fused_qkv_input = !fused_inputs &&
        getenv("H3_BENCH_FUSED_INT8_QKV_INPUT_AB") != NULL;
    int fused_mlp_input = !fused_qkv_input &&
        getenv("H3_BENCH_FUSED_INT8_MLP_INPUT_AB") != NULL;
    int attention_out = !fused_mlp_input &&
        getenv("H3_BENCH_INT8_ATTENTION_OUT_AB") != NULL;
    const char *disable = full_k_qkv ?
        "H3_DISABLE_QKV_FULL_K" : head_major_attention_output ?
        "H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT" : vector_gate_adaln ?
        "H3_DISABLE_VECTOR_GATE_ADALN" : vector_qkv_rope ?
        "H3_DISABLE_VECTOR_QKV_ROPE" : known_linear ?
        "H3_DISABLE_INT8_LINEAR_KNOWN" : local_qkv_scales ?
        "H3_DISABLE_QKV_LOCAL_SCALES" : local_linear_scales ?
        "H3_DISABLE_INT8_LOCAL_SCALES" : vector_qkv_rms ?
        "H3_DISABLE_VECTOR_QKV_RMS" : fused_qkv_rope ?
        "H3_DISABLE_FUSED_INT8_QKV_ROPE" : fused_inputs ?
        "H3_DISABLE_FUSED_INT8_MLP_INPUT" : fused_qkv_input ?
        "H3_DISABLE_FUSED_INT8_QKV_INPUT" : fused_mlp_input ?
        "H3_DISABLE_FUSED_INT8_MLP_INPUT" : attention_out ?
        "H3_DISABLE_INT8_ATTENTION_OUT" : "H3_DISABLE_INT8_QKV";
    const char *disable2 = fused_inputs ?
        "H3_DISABLE_FUSED_INT8_QKV_INPUT" : NULL;
    const char *name = full_k_qkv ?
        "full-K int8 QKV" : head_major_attention_output ?
        "head-major attention output" : vector_gate_adaln ?
        "vector fused int8 gate AdaLN" : vector_qkv_rope ?
        "vector fused int8 QKV RoPE" : known_linear ?
        "compile-time-shape int8 attention output" :
        local_qkv_scales ? "local int8 QKV scales" :
        local_linear_scales ? "local int8 linear scales" :
        vector_qkv_rms ? "vector int8 QKV RMS" :
        fused_qkv_rope ? "fused int8 QKV norm/RoPE" :
        fused_inputs ? "fused int8 projection inputs" :
        fused_qkv_input ? "fused int8 QKV input" :
        fused_mlp_input ? "fused int8 MLP input" :
        attention_out ? "int8 attention output" : "int8 QKV";
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating int8 projection AB references");
    setenv(disable, "1", 1);
    if (disable2) setenv(disable2, "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv(disable);
    if (disable2) unsetenv(disable2);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    int invert = getenv("H3_BENCH_PROJECTION_INVERT") != NULL;
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0;
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    for (size_t index = 0;
         index < sizeof(candidate_pattern) / sizeof(*candidate_pattern);
         index++) {
        int candidate = candidate_pattern[index] ^ invert;
        if (candidate) {
            unsetenv(disable);
            if (disable2) unsetenv(disable2);
        } else {
            setenv(disable, "1", 1);
            if (disable2) setenv(disable2, "1", 1);
        }
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (candidate) {
            if ((full_k_qkv || head_major_attention_output ||
                 vector_gate_adaln ||
                 vector_qkv_rope || known_linear) &&
                (memcmp(video_velocity, video_reference,
                        VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                 memcmp(audio_velocity, audio_reference,
                        AUDIO_ELEMENTS * sizeof(*audio_reference))))
                die("layout/specialized int8 attention output changed bytes");
            video_rel = relative_l2(video_velocity, video_reference,
                                    VIDEO_ELEMENTS, &video_abs);
            audio_rel = relative_l2(audio_velocity, audio_reference,
                                    AUDIO_ELEMENTS, &audio_abs);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  %s AB candidate %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", name, elapsed, video_rel, audio_rel);
        } else {
            if (memcmp(video_velocity, video_reference,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_velocity, audio_reference,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die("repeated BF16 projection baseline changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  %s AB baseline %.3fs\n", name, elapsed);
        }
    }
    unsetenv(disable);
    if (disable2) unsetenv(disable2);
    printf("DiT %s AB baseline %.4fs, candidate %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           name, baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(video_reference);
    free(audio_reference);
}

static void run_nax_morton_ab(h3_dit *dit, float *video, float *audio,
                              float *video_velocity,
                              float *audio_velocity) {
    char error[512];
    int compact_ab = getenv("H3_BENCH_NAX_MORTON4_AB") != NULL;
    const char *disable = compact_ab ? "H3_DISABLE_NAX_MORTON4" :
                                       "H3_DISABLE_NAX_MORTON";
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating NAX Morton AB references");
    setenv(disable, "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv(disable);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int pattern[] = {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
    double row_major_seconds = 0.0, morton_seconds = 0.0;
    int row_major_count = 0, morton_count = 0;
    for (size_t run = 0; run < sizeof(pattern) / sizeof(*pattern); run++) {
        if (pattern[run]) unsetenv(disable);
        else setenv(disable, "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (memcmp(video_velocity, video_reference,
                   VIDEO_ELEMENTS * sizeof(*video_reference)) ||
            memcmp(audio_velocity, audio_reference,
                   AUDIO_ELEMENTS * sizeof(*audio_reference)))
            die("NAX Morton order changed output bytes");
        if (pattern[run]) {
            morton_seconds += elapsed;
            morton_count++;
            printf("  NAX %s Morton %.3fs\n",
                   compact_ab ? "compact" : "padded", elapsed);
        } else {
            row_major_seconds += elapsed;
            row_major_count++;
            printf("  NAX %s %.3fs\n",
                   compact_ab ? "padded Morton" : "row-major", elapsed);
        }
    }
    unsetenv(disable);
    printf("DiT NAX Morton AB %s %.4fs, %s %.4fs, ratio %.4f; "
           "outputs byte-identical\n",
           compact_ab ? "padded" : "row-major",
           row_major_seconds / row_major_count,
           compact_ab ? "compact" : "Morton",
           morton_seconds / morton_count,
           morton_seconds * row_major_count /
               (row_major_seconds * morton_count));
    free(video_reference);
    free(audio_reference);
}

static void run_nax_linear_ab(h3_dit *dit, float *video, float *audio,
                              float *video_velocity,
                              float *audio_velocity) {
    char error[512];
    int split_boundary_ab =
        getenv("H3_BENCH_NAX_SPLIT_BOUNDARY_AB") != NULL;
    int head_major_ab = getenv("H3_BENCH_HEAD_MAJOR_SDPA_AB") != NULL;
    int split_qkv_ab = getenv("H3_BENCH_SPLIT_NAX_QKV_AB") != NULL;
    int schedule_ab = getenv("H3_BENCH_NAX_LINEAR_MORTON_AB") != NULL;
    const char *disable = split_boundary_ab ? "H3_NAX_SPLIT_ROWS" :
        head_major_ab ? "H3_DISABLE_HEAD_MAJOR_SDPA" :
        split_qkv_ab ? "H3_DISABLE_SPLIT_NAX_QKV" :
        schedule_ab ? "H3_DISABLE_NAX_LINEAR_MORTON4" :
                      "H3_DISABLE_NAX_LINEAR";
    const char *baseline_name = split_boundary_ab ? "split 1536" :
        head_major_ab ? "row-major split QKV" :
        split_qkv_ab ? "grouped NAX QKV" :
        schedule_ab ? "row-major NAX" : "MPSGraph";
    const char *candidate_name = split_boundary_ab ? "split 2048" :
        head_major_ab ? "head-major split QKV" :
        split_qkv_ab ? "split NAX QKV" :
        schedule_ab ? "compact NAX" : "NAX";
    float *baseline_video = malloc(VIDEO_ELEMENTS * sizeof(*baseline_video));
    float *baseline_audio = malloc(AUDIO_ELEMENTS * sizeof(*baseline_audio));
    float *candidate_video = malloc(VIDEO_ELEMENTS * sizeof(*candidate_video));
    float *candidate_audio = malloc(AUDIO_ELEMENTS * sizeof(*candidate_audio));
    if (!baseline_video || !baseline_audio ||
        !candidate_video || !candidate_audio)
        die("out of memory allocating NAX linear AB references");
    h3_gpu_stats stats_before, stats_baseline, stats_candidate;
    if (!h3_dit_get_gpu_stats(dit, &stats_before))
        die("cannot read NAX linear baseline GPU stats");
    setenv(disable, split_boundary_ab ? "1536" : "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    if (!h3_dit_get_gpu_stats(dit, &stats_baseline))
        die("cannot read NAX linear baseline GPU stats");
    memcpy(baseline_video, video_velocity,
           VIDEO_ELEMENTS * sizeof(*baseline_video));
    memcpy(baseline_audio, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*baseline_audio));
    if (split_boundary_ab) setenv(disable, "2048", 1);
    else unsetenv(disable);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    if (!h3_dit_get_gpu_stats(dit, &stats_candidate))
        die("cannot read NAX linear candidate GPU stats");
    memcpy(candidate_video, video_velocity,
           VIDEO_ELEMENTS * sizeof(*candidate_video));
    memcpy(candidate_audio, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*candidate_audio));

    static const int pattern[] = {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0;
    for (size_t run = 0; run < sizeof(pattern) / sizeof(*pattern); run++) {
        if (pattern[run]) {
            if (split_boundary_ab) setenv(disable, "2048", 1);
            else unsetenv(disable);
        } else {
            setenv(disable, split_boundary_ab ? "1536" : "1", 1);
        }
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        const float *expected_video = pattern[run] ?
            candidate_video : baseline_video;
        const float *expected_audio = pattern[run] ?
            candidate_audio : baseline_audio;
        if (memcmp(video_velocity, expected_video,
                   VIDEO_ELEMENTS * sizeof(*expected_video)) ||
            memcmp(audio_velocity, expected_audio,
                   AUDIO_ELEMENTS * sizeof(*expected_audio)))
            die("NAX linear output changed between repetitions");
        if (pattern[run]) {
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  %s %.3fs\n", candidate_name, elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  %s %.3fs\n", baseline_name, elapsed);
        }
    }
    double video_abs = 0.0, audio_abs = 0.0;
    double video_rel = relative_l2(candidate_video, baseline_video,
                                   VIDEO_ELEMENTS, &video_abs);
    double audio_rel = relative_l2(candidate_audio, baseline_audio,
                                   AUDIO_ELEMENTS, &audio_abs);
    unsetenv(disable);
    printf("DiT NAX linear AB %s %.4fs, %s %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           baseline_name, baseline_seconds / baseline_count,
           candidate_name, candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_rel, video_abs, audio_rel, audio_abs);
    printf("NAX linear dispatches %s: direct %llu, MPS linear %llu; "
           "%s: direct %llu, MPS linear %llu\n",
           baseline_name,
           (unsigned long long)(stats_baseline.direct_dispatches -
                                stats_before.direct_dispatches),
           (unsigned long long)(stats_baseline.mps_linear_dispatches -
                                stats_before.mps_linear_dispatches),
           candidate_name,
           (unsigned long long)(stats_candidate.direct_dispatches -
                                stats_baseline.direct_dispatches),
           (unsigned long long)(stats_candidate.mps_linear_dispatches -
                                stats_baseline.mps_linear_dispatches));
    free(baseline_video); free(baseline_audio);
    free(candidate_video); free(candidate_audio);
}

static void run_sampler_ab(h3_dit *dit, float *video, float *audio,
                           float *video_velocity, float *audio_velocity) {
    char error[512];
    float *initial_video = malloc(VIDEO_ELEMENTS * sizeof(*initial_video));
    float *initial_audio = malloc(AUDIO_ELEMENTS * sizeof(*initial_audio));
    float *reference_video = malloc(VIDEO_ELEMENTS * sizeof(*reference_video));
    float *reference_audio = malloc(AUDIO_ELEMENTS * sizeof(*reference_audio));
    if (!initial_video || !initial_audio || !reference_video || !reference_audio)
        die("out of memory allocating sampler AB state");
    memcpy(initial_video, video, VIDEO_ELEMENTS * sizeof(*initial_video));
    memcpy(initial_audio, audio, AUDIO_ELEMENTS * sizeof(*initial_audio));

    /* Populate the shape-keyed MPSGraph caches outside the measured runs. */
    if (!h3_dit_forward(dit, 0, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    static const int gpu_pattern[] = {0, 1, 1, 0, 1, 0, 0, 1};
    double cpu_seconds = 0.0;
    double gpu_seconds = 0.0;
    int cpu_count = 0;
    int gpu_count = 0;
    int have_reference = 0;
    const char *gpu_command_blocks = getenv("H3_BENCH_GPU_COMMAND_BLOCKS");
    double video_rel = 0.0, audio_rel = 0.0;
    double video_abs = 0.0, audio_abs = 0.0;
    for (size_t run = 0; run < sizeof(gpu_pattern) / sizeof(*gpu_pattern);
         run++) {
        memcpy(video, initial_video, VIDEO_ELEMENTS * sizeof(*video));
        memcpy(audio, initial_audio, AUDIO_ELEMENTS * sizeof(*audio));
        if (gpu_pattern[run]) {
            unsetenv("H3_CPU_SAMPLER");
            setenv("H3_GPU_SAMPLER", "1", 1);
            if (gpu_command_blocks)
                setenv("H3_DIT_COMMAND_BLOCKS", gpu_command_blocks, 1);
        } else {
            setenv("H3_CPU_SAMPLER", "1", 1);
            unsetenv("H3_GPU_SAMPLER");
            if (gpu_command_blocks) unsetenv("H3_DIT_COMMAND_BLOCKS");
        }
        double start = seconds();
        if (!h3_dit_denoise_euler(dit, video, audio, 3, NULL, NULL,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (!have_reference && !gpu_pattern[run]) {
            memcpy(reference_video, video,
                   VIDEO_ELEMENTS * sizeof(*reference_video));
            memcpy(reference_audio, audio,
                   AUDIO_ELEMENTS * sizeof(*reference_audio));
            have_reference = 1;
        }
        if (gpu_pattern[run]) {
            video_rel = relative_l2(video, reference_video, VIDEO_ELEMENTS,
                                    &video_abs);
            audio_rel = relative_l2(audio, reference_audio, AUDIO_ELEMENTS,
                                    &audio_abs);
            gpu_seconds += elapsed;
            gpu_count++;
            printf("  sampler AB GPU %.3fs video relL2 %.6g max %.6g; "
                   "audio relL2 %.6g max %.6g\n", elapsed, video_rel,
                   video_abs, audio_rel, audio_abs);
        } else {
            if (have_reference &&
                (memcmp(video, reference_video, sizeof(*video) * VIDEO_ELEMENTS) ||
                 memcmp(audio, reference_audio, sizeof(*audio) * AUDIO_ELEMENTS)))
                die("repeated CPU sampler changed output bytes");
            cpu_seconds += elapsed;
            cpu_count++;
            printf("  sampler AB CPU %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_GPU_SAMPLER");
    unsetenv("H3_CPU_SAMPLER");
    if (gpu_command_blocks) unsetenv("H3_DIT_COMMAND_BLOCKS");
    printf("DiT sampler AB CPU %.4fs, GPU-chain %.4fs, ratio %.4f; "
           "video relL2 %.6g max %.6g; audio relL2 %.6g max %.6g\n",
           cpu_seconds / cpu_count, gpu_seconds / gpu_count,
           gpu_seconds * cpu_count / (cpu_seconds * gpu_count),
           video_rel, video_abs, audio_rel, audio_abs);
    free(initial_video);
    free(initial_audio);
    free(reference_video);
    free(reference_audio);
}

static void run_token_reduction_ab(h3_dit *dit, float *video, float *audio,
                                   float *video_velocity,
                                   float *audio_velocity) {
    char error[512];
    float *video_reference = malloc(VIDEO_ELEMENTS * sizeof(*video_reference));
    float *audio_reference = malloc(AUDIO_ELEMENTS * sizeof(*audio_reference));
    if (!video_reference || !audio_reference)
        die("out of memory allocating token-reduction AB references");
    setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    memcpy(video_reference, video_velocity,
           VIDEO_ELEMENTS * sizeof(*video_reference));
    memcpy(audio_reference, audio_velocity,
           AUDIO_ELEMENTS * sizeof(*audio_reference));
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);

    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0;
    double candidate_seconds = 0.0;
    int baseline_count = 0;
    int candidate_count = 0;
    uint64_t candidate_video_hash = 0, candidate_audio_hash = 0;
    double video_relative = 0.0, audio_relative = 0.0;
    double video_absolute = 0.0, audio_absolute = 0.0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        if (candidate_pattern[run]) unsetenv("H3_DISABLE_TOKEN_REDUCTION");
        else setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (candidate_pattern[run]) {
            uint64_t video_hash = hash_bytes(
                video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
            uint64_t audio_hash = hash_bytes(
                audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
            if (candidate_count &&
                (video_hash != candidate_video_hash ||
                 audio_hash != candidate_audio_hash))
                die("token-reduction candidate changed output bytes");
            candidate_video_hash = video_hash;
            candidate_audio_hash = audio_hash;
            video_relative = relative_l2(
                video_velocity, video_reference, VIDEO_ELEMENTS,
                &video_absolute);
            audio_relative = relative_l2(
                audio_velocity, audio_reference, AUDIO_ELEMENTS,
                &audio_absolute);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  token reduction candidate %.3fs video relL2 %.6g; "
                   "audio relL2 %.6g\n", elapsed, video_relative,
                   audio_relative);
        } else {
            if (memcmp(video_reference, video_velocity,
                       VIDEO_ELEMENTS * sizeof(*video_reference)) ||
                memcmp(audio_reference, audio_velocity,
                       AUDIO_ELEMENTS * sizeof(*audio_reference)))
                die("token-reduction baseline changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  token reduction baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    printf("DiT token reduction AB baseline %.4fs, candidate %.4fs, "
           "ratio %.4f; video relL2 %.6g max %.6g; audio relL2 %.6g "
           "max %.6g\n", baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_relative, video_absolute, audio_relative, audio_absolute);
    printf("token reduction candidate hashes video %016llx audio %016llx\n",
           (unsigned long long)candidate_video_hash,
           (unsigned long long)candidate_audio_hash);
    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats))
        die("cannot read token-reduction GPU allocation stats");
    printf("token reduction cumulative GPU allocations %.3f GiB\n",
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0));
    free(video_reference);
    free(audio_reference);
}

static void run_token_reduction_denoise_ab(h3_dit *dit, float *video,
                                   float *audio, int reuse_interval) {
    char error[512];
    float *initial_video = malloc(VIDEO_ELEMENTS * sizeof(*initial_video));
    float *initial_audio = malloc(AUDIO_ELEMENTS * sizeof(*initial_audio));
    float *reference_video = malloc(VIDEO_ELEMENTS * sizeof(*reference_video));
    float *reference_audio = malloc(AUDIO_ELEMENTS * sizeof(*reference_audio));
    if (!initial_video || !initial_audio || !reference_video || !reference_audio)
        die("out of memory allocating token-reduction denoise AB state");
    memcpy(initial_video, video, VIDEO_ELEMENTS * sizeof(*initial_video));
    memcpy(initial_audio, audio, AUDIO_ELEMENTS * sizeof(*initial_audio));
    static const int candidate_pattern[] = {0, 1, 1, 0};
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0, have_reference = 0;
    uint64_t candidate_video_hash = 0, candidate_audio_hash = 0;
    double video_relative = 0.0, audio_relative = 0.0;
    double video_absolute = 0.0, audio_absolute = 0.0;
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        memcpy(video, initial_video, VIDEO_ELEMENTS * sizeof(*video));
        memcpy(audio, initial_audio, AUDIO_ELEMENTS * sizeof(*audio));
        if (candidate_pattern[run]) unsetenv("H3_DISABLE_TOKEN_REDUCTION");
        else setenv("H3_DISABLE_TOKEN_REDUCTION", "1", 1);
        double start = seconds();
        if (!h3_dit_denoise_euler(dit, video, audio, reuse_interval, NULL, NULL,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        if (!have_reference && !candidate_pattern[run]) {
            memcpy(reference_video, video,
                   VIDEO_ELEMENTS * sizeof(*reference_video));
            memcpy(reference_audio, audio,
                   AUDIO_ELEMENTS * sizeof(*reference_audio));
            have_reference = 1;
        }
        if (candidate_pattern[run]) {
            uint64_t video_hash = hash_bytes(
                video, VIDEO_ELEMENTS * sizeof(*video));
            uint64_t audio_hash = hash_bytes(
                audio, AUDIO_ELEMENTS * sizeof(*audio));
            if (candidate_count &&
                (video_hash != candidate_video_hash ||
                 audio_hash != candidate_audio_hash))
                die("token-reduction denoise candidate changed output bytes");
            candidate_video_hash = video_hash;
            candidate_audio_hash = audio_hash;
            video_relative = relative_l2(video, reference_video,
                                         VIDEO_ELEMENTS, &video_absolute);
            audio_relative = relative_l2(audio, reference_audio,
                                         AUDIO_ELEMENTS, &audio_absolute);
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  token reduction denoise candidate %.3fs video relL2 "
                   "%.6g; audio relL2 %.6g\n", elapsed, video_relative,
                   audio_relative);
        } else {
            if (have_reference &&
                (memcmp(video, reference_video,
                        VIDEO_ELEMENTS * sizeof(*reference_video)) ||
                 memcmp(audio, reference_audio,
                        AUDIO_ELEMENTS * sizeof(*reference_audio))))
                die("token-reduction denoise baseline changed output bytes");
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  token reduction denoise baseline %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_TOKEN_REDUCTION");
    printf("DiT token reduction denoise AB baseline %.4fs, candidate %.4fs, "
           "ratio %.4f; video relL2 %.6g max %.6g; audio relL2 %.6g max "
           "%.6g\n", baseline_seconds / baseline_count,
           candidate_seconds / candidate_count,
           candidate_seconds * baseline_count /
               (baseline_seconds * candidate_count),
           video_relative, video_absolute, audio_relative, audio_absolute);
    printf("token reduction denoise reuse-%d hashes video %016llx audio "
           "%016llx\n", reuse_interval,
           (unsigned long long)candidate_video_hash,
           (unsigned long long)candidate_audio_hash);
    free(initial_video); free(initial_audio);
    free(reference_video); free(reference_audio);
}

static void run_cross_adaln_ab(h3_dit *dit, float *video, float *audio,
                               float *video_velocity,
                               float *audio_velocity) {
    char error[512];
    static const int candidate_pattern[] = {
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0
    };
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    int baseline_count = 0, candidate_count = 0;
    uint64_t video_hash = 0, audio_hash = 0;
    setenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    video_hash = hash_bytes(
        video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
    audio_hash = hash_bytes(
        audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
    unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    if (hash_bytes(video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_velocity)) != video_hash ||
        hash_bytes(audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_velocity)) != audio_hash)
        die("cross-block AdaLN fusion warmup changed output bytes");
    for (size_t run = 0;
         run < sizeof(candidate_pattern) / sizeof(*candidate_pattern); run++) {
        if (candidate_pattern[run])
            unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
        else
            setenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "1", 1);
        double start = seconds();
        if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                            audio_velocity, error, sizeof(error))) die(error);
        double elapsed = seconds() - start;
        uint64_t current_video = hash_bytes(
            video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
        uint64_t current_audio = hash_bytes(
            audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
        if (run && (current_video != video_hash || current_audio != audio_hash))
            die("cross-block AdaLN fusion changed output bytes");
        video_hash = current_video;
        audio_hash = current_audio;
        if (candidate_pattern[run]) {
            candidate_seconds += elapsed;
            candidate_count++;
            printf("  cross-block AdaLN fused %.3fs\n", elapsed);
        } else {
            baseline_seconds += elapsed;
            baseline_count++;
            printf("  cross-block AdaLN separate %.3fs\n", elapsed);
        }
    }
    unsetenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN");
    double baseline = baseline_seconds / baseline_count;
    double candidate = candidate_seconds / candidate_count;
    printf("DiT cross-block AdaLN AB separate %.4fs, fused %.4fs, "
           "ratio %.4f; hashes video %016llx audio %016llx\n",
           baseline, candidate, candidate / baseline,
           (unsigned long long)video_hash, (unsigned long long)audio_hash);
}

static void run_final_slice_ab(h3_dit *dit, float *video, float *audio,
                               float *video_velocity,
                               float *audio_velocity) {
    char error[512];
    setenv("H3_DISABLE_FUSED_FINAL_HEAD", "1", 1);
    setenv("H3_DISABLE_FUSED_FINAL_SLICE", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    uint64_t video_hash = hash_bytes(
        video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
    uint64_t audio_hash = hash_bytes(
        audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
    unsetenv("H3_DISABLE_FUSED_FINAL_SLICE");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    if (hash_bytes(video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_velocity)) != video_hash ||
        hash_bytes(audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_velocity)) != audio_hash)
        die("offset-bound final AdaLN changed output bytes");
    unsetenv("H3_DISABLE_FUSED_FINAL_HEAD");
    printf("DiT final slice/AdaLN outputs byte-identical; hashes video "
           "%016llx audio %016llx\n",
           (unsigned long long)video_hash, (unsigned long long)audio_hash);
}

static void run_final_head_ab(h3_dit *dit, float *video, float *audio,
                              float *video_velocity,
                              float *audio_velocity) {
    char error[512];
    setenv("H3_DISABLE_FUSED_FINAL_HEAD", "1", 1);
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    uint64_t video_hash = hash_bytes(
        video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity));
    uint64_t audio_hash = hash_bytes(
        audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity));
    unsetenv("H3_DISABLE_FUSED_FINAL_HEAD");
    if (!h3_dit_forward(dit, 6, video, audio, video_velocity,
                        audio_velocity, error, sizeof(error))) die(error);
    if (hash_bytes(video_velocity,
                   VIDEO_ELEMENTS * sizeof(*video_velocity)) != video_hash ||
        hash_bytes(audio_velocity,
                   AUDIO_ELEMENTS * sizeof(*audio_velocity)) != audio_hash)
        die("fused final AdaLN/head changed output bytes");
    printf("DiT fused final AdaLN/head outputs byte-identical; hashes video "
           "%016llx audio %016llx\n",
           (unsigned long long)video_hash, (unsigned long long)audio_hash);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *prompt_fixture = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_prompt_bf16.safetensors";
    uint16_t *text_values = getenv("H3_BENCH_SYNTHETIC_TEXT") ?
        calloc(TEXT_ROWS * TEXT_WIDTH, sizeof(*text_values)) :
        load_text(prompt_fixture);
    float *video = calloc(VIDEO_ELEMENTS, sizeof(*video));
    float *audio = calloc(AUDIO_ELEMENTS, sizeof(*audio));
    float *video_velocity = malloc(VIDEO_ELEMENTS * sizeof(*video_velocity));
    float *audio_velocity = malloc(AUDIO_ELEMENTS * sizeof(*audio_velocity));
    if (!video || !audio || !video_velocity || !audio_velocity)
        die("out of memory allocating benchmark latents");
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++)
        video[index] = (float)((int)(index % 257) - 128) / 128.0f;
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++)
        audio[index] = (float)((int)(index % 127) - 63) / 64.0f;

    h3_text_embedding text = {
        .tokens = TEXT_ROWS,
        .width = TEXT_WIDTH,
        .values = text_values
    };
    int ref_layout = getenv("H3_BENCH_REF_LAYOUT") != NULL;
    int use_int8_row_fc2 = getenv("H3_BENCH_INT8_ROW_FC2") != NULL;
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 3, 16, 24, 20},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 24}
    };
    h3_layout_spec spec = {TEXT_ROWS, LATENT_T, LATENT_H, LATENT_W, AUDIO_T,
                           22, NULL, 0, ref_layout ? references : NULL,
                           ref_layout ? sizeof(references) / sizeof(*references)
                                      : 0};
    h3_layout layout;
    h3_sigma_schedule sigmas;
    char error[512];
    int sampler_ab = getenv("H3_BENCH_SAMPLER_AB") != NULL;
    int token_reduction_ab =
        getenv("H3_BENCH_TOKEN_REDUCTION_AB") != NULL;
    int cross_adaln_ab = getenv("H3_BENCH_CROSS_ADALN_AB") != NULL;
    int final_slice_ab = getenv("H3_BENCH_FINAL_SLICE_AB") != NULL;
    int final_head_ab = getenv("H3_BENCH_FINAL_HEAD_AB") != NULL;
    int int8_qkv_ab = getenv("H3_BENCH_INT8_QKV_AB") != NULL;
    int full_k_qkv_ab = getenv("H3_BENCH_QKV_FULL_K_AB") != NULL;
    int head_major_attention_output_ab =
        getenv("H3_BENCH_HEAD_MAJOR_ATTENTION_OUT_AB") != NULL;
    int vector_gate_adaln_ab =
        getenv("H3_BENCH_VECTOR_GATE_ADALN_AB") != NULL;
    int vector_qkv_rope_ab =
        getenv("H3_BENCH_VECTOR_QKV_ROPE_AB") != NULL;
    int known_int8_linear_ab =
        getenv("H3_BENCH_INT8_LINEAR_KNOWN_AB") != NULL;
    int int8_attention_out_ab =
        getenv("H3_BENCH_INT8_ATTENTION_OUT_AB") != NULL;
    int fused_int8_mlp_input_ab =
        getenv("H3_BENCH_FUSED_INT8_MLP_INPUT_AB") != NULL;
    int fused_int8_qkv_input_ab =
        getenv("H3_BENCH_FUSED_INT8_QKV_INPUT_AB") != NULL;
    int fused_int8_inputs_ab =
        getenv("H3_BENCH_FUSED_INT8_INPUTS_AB") != NULL;
    int fused_int8_qkv_rope_ab =
        getenv("H3_BENCH_FUSED_INT8_QKV_ROPE_AB") != NULL;
    int vector_int8_qkv_rms_ab =
        getenv("H3_BENCH_VECTOR_QKV_RMS_AB") != NULL;
    int local_int8_linear_scales_ab =
        getenv("H3_BENCH_INT8_LOCAL_SCALES_AB") != NULL;
    int local_int8_qkv_scales_ab =
        getenv("H3_BENCH_QKV_LOCAL_SCALES_AB") != NULL;
    const char *qkv_ab = getenv("H3_BENCH_QKV_AB");
    int qkv_denoise_ab = qkv_ab && !strcmp(qkv_ab, "denoise");
    int qkv_step_ab = qkv_ab && !strcmp(qkv_ab, "steps");
    if (!h3_layout_build(&spec, &layout, error, sizeof(error)) ||
        !((sampler_ab || token_reduction_ab || cross_adaln_ab ||
           final_slice_ab || final_head_ab || qkv_denoise_ab || qkv_step_ab)
              ? h3_serving_schedule_build(20, &sigmas)
              : h3_schedule_build(20, &sigmas)))
        die("cannot build benchmark layout");
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/%s/transformer", model_root,
             ref_layout ? "Ref2VA" : "FL2VA");
    unsigned active_blocks = 50;
    int reuse_interval = 1;
    const char *layers = getenv("H3_BENCH_LAYERS");
    if (layers && *layers) {
        char *end = NULL;
        long parsed = strtol(layers, &end, 10);
        if (end == layers || *end || parsed < 25 || parsed > 50)
            die("H3_BENCH_LAYERS must be in [25, 50]");
        active_blocks = (unsigned)parsed;
    }
    const char *reuse = getenv("H3_BENCH_REUSE");
    if (reuse && *reuse) {
        char *end = NULL;
        long parsed = strtol(reuse, &end, 10);
        if (end == reuse || *end || parsed < 1 || parsed > 3)
            die("H3_BENCH_REUSE must be in [1, 3]");
        reuse_interval = (int)parsed;
    }
    double load_start = seconds();
    if (final_slice_ab) {
        setenv("H3_DISABLE_FUSED_FINAL_HEAD", "1", 1);
        setenv("H3_DISABLE_FUSED_FINAL_SLICE", "1", 1);
    } else if (final_head_ab) {
        setenv("H3_DISABLE_FUSED_FINAL_HEAD", "1", 1);
    }
    int enable_token_reduction = token_reduction_ab || cross_adaln_ab ||
        final_slice_ab || final_head_ab;
    int use_slower_grouped_quantizer =
        getenv("H3_BENCH_USE_SLOWER_GROUPED_QUANTIZER") != NULL;
    int ssd_streaming = getenv("H3_BENCH_SSD_STREAMING") != NULL;
    int all_bf16 = getenv("H3_BENCH_ALL_BF16") != NULL;
    h3_dit *dit;
    if (ref_layout) {
        size_t video_condition_elements =
            layout.img_cond_rows * (size_t)96;
        size_t audio_condition_elements =
            layout.audio_cond_rows * (size_t)32;
        float *video_condition = calloc(
            video_condition_elements ? video_condition_elements : 1,
            sizeof(*video_condition));
        float *audio_condition = calloc(
            audio_condition_elements ? audio_condition_elements : 1,
            sizeof(*audio_condition));
        if (!video_condition || !audio_condition)
            die("out of memory allocating reference conditions");
        dit = h3_dit_load_conditioned(
            weights, "h3_shaders.metal", &text, &layout, &sigmas,
            active_blocks, 1, enable_token_reduction, ssd_streaming, 1.0f,
            all_bf16, all_bf16, all_bf16, 0, 0, 0, 0, 0, 0,
            use_slower_grouped_quantizer, use_int8_row_fc2,
            video_condition,
            video_condition_elements, audio_condition,
            audio_condition_elements, NULL, NULL, error, sizeof(error));
        free(video_condition);
        free(audio_condition);
    } else {
        dit = h3_dit_load_t2va(
            weights, "h3_shaders.metal", &text, &layout, &sigmas,
            active_blocks, 1, enable_token_reduction, ssd_streaming, 1.0f,
            all_bf16, all_bf16, all_bf16, 0, 0, 0, 0, 0, 0,
            use_slower_grouped_quantizer, use_int8_row_fc2,
            NULL, NULL, error,
            sizeof(error));
    }
    if (!dit) die(error);
    if (final_slice_ab) {
        unsetenv("H3_DISABLE_FUSED_FINAL_HEAD");
        unsetenv("H3_DISABLE_FUSED_FINAL_SLICE");
    } else if (final_head_ab) {
        unsetenv("H3_DISABLE_FUSED_FINAL_HEAD");
    }
    double load_seconds = seconds() - load_start;

    if (full_k_qkv_ab || head_major_attention_output_ab ||
        vector_gate_adaln_ab ||
        vector_qkv_rope_ab ||
        known_int8_linear_ab ||
        int8_qkv_ab || int8_attention_out_ab ||
        fused_int8_mlp_input_ab ||
        fused_int8_qkv_input_ab || fused_int8_inputs_ab ||
        fused_int8_qkv_rope_ab || vector_int8_qkv_rms_ab ||
        local_int8_linear_scales_ab || local_int8_qkv_scales_ab) {
        run_int8_projection_ab(
            dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before int8 projection AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }

    const char *command_ab = getenv("H3_BENCH_COMMAND_AB");
    if (command_ab && *command_ab) {
        char *end = NULL;
        long interval = strtol(command_ab, &end, 10);
        if (end == command_ab || *end || interval < 1 || interval > 50)
            die("H3_BENCH_COMMAND_AB must be in [1, 50]");
        run_command_ab(dit, (int)interval, video, audio, video_velocity,
                       audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before command AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_GRAPH_DATA_AB")) {
        run_graph_data_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before graph-data AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_MPS_COMMAND_AB")) {
        run_mps_command_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before MPS-command AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_NAX_MLP_AB") ||
        getenv("H3_BENCH_INT8_MLP_AB") ||
        getenv("H3_BENCH_INT8_QUANT_AB") ||
        getenv("H3_BENCH_INT8_GROUP_LOCAL_AB") ||
        getenv("H3_BENCH_INT8_GROUP_LOCAL128_AB") ||
        getenv("H3_BENCH_INT8_FC1_LOCAL_AB") ||
        getenv("H3_BENCH_INT8_FC1_FULL_K_AB") ||
        getenv("H3_BENCH_INT8_FC1_KNOWN_AB") ||
        getenv("H3_BENCH_INT8_GROUP_QUANT128_CACHE_AB") ||
        getenv("H3_BENCH_INT8_GROUP_QUANT128_AB")) {
        run_nax_mlp_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before MLP AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_NAX_MORTON_AB") ||
        getenv("H3_BENCH_NAX_MORTON4_AB")) {
        run_nax_morton_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before NAX Morton AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_NAX_LINEAR_AB") ||
        getenv("H3_BENCH_NAX_SPLIT_BOUNDARY_AB") ||
        getenv("H3_BENCH_HEAD_MAJOR_SDPA_AB") ||
        getenv("H3_BENCH_SPLIT_NAX_QKV_AB") ||
        getenv("H3_BENCH_NAX_LINEAR_MORTON_AB")) {
        run_nax_linear_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before NAX linear AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_QKV_CACHE_AB")) {
        run_qkv_cache_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before cached QKV AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (getenv("H3_BENCH_EXACT_SIMD_GATE_ADALN_AB")) {
        run_exact_simd_gate_adaln_ab(
            dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before exact SIMD gate AdaLN "
               "AB\n", (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (qkv_ab) {
        if (qkv_denoise_ab)
            run_qkv_denoise_ab(dit, video, audio, reuse_interval);
        else if (qkv_step_ab)
            run_qkv_step_ab(dit, video, audio, video_velocity,
                            audio_velocity, reuse_interval);
        else
            run_qkv_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before cooperative QKV AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (sampler_ab) {
        run_sampler_ab(dit, video, audio, video_velocity, audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before sampler AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (cross_adaln_ab) {
        run_cross_adaln_ab(dit, video, audio, video_velocity,
                           audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before cross-block AdaLN AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (final_slice_ab) {
        run_final_slice_ab(dit, video, audio, video_velocity,
                           audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before final slice AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (final_head_ab) {
        run_final_head_ab(dit, video, audio, video_velocity,
                          audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before final head AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }
    if (token_reduction_ab) {
        if (!strcmp(getenv("H3_BENCH_TOKEN_REDUCTION_AB"), "denoise"))
            run_token_reduction_denoise_ab(dit, video, audio,
                                           reuse_interval);
        else
            run_token_reduction_ab(dit, video, audio, video_velocity,
                                   audio_velocity);
        printf("DiT %ux%u/%u-layer load %.3fs before token reduction AB\n",
               (unsigned)CANVAS_W, (unsigned)CANVAS_H,
               active_blocks, load_seconds);
        h3_dit_free(dit);
        h3_layout_free(&layout);
        free(text_values); free(video); free(audio);
        free(video_velocity); free(audio_velocity);
        return 0;
    }

    static const int steps[] = {0, 3, 6, 9, 12, 15, 18};
    size_t forward_count = sizeof(steps) / sizeof(*steps);
    const char *forward_count_value = getenv("H3_BENCH_FORWARD_COUNT");
    if (forward_count_value && *forward_count_value) {
        char *end = NULL;
        unsigned long parsed = strtoul(forward_count_value, &end, 10);
        if (end == forward_count_value || *end || parsed < 1 ||
            parsed > forward_count)
            die("H3_BENCH_FORWARD_COUNT must be in [1, 7]");
        forward_count = (size_t)parsed;
    }
    double forward_start = seconds();
    for (size_t index = 0; index < forward_count; index++) {
        double step_start = seconds();
        if (!h3_dit_forward(dit, steps[index], video, audio,
                            video_velocity, audio_velocity,
                            error, sizeof(error))) die(error);
        for (size_t element = 0; element < VIDEO_ELEMENTS; element++)
            video[element] += video_velocity[element] * 0.001f;
        for (size_t element = 0; element < AUDIO_ELEMENTS; element++)
            audio[element] += audio_velocity[element] * 0.001f;
        printf("  core %zu step %d %.3fs\n", index + 1, steps[index],
               seconds() - step_start);
    }
    double forward_seconds = seconds() - forward_start;
    printf("DiT %ux%u load %.3fs, %zu forwards %.3fs, combined %.3fs; "
           "hashes video %016llx audio %016llx\n",
           (unsigned)CANVAS_W, (unsigned)CANVAS_H,
           load_seconds, forward_count, forward_seconds,
           load_seconds + forward_seconds,
           (unsigned long long)hash_bytes(
               video_velocity, VIDEO_ELEMENTS * sizeof(*video_velocity)),
           (unsigned long long)hash_bytes(
               audio_velocity, AUDIO_ELEMENTS * sizeof(*audio_velocity)));

    h3_dit_free(dit);
    h3_layout_free(&layout);
    free(text_values); free(video); free(audio);
    free(video_velocity); free(audio_velocity);
    return 0;
}
