/* A/B comparison of the video VAE decoder in its two supported precisions.
 *
 * The decoder normally runs in BF16 so that the MPS matmul and SDPA paths
 * can use the Apple tensor cores. H3_VAE_PRECISION=f32 keeps the original
 * float32 path. This test decodes the same fixed pseudo-random latent with
 * both paths and reports the timing and the numerical difference.
 *
 * Usage:
 *   h3_video_vae_tests [model_root] [shader] [latent_h] [latent_w]
 *                       [latent_time]
 *
 * H3_VAE_TEST_TILES may hold a comma separated list of spatial tile sizes
 * (for example "256,304"); every entry is measured, compared against the
 * float32 path, and compared against the first entry so the effect of the
 * tile size on the decoded frames is visible.
 */
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    /* The released decoder consumes 24 latent channels and a 16x pixel ratio.
     * The default geometry keeps the test quick; the full 960x544x124 frame
     * target can be requested from the command line. */
    LATENT_CHANNELS = 24,
    DEFAULT_LATENT_TIME = 7,
    DEFAULT_LATENT_HEIGHT = 8,
    DEFAULT_LATENT_WIDTH = 12,
    SPATIAL_RATIO = 16
};

typedef struct {
    double load_seconds;
    double decode_seconds;
    double gpu_seconds;
    unsigned long long mps_linears;
    unsigned long long mps_sdpa;
    unsigned long long submissions;
    unsigned long long peak_live_bytes;
    int frames;
    int height;
    int width;
    float *rgb;
} decode_result;

typedef struct {
    double rel_l2;
    double max_abs;
    double mean_abs;
    double reference_min;
    double reference_max;
    double other_min;
    double other_max;
} comparison;

typedef struct {
    int tile_pixels;
    decode_result f32;
    decode_result bf16;
} tile_result;

static int latent_time = DEFAULT_LATENT_TIME;
static int latent_height = DEFAULT_LATENT_HEIGHT;
static int latent_width = DEFAULT_LATENT_WIDTH;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_video_vae.c: %s\n", message);
    exit(1);
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void progress(int completed, int total, void *opaque) {
    (void)completed;
    (void)total;
    (void)opaque;
}

/* A fixed linear congruential generator keeps the comparison reproducible
 * across runs, machines, and precisions. */
static void fill_latent(float *values, size_t count) {
    uint64_t state = 0x9e3779b97f4a7c15ull;
    for (size_t index = 0; index < count; index++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        uint32_t bits = (uint32_t)(state >> 32);
        float unit = (float)(bits & 0xffffffu) / (float)0xffffffu;
        values[index] = (3.0f * unit - 1.5f) * 2.0f;
    }
}

/* Loads the whole checkpoint in the given precision and reports how long the
 * weight upload takes. The video VAE checkpoint stores 560 F32 tensors, so
 * the BF16 path pays an extra host-side F32 -> BF16 conversion on top of the
 * upload. */
static int load_benchmark(const char *weights, const char *shader,
                          const char *precision, int repeats) {
    double *samples = malloc((size_t)repeats * sizeof(*samples));
    if (!samples) die("out of memory allocating load samples");
    double minimum = INFINITY, total = 0.0;
    for (int index = 0; index < repeats; index++) {
        char error[512];
        setenv("H3_VAE_PRECISION", precision, 1);
        double start = now_seconds();
        h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
                            weights, shader, latent_height, latent_width,
                            progress, NULL, error, sizeof(error));
        double elapsed = now_seconds() - start;
        if (!decoder) {
            fprintf(stderr, "h3: %s video VAE load failed: %s\n", precision,
                    error);
            free(samples);
            return 0;
        }
        h3_video_vae_decoder_free(decoder);
        samples[index] = elapsed;
        if (elapsed < minimum) minimum = elapsed;
        total += elapsed;
    }
    for (int index = 0; index < repeats; index++)
        if (samples[index] < minimum) minimum = samples[index];
    printf("  %-5s: %d loads, min %7.3f s, mean %7.3f s\n", precision,
           repeats, minimum, total / (double)repeats);
    free(samples);
    return 1;
}

static int decode_with(const char *weights, const char *shader,
                       const char *precision, int tile_pixels,
                       const float *latent, decode_result *result) {
    char error[512];
    char tile[32];
    setenv("H3_VAE_PRECISION", precision, 1);
    if (tile_pixels > 0) {
        snprintf(tile, sizeof(tile), "%d", tile_pixels);
        setenv("H3_VAE_TILE_PIXELS", tile, 1);
    } else {
        unsetenv("H3_VAE_TILE_PIXELS");
    }
    memset(result, 0, sizeof(*result));
    double start = now_seconds();
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
                        weights, shader, latent_height, latent_width,
                        progress, NULL, error, sizeof(error));
    if (!decoder) {
        fprintf(stderr, "h3: %s video VAE load failed: %s\n", precision, error);
        return 0;
    }
    result->load_seconds = now_seconds() - start;
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    start = now_seconds();
    int ok = h3_video_vae_decoder_decode(decoder, latent, latent_time,
                                          &frames, error, sizeof(error));
    result->decode_seconds = now_seconds() - start;
    if (!ok) {
        fprintf(stderr, "h3: %s video VAE decode failed: %s\n", precision,
                error);
        h3_video_vae_decoder_free(decoder);
        return 0;
    }
    result->gpu_seconds = frames.gpu_stats.gpu_seconds;
    result->mps_linears = frames.gpu_stats.mps_linear_dispatches;
    result->mps_sdpa = frames.gpu_stats.mps_sdpa_dispatches;
    result->submissions = frames.gpu_stats.submissions;
    result->peak_live_bytes = frames.gpu_stats.peak_live_bytes;
    result->frames = frames.frames;
    result->height = frames.height;
    result->width = frames.width;
    result->rgb = frames.rgb;
    h3_video_vae_decoder_free(decoder);
    return 1;
}

static int compare(const decode_result *reference, const decode_result *other,
                   comparison *out) {
    size_t count = (size_t)reference->frames * (size_t)reference->height *
                   (size_t)reference->width * 3;
    if (count != (size_t)other->frames * (size_t)other->height *
                  (size_t)other->width * 3) return 0;
    double square_error = 0.0, square_value = 0.0, absolute = 0.0;
    double maximum = 0.0;
    double reference_min = INFINITY, reference_max = -INFINITY;
    double other_min = INFINITY, other_max = -INFINITY;
    for (size_t index = 0; index < count; index++) {
        double want = (double)reference->rgb[index];
        double got = (double)other->rgb[index];
        double delta = got - want;
        double magnitude = fabs(delta);
        if (magnitude > maximum) maximum = magnitude;
        absolute += magnitude;
        square_error += delta * delta;
        square_value += want * want;
        if (want < reference_min) reference_min = want;
        if (want > reference_max) reference_max = want;
        if (got < other_min) other_min = got;
        if (got > other_max) other_max = got;
    }
    out->rel_l2 = square_value > 1e-24 ?
        sqrt(square_error / square_value) : sqrt(square_error);
    out->max_abs = maximum;
    out->mean_abs = count ? absolute / (double)count : 0.0;
    out->reference_min = reference_min;
    out->reference_max = reference_max;
    out->other_min = other_min;
    out->other_max = other_max;
    return 1;
}

static void print_delta(const char *label, const comparison *stats) {
    printf("  %-28s rel-L2 %.6g, max |delta| %.6g, mean |delta| %.6g\n",
           label, stats->rel_l2, stats->max_abs, stats->mean_abs);
}

static int decode_tile(const char *weights, const char *shader, int tile_pixels,
                       const float *latent, tile_result *out) {
    out->tile_pixels = tile_pixels;
    if (!decode_with(weights, shader, "f32", tile_pixels, latent, &out->f32))
        die("the float32 video VAE decode failed");
    if (!decode_with(weights, shader, "bf16", tile_pixels, latent, &out->bf16))
        die("the BF16 video VAE decode failed");
    if (out->f32.frames != out->bf16.frames ||
        out->f32.height != out->bf16.height ||
        out->f32.width != out->bf16.width)
        die("the float32 and BF16 decodes disagree on the output shape");
    if (out->f32.frames < 1 || out->f32.height < 1 || out->f32.width < 1)
        die("the video VAE produced an empty frame");
    if (out->f32.height != latent_height * SPATIAL_RATIO ||
        out->f32.width != latent_width * SPATIAL_RATIO)
        die("the video VAE produced the wrong pixel resolution");
    int expected_frames = ((latent_time - 2) / 5) * 17 + 5;
    if (out->f32.frames != expected_frames) {
        fprintf(stderr, "h3: expected %d decoded frames, got %d\n",
                expected_frames, out->f32.frames);
        exit(1);
    }
    return 1;
}

static void print_tile(const tile_result *tile) {
    printf("video VAE decode: %d frames at %dx%d, tile %d px\n",
           tile->f32.frames, tile->f32.height, tile->f32.width,
           tile->tile_pixels);
    printf("  F32 : load %7.3f s, decode %7.3f s, GPU %7.3f s, "
           "%llu MPS linears, %llu MPS SDPA, peak %.2f GiB\n",
           tile->f32.load_seconds, tile->f32.decode_seconds,
           tile->f32.gpu_seconds, tile->f32.mps_linears,
           tile->f32.mps_sdpa,
           (double)tile->f32.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  BF16: load %7.3f s, decode %7.3f s, GPU %7.3f s, "
           "%llu MPS linears, %llu MPS SDPA, peak %.2f GiB\n",
           tile->bf16.load_seconds, tile->bf16.decode_seconds,
           tile->bf16.gpu_seconds, tile->bf16.mps_linears,
           tile->bf16.mps_sdpa,
           (double)tile->bf16.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));
    if (tile->bf16.decode_seconds > 0.0)
        printf("  BF16 decode speedup: %.2fx\n",
               tile->f32.decode_seconds / tile->bf16.decode_seconds);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *shader = argc > 2 ? argv[2] : "h3_shaders.metal";
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source",
             model_root);
    if (argc > 3) latent_height = atoi(argv[3]);
    if (argc > 4) latent_width = atoi(argv[4]);
    if (argc > 5) latent_time = atoi(argv[5]);
    if (latent_height < 1 || latent_width < 1 || latent_time < 7)
        die("invalid latent geometry");
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent_time *
                          (size_t)latent_height * (size_t)latent_width;
    float *latent = malloc(latent_count * sizeof(*latent));
    if (!latent) die("out of memory allocating the test latent");
    fill_latent(latent, latent_count);

    int tiles[8];
    int tile_count = 0;
    const char *list = getenv("H3_VAE_TEST_TILES");
    if (list && *list) {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s", list);
        char *cursor = buffer;
        while (cursor && *cursor && tile_count < 8) {
            char *comma = strchr(cursor, ',');
            if (comma) *comma = '\0';
            int pixels = atoi(cursor);
            if (pixels > 0) tiles[tile_count++] = pixels;
            cursor = comma ? comma + 1 : NULL;
        }
    }
    if (!tile_count) tiles[tile_count++] = 0;

    const char *loads = getenv("H3_VAE_TEST_LOADS");
    if (loads && *loads) {
        int repeats = atoi(loads);
        if (repeats < 1) repeats = 1;
        printf("video VAE weight load (36 blocks, F32 checkpoint):\n");
        if (!load_benchmark(weights, shader, "f32", repeats))
            die("the float32 video VAE load benchmark failed");
        if (!load_benchmark(weights, shader, "bf16", repeats))
            die("the BF16 video VAE load benchmark failed");
    }

    int ok = 1;
    tile_result baseline;
    memset(&baseline, 0, sizeof(baseline));
    for (int index = 0; index < tile_count; index++) {
        tile_result tile;
        memset(&tile, 0, sizeof(tile));
        decode_tile(weights, shader, tiles[index], latent, &tile);
        printf("--- spatial tile %d px ---\n", tiles[index]);
        print_tile(&tile);
        comparison precision;
        if (!compare(&tile.f32, &tile.bf16, &precision))
            die("the float32 and BF16 decodes produced different pixel counts");
        print_delta("BF16 vs F32:", &precision);
        printf("  RGB range: F32 [%.6g, %.6g], BF16 [%.6g, %.6g]\n",
               precision.reference_min, precision.reference_max,
               precision.other_min, precision.other_max);
        if (precision.rel_l2 > 0.01) {
            fprintf(stderr, "h3: the BF16 decoder exceeds the rel-L2 0.01 "
                            "bound (%.6g)\n", precision.rel_l2);
            ok = 0;
        }
        if (precision.other_min < -1e-6 || precision.other_max > 1.0 + 1e-6) {
            fprintf(stderr, "h3: the BF16 decoder left the [0,1] RGB range\n");
            ok = 0;
        }
        if (index == 0) {
            baseline = tile;
            continue;
        }
        comparison tiles_f32;
        if (compare(&baseline.f32, &tile.f32, &tiles_f32)) {
            char label[64];
            snprintf(label, sizeof(label), "tile %d vs %d (F32):",
                     tiles[index], baseline.tile_pixels);
            print_delta(label, &tiles_f32);
        }
        comparison tiles_bf16;
        if (compare(&baseline.bf16, &tile.bf16, &tiles_bf16)) {
            char label[64];
            snprintf(label, sizeof(label), "tile %d vs %d (BF16):",
                     tiles[index], baseline.tile_pixels);
            print_delta(label, &tiles_bf16);
        }
        free(tile.f32.rgb);
        free(tile.bf16.rgb);
    }
    if (baseline.f32.rgb) {
        free(baseline.f32.rgb);
        free(baseline.bf16.rgb);
    }
    free(latent);
    if (!ok) exit(1);
    puts("ok: the BF16 video VAE matches the float32 decoder");
    return 0;
}
