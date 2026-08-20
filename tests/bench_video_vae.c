#include "h3_video_vae.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Quality preset latent shape: 1024x576 canvas, 107 frames.
 * latent_time 32 -> 6 temporal chunks -> 107 decoded frames. */
#ifndef H3_BENCH_VAE_LATENT_T
#define H3_BENCH_VAE_LATENT_T 32
#endif
#ifndef H3_BENCH_VAE_LATENT_H
#define H3_BENCH_VAE_LATENT_H 36
#endif
#ifndef H3_BENCH_VAE_LATENT_W
#define H3_BENCH_VAE_LATENT_W 64
#endif

enum {
    LATENT_CHANNELS = 24,
    LATENT_TIME = H3_BENCH_VAE_LATENT_T,
    LATENT_H = H3_BENCH_VAE_LATENT_H,
    LATENT_W = H3_BENCH_VAE_LATENT_W,
    EXPECTED_FRAMES = (LATENT_TIME - 2) / 5 * 17 + 5,
    MAX_RUNS = 16
};

static void die(const char *message) {
    fprintf(stderr, "h3_vae_bench: %s\n", message);
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv) {
    const char *model_dir = argc > 1 ? argv[1] : "./MiniMax-H3";
    int runs = argc > 2 ? atoi(argv[2]) : 3;
    if (runs < 1) runs = 1;
    if (runs > MAX_RUNS) runs = MAX_RUNS;
    char weights[4096];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_dir);
    size_t latent_elements =
        (size_t)LATENT_TIME * LATENT_H * LATENT_W * LATENT_CHANNELS;
    float *latent = malloc(latent_elements * sizeof(*latent));
    if (!latent) die("out of memory for latents");
    /* Deterministic pseudo-random latents: decode timing is value
     * independent, so synthetic latents profile the real compute path. */
    uint64_t state = UINT64_C(0x9E3779B97F4A7C15);
    for (size_t index = 0; index < latent_elements; index++) {
        state = state * UINT64_C(6364136223846793005) +
                UINT64_C(1442695040888963407);
        latent[index] = (float)((double)(state >> 40) / 8388608.0 - 1.0);
    }
    double wall[MAX_RUNS], gpu[MAX_RUNS];
    char error[512];
    for (int run = 0; run < runs; run++) {
        h3_video_frames frames;
        double start = seconds();
        int ok = h3_video_vae_decode(weights, "h3_shaders.metal", latent,
            LATENT_TIME, LATENT_H, LATENT_W, NULL, NULL, &frames, error,
            sizeof(error));
        double elapsed = seconds() - start;
        if (!ok) die(error);
        if (frames.frames != EXPECTED_FRAMES || frames.height != LATENT_H * 16 ||
            frames.width != LATENT_W * 16) {
            h3_video_frames_free(&frames);
            die("decoded frame shape does not match the quality preset");
        }
        wall[run] = elapsed;
        gpu[run] = frames.gpu_stats.gpu_seconds;
        printf("run %d: wall %.3fs, gpu %.3fs, submissions %llu, peak %.3f GB\n",
               run + 1, elapsed, frames.gpu_stats.gpu_seconds,
               (unsigned long long)frames.gpu_stats.submissions,
               (double)frames.gpu_stats.peak_live_bytes / 1e9);
        fflush(stdout);
        h3_video_frames_free(&frames);
    }
    free(latent);
    qsort(wall, (size_t)runs, sizeof(*wall), compare_double);
    qsort(gpu, (size_t)runs, sizeof(*gpu), compare_double);
    printf("median: wall %.3fs, gpu %.3fs over %d run(s)\n",
           wall[runs / 2], gpu[runs / 2], runs);
    return 0;
}
