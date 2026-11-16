/* Replay a video VAE decode from a latent saved with H3_DUMP_VIDEO_LATENT.
 *
 *   vae_bench <video_vae/source dir> <latent.bin> [out.rgbf32] [ref.rgbf32]
 *
 * Prints load/decode wall time; with a reference it prints max abs error and
 * PSNR over the [0,1] RGB output, so precision variants (H3_VAE_PRECISION)
 * can be compared against the released F32 path on identical input. */
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *read_all(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long end = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *data = end > 0 ? malloc((size_t)end) : NULL;
    if (data && fread(data, 1, (size_t)end, file) != (size_t)end) {
        free(data);
        data = NULL;
    }
    fclose(file);
    if (data) *size = (size_t)end;
    return data;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s VAE_DIR LATENT [OUT] [REF]\n", argv[0]);
        return 2;
    }
    size_t size = 0;
    int32_t *blob = read_all(argv[2], &size);
    if (!blob || size < 12) {
        fprintf(stderr, "cannot read latent %s\n", argv[2]);
        return 1;
    }
    int t = blob[0], h = blob[1], w = blob[2];
    size_t count = (size_t)24 * (size_t)t * (size_t)h * (size_t)w;
    if (size != 12 + count * sizeof(float)) {
        fprintf(stderr, "latent size mismatch\n");
        return 1;
    }
    const float *latent = (const float *)(blob + 3);
    char error[512];
    double start = now();
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        argv[1], "h3_shaders.metal", h, w, NULL, NULL, error, sizeof(error));
    if (!decoder) {
        fprintf(stderr, "load: %s\n", error);
        return 1;
    }
    double loaded = now();
    h3_video_frames frames = {0};
    if (!h3_video_vae_decoder_decode(decoder, latent, t, &frames, error,
                                     sizeof(error))) {
        fprintf(stderr, "decode: %s\n", error);
        return 1;
    }
    double decoded = now();
    size_t values = (size_t)frames.frames * (size_t)frames.height *
                    (size_t)frames.width * 3;
    printf("latent %dx%dx%d -> %d frames %dx%d  load %.2fs  decode %.2fs\n",
           t, h, w, frames.frames, frames.width, frames.height,
           loaded - start, decoded - loaded);
    if (argc > 3 && argv[3][0] && strcmp(argv[3], "-") != 0) {
        FILE *out = fopen(argv[3], "wb");
        if (!out || fwrite(frames.rgb, sizeof(float), values, out) != values) {
            fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
        fclose(out);
    }
    if (argc > 4) {
        size_t ref_size = 0;
        float *ref = read_all(argv[4], &ref_size);
        if (!ref || ref_size != values * sizeof(float)) {
            fprintf(stderr, "reference size mismatch\n");
            return 1;
        }
        double max_error = 0.0, square = 0.0;
        for (size_t index = 0; index < values; index++) {
            double delta = (double)frames.rgb[index] - ref[index];
            if (fabs(delta) > max_error) max_error = fabs(delta);
            square += delta * delta;
        }
        double mse = square / (double)values;
        printf("vs ref: max_abs %.5f  rmse %.6f  psnr %.2f dB\n", max_error,
               sqrt(mse), mse > 0 ? 10.0 * log10(1.0 / mse) : INFINITY);
        free(ref);
    }
    free(frames.rgb);
    h3_video_vae_decoder_free(decoder);
    free(blob);
    return 0;
}
