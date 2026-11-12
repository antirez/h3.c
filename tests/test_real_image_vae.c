#include "h3_image_vae.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_vae.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "image VAE tile %d/%d\n", completed, total);
}

static void *read_tensor(const h3_st_header *fixture, const char *name,
                         h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_vae.c: fixture tensor %s "
                "is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(error);
    return data;
}

static double relative_l2(const float *got, const float *want, size_t count,
                          double *maximum) {
    double square_error = 0.0, square_value = 0.0;
    *maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        if (fabs(delta) > *maximum) *maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    return sqrt(square_error / square_value);
}

static void check_bytes(const h3_image_planes *image, const uint8_t *want,
                        const char *label) {
    size_t count = 3 * (size_t)image->height * (size_t)image->width;
    uint8_t *got = malloc(count);
    if (!got) die("out of memory");
    h3_image_planes_to_u8(image, got);
    size_t exact = 0, within_one = 0;
    int maximum = 0;
    for (size_t index = 0; index < count; index++) {
        int delta = abs((int)got[index] - (int)want[index]);
        if (delta == 0) exact++;
        if (delta <= 1) within_one++;
        if (delta > maximum) maximum = delta;
    }
    free(got);
    double within_fraction = (double)within_one / (double)count;
    printf("%s: %.4f%% exact, %.4f%% within +-1, max abs diff %d\n", label,
           100.0 * (double)exact / (double)count, 100.0 * within_fraction,
           maximum);
    if (within_fraction < 0.999) die("fewer than 99.9% of bytes within +-1");
    if (maximum > 2) die("a byte differs by more than 2");
}

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] :
        "MiniMax-H3-Image/vae_diffusers";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_image_vae_decode.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    enum {
        TILE = 256, TILE_LATENT = 16, FULL = 512, FULL_LATENT = 32,
        NS_LATENT_H = 10, NS_LATENT_W = 37
    };
    float *latents = read_tensor(&fixture, "latents", H3_DTYPE_F32,
                                 24 * FULL_LATENT * FULL_LATENT);
    uint8_t *want_bytes = read_tensor(&fixture, "image_u8", H3_DTYPE_U8,
                                      3 * FULL * FULL);
    float *latents_1 = read_tensor(&fixture, "latents_1", H3_DTYPE_F32,
                                   24 * TILE_LATENT * TILE_LATENT);
    float *want_rgb_1 = read_tensor(&fixture, "image_1", H3_DTYPE_F32,
                                    3 * TILE * TILE);
    float *want_raw_1 = read_tensor(&fixture, "image_1_raw", H3_DTYPE_F32,
                                    3 * TILE * TILE);
    float *latents_ns = read_tensor(&fixture, "latents_nonsquare",
        H3_DTYPE_F32, 24 * NS_LATENT_H * NS_LATENT_W);
    uint8_t *want_ns = read_tensor(&fixture, "image_nonsquare_u8",
        H3_DTYPE_U8, 3 * NS_LATENT_H * 16 * NS_LATENT_W * 16);

    h3_image_planes single;
    if (!h3_image_vae_decode(weights, "h3_shaders.metal", latents_1,
                             TILE_LATENT, TILE_LATENT, progress, NULL,
                             &single, error, sizeof(error))) die(error);
    if (single.height != TILE || single.width != TILE)
        die("single tile has the wrong shape");
    double raw_max, rgb_max;
    double raw_l2 = relative_l2(single.data, want_raw_1, 3 * TILE * TILE,
                                &raw_max);
    h3_image_planes_to_rgb(&single);
    double rgb_l2 = relative_l2(single.data, want_rgb_1, 3 * TILE * TILE,
                                &rgb_max);
    printf("single tile raw output: rel-L2 %.6g, max abs %.6g\n", raw_l2,
           raw_max);
    printf("single tile RGB: rel-L2 %.6g, max abs %.6g\n", rgb_l2, rgb_max);
    printf("single tile: %.3f GiB allocated, %.3f GPU seconds\n",
           (double)single.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0), single.gpu_stats.gpu_seconds);
    if (raw_l2 >= 1e-4) die("single tile exceeds the F32 parity bound");
    if (rgb_l2 >= 1e-4) die("single tile RGB exceeds the F32 parity bound");
    h3_image_planes_free(&single);

    h3_image_planes full;
    if (!h3_image_vae_decode(weights, "h3_shaders.metal", latents,
                             FULL_LATENT, FULL_LATENT, progress, NULL, &full,
                             error, sizeof(error))) die(error);
    if (full.height != FULL || full.width != FULL)
        die("tiled image has the wrong shape");
    h3_image_planes_to_rgb(&full);
    check_bytes(&full, want_bytes, "tiled 512x512");
    printf("tiled: %.3f GPU seconds, %llu submissions\n",
           full.gpu_stats.gpu_seconds,
           (unsigned long long)full.gpu_stats.submissions);
    h3_image_planes_free(&full);

    h3_image_planes nonsquare;
    if (!h3_image_vae_decode(weights, "h3_shaders.metal", latents_ns,
                             NS_LATENT_H, NS_LATENT_W, progress, NULL,
                             &nonsquare, error, sizeof(error))) die(error);
    if (nonsquare.height != NS_LATENT_H * 16 ||
        nonsquare.width != NS_LATENT_W * 16)
        die("non-square image has the wrong shape");
    h3_image_planes_to_rgb(&nonsquare);
    check_bytes(&nonsquare, want_ns, "non-square 160x592");
    h3_image_planes_free(&nonsquare);
    free(latents_ns); free(want_ns);
    free(latents); free(want_bytes); free(latents_1);
    free(want_rgb_1); free(want_raw_1);
    h3_st_free_header(&fixture);
    puts("ok: native Metal image VAE matches the diffusers decoder");
    return 0;
}
