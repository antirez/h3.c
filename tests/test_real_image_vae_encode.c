#include "h3_dit.h"
#include "h3_image_vae_encode.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CASES = 3, SIZES = 12 };

/* About 1.4 times the values measured on the fixture. The encoder is F32, so
 * the moments agree to rounding; the final latents also differ where a float16
 * rounding step flips. */
#define MEAN_BOUND 1.8e-6
#define LOGVAR_BOUND 5e-7
#define LATENT_BOUND 2.4e-5
#define NOISED_BOUND 2.4e-5

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_vae_encode.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total) fprintf(stderr, "image VAE encoder tiles: %d\n", total);
}

static void *read_tensor(const h3_st_header *fixture, const char *name,
                         h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_vae_encode.c: fixture "
                "tensor %s is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(error);
    return data;
}

static void *read_named(const h3_st_header *fixture, const char *name,
                        int index, h3_dtype dtype, size_t elements) {
    char full[64];
    snprintf(full, sizeof(full), "%s_%d", name, index);
    return read_tensor(fixture, full, dtype, elements);
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

static size_t differing_bytes(const uint8_t *got, const uint8_t *want,
                              size_t count) {
    size_t different = 0;
    for (size_t index = 0; index < count; index++)
        if (got[index] != want[index]) different++;
    return different;
}

static void check_bound(const char *what, double value, double bound) {
    printf("  %s: rel-L2 %.3g (bound %.3g)\n", what, value, bound);
    if (!(value < bound)) {
        fprintf(stderr, "FAIL tests/test_real_image_vae_encode.c: %s "
                "exceeds its bound\n", what);
        exit(1);
    }
}

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] :
        "MiniMax-H3-Image/vae_diffusers";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_image_edit_vae.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    int *bounds = read_tensor(&fixture, "bounds", H3_DTYPE_I32, 4);
    float *aug = read_tensor(&fixture, "reference_noise_aug", H3_DTYPE_F32, 1);
    if (aug[0] != H3_REFERENCE_NOISE_AUG ||
        bounds[0] != H3_REFERENCE_MIN_SHORT_EDGE ||
        bounds[1] != H3_REFERENCE_MAX_SHORT_EDGE ||
        bounds[2] != H3_VISION_MIN_SHORT_EDGE ||
        bounds[3] != H3_VISION_MAX_SHORT_EDGE)
        die("fixture was made with other reference settings");

    for (int index = 0; index < CASES; index++) {
        printf("reference %d\n", index);
        int *sizes = read_named(&fixture, "sizes", index, H3_DTYPE_I32, SIZES);
        int height = sizes[0], width = sizes[1];
        int resized_h = sizes[2], resized_w = sizes[3];
        int moments_h = sizes[6], moments_w = sizes[7];
        int latent_h = sizes[8], latent_w = sizes[9];
        uint8_t *reference = read_named(&fixture, "reference_u8", index,
            H3_DTYPE_U8, (size_t)3 * height * width);
        uint8_t *want_resized = read_named(&fixture, "resized_u8", index,
            H3_DTYPE_U8, (size_t)3 * resized_h * resized_w);

        int size_w, size_h, vision_w, vision_h;
        if (!h3_reference_bound_short_edge(width, height, bounds[0], bounds[1],
                H3_REFERENCE_CANVAS_MULTIPLE, &size_w, &size_h) ||
            !h3_reference_bound_short_edge(width, height, bounds[2], bounds[3],
                H3_REFERENCE_CANVAS_MULTIPLE, &vision_w, &vision_h))
            die("size rule failed");
        if (size_w != resized_w || size_h != resized_h ||
            vision_h != sizes[10] || vision_w != sizes[11])
            die("size rule differs from the reference");
        uint8_t *resized = malloc((size_t)3 * resized_h * resized_w);
        if (!resized) die("out of memory");
        if (size_w == width && size_h == height)
            memcpy(resized, reference, (size_t)3 * height * width);
        else if (!h3_reference_resize_lanczos(reference, height, width,
                                              size_h, size_w, resized))
            die("resize failed");
        size_t different = differing_bytes(resized, want_resized,
                                           (size_t)3 * resized_h * resized_w);
        printf("  %dx%d -> %dx%d: %zu of %zu resized bytes differ from PIL\n",
               width, height, resized_w, resized_h, different,
               (size_t)3 * resized_h * resized_w);
        if (different) die("Lanczos resize is not byte exact");

        size_t moments_plane = (size_t)moments_h * moments_w;
        size_t latent_plane = (size_t)latent_h * latent_w;
        float *noise = read_named(&fixture, "noise", index, H3_DTYPE_F32,
                                  24 * moments_plane);
        float *want_mean = read_named(&fixture, "mean", index, H3_DTYPE_F32,
                                      24 * moments_plane);
        float *want_logvar = read_named(&fixture, "logvar", index,
            H3_DTYPE_F32, 24 * moments_plane);
        float *want_latents = read_named(&fixture, "latents", index,
            H3_DTYPE_F32, 24 * latent_plane);
        float *reference_noise = read_named(&fixture, "reference_noise", index,
            H3_DTYPE_F32, 24 * latent_plane);
        float *want_noised = read_named(&fixture, "noised", index,
            H3_DTYPE_F32, 24 * latent_plane);
        float *want_rows = read_named(&fixture, "noised_rows", index,
            H3_DTYPE_F32, latent_plane / 4 * 96);

        h3_reference_latents got;
        if (!h3_image_vae_encode(weights, "h3_shaders.metal", resized,
                                 resized_h, resized_w, noise, 0, progress,
                                 NULL, &got, error, sizeof(error)))
            die(error);
        if (got.moments_height != moments_h || got.moments_width != moments_w ||
            got.latent_height != latent_h || got.latent_width != latent_w)
            die("encoder returned the wrong shape");
        double maximum;
        check_bound("posterior mean", relative_l2(got.moments, want_mean,
            24 * moments_plane, &maximum), MEAN_BOUND);
        check_bound("posterior logvar", relative_l2(got.moments +
            24 * moments_plane, want_logvar, 24 * moments_plane, &maximum),
            LOGVAR_BOUND);
        check_bound("latents", relative_l2(got.latents, want_latents,
            24 * latent_plane, &maximum), LATENT_BOUND);
        printf("  latents: max abs %.3g\n", maximum);

        float *noised = malloc(24 * latent_plane * sizeof(*noised));
        float *rows = malloc(24 * latent_plane * sizeof(*rows));
        if (!noised || !rows) die("out of memory");
        h3_reference_scale_noise(got.latents, reference_noise,
                                 H3_REFERENCE_NOISE_AUG, 24 * latent_plane,
                                 noised);
        check_bound("noised latents", relative_l2(noised, want_noised,
            24 * latent_plane, &maximum), NOISED_BOUND);
        double exact_l2 = 0.0;
        float *exact = malloc(24 * latent_plane * sizeof(*exact));
        if (!exact) die("out of memory");
        h3_reference_scale_noise(want_latents, reference_noise,
                                 H3_REFERENCE_NOISE_AUG, 24 * latent_plane,
                                 exact);
        exact_l2 = relative_l2(exact, want_noised, 24 * latent_plane, &maximum);
        printf("  noised from the reference latents: rel-L2 %.3g, max abs %.3g\n",
               exact_l2, maximum);
        if (maximum != 0.0) die("scale_noise is not bit exact");
        if (!h3_dit_patchify_video(noised, 24, 1, latent_h, latent_w, rows,
                                   24 * latent_plane))
            die("patchify failed");
        check_bound("noised rows", relative_l2(rows, want_rows,
            latent_plane / 4 * 96, &maximum), NOISED_BOUND);
        printf("  %.3f GiB allocated, %.3f GPU seconds\n",
               (double)got.gpu_stats.allocated_bytes /
                   (1024.0 * 1024.0 * 1024.0), got.gpu_stats.gpu_seconds);

        free(exact); free(rows); free(noised);
        h3_reference_latents_free(&got);
        free(want_rows); free(want_noised); free(reference_noise);
        free(want_latents); free(want_logvar); free(want_mean); free(noise);
        free(resized); free(want_resized); free(reference); free(sizes);
    }

    uint8_t *down_in = read_tensor(&fixture, "resize_down_in_u8", H3_DTYPE_U8,
                                   (size_t)3 * 203 * 317);
    uint8_t *down_want = read_tensor(&fixture, "resize_down_out_u8",
                                     H3_DTYPE_U8, (size_t)3 * 61 * 100);
    uint8_t down_got[3 * 61 * 100];
    if (!h3_reference_resize_lanczos(down_in, 203, 317, 61, 100, down_got))
        die("downscale failed");
    size_t different = differing_bytes(down_got, down_want, sizeof(down_got));
    printf("downscale 317x203 -> 100x61: %zu of %zu bytes differ from PIL\n",
           different, sizeof(down_got));
    if (different) die("Lanczos downscale is not byte exact");
    free(down_want); free(down_in); free(aug); free(bounds);
    h3_st_free_header(&fixture);
    puts("ok: native Metal image VAE encoder and reference steps match diffusers");
    return 0;
}
