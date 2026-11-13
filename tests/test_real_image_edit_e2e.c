#include "h3.h"
#include "h3_ffmpeg.h"
#include "h3_image.h"
#include "h3_image_vae_encode.h"
#include "h3_internal.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { LATENT_CHANNELS = 24, REFERENCE_CASES = 3, SIZES = 12 };

/* Reference rows against the M5a fixture (the VAE only, no text encoder or
 * transformer): the encoder is F32, so the rows agree to rounding. */
#define REFERENCE_ROWS_BOUND 2.4e-5

/* 1.4x the measured distance to the F32 truth (PSNR lowered by 20 log10 1.4,
 * the byte miss fraction and the latent rel-L2 scaled by 1.4). At S = 2324
 * the six free-running steps amplify BF16 rounding: the diffusers BF16
 * pipeline itself lands at latent rel-L2 0.143 and 24.3 dB from the same
 * truth. The parity run feeds the fixture's resized copies and noises. */
typedef struct {
    double psnr, within16, latent;
} bounds;
static const bounds BF16_BOUNDS = {19.8, 0.65, 0.264};
static const bounds INT8_BOUNDS = {19.6, 0.63, 0.263};
/* The same run with h3.c's own resampling of the reference. */
static const bounds OWN_BF16_BOUNDS = {19.8, 0.65, 0.264};
static const bounds OWN_INT8_BOUNDS = {19.6, 0.63, 0.263};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_edit_e2e.c: %s\n", message);
    exit(1);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_edit_e2e.c: fixture "
                "tensor %s is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static void *read_named(const h3_st_header *fixture, const char *name,
                        int index, h3_dtype dtype, size_t elements) {
    char full[64];
    snprintf(full, sizeof(full), "%s_%d", name, index);
    return read_fixture(fixture, full, dtype, elements);
}

static int32_t read_scalar(const h3_st_header *fixture, const char *name) {
    int32_t *data = read_fixture(fixture, name, H3_DTYPE_I32, 1);
    int32_t value = data[0];
    free(data);
    return value;
}

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int progress(const char *phase, int completed, int total,
                    void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "  %s %d/%d\n", phase, completed, total);
    return 0;
}

static void write_reference(const char *path, const uint8_t *planar,
                            int width, int height) {
    char error[512];
    size_t area = (size_t)width * (size_t)height;
    uint8_t *interleaved = malloc(area * 3);
    if (!interleaved) die("oom");
    for (size_t pixel = 0; pixel < area; pixel++)
        for (size_t channel = 0; channel < 3; channel++)
            interleaved[3 * pixel + channel] = planar[channel * area + pixel];
    if (!h3_ffmpeg_write_png_rgb24(path, interleaved, width, height, error,
                                   sizeof(error))) die(error);
    free(interleaved);
}

static void require_shape(const h3_st_header *fixture, const char *name,
                          int height, int width, const char *what) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->ndim != 3 || tensor->shape[0] != 3 ||
        tensor->shape[1] != (uint64_t)height ||
        tensor->shape[2] != (uint64_t)width) {
        fprintf(stderr, "FAIL tests/test_real_image_edit_e2e.c: fixture "
                "tensor %s is not 3x%dx%d (%s)\n", name, height, width, what);
        exit(1);
    }
}

static double relative_l2(const float *got, const float *want, size_t count) {
    double square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    return sqrt(square_error / square_value);
}

/* The M5a fixture's three references through h3_generate_image_with_hooks,
 * stopped after the reference VAE: the condition rows of the whole reference
 * phase (own decode and PIL-exact resize, the fixture's noises) against the
 * diffusers rows. */
static void check_reference_rows(h3_ctx *ctx, const char *fixture_path,
                                 char *directory) {
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    h3_image_reference references[REFERENCE_CASES];
    char paths[REFERENCE_CASES][600];
    const float *posterior[REFERENCE_CASES], *aug[REFERENCE_CASES];
    float *want_rows[REFERENCE_CASES];
    size_t row_elements[REFERENCE_CASES];
    for (int index = 0; index < REFERENCE_CASES; index++) {
        int *sizes = read_named(&fixture, "sizes", index, H3_DTYPE_I32, SIZES);
        int height = sizes[0], width = sizes[1];
        int moments_h = sizes[6], moments_w = sizes[7];
        int latent_h = sizes[8], latent_w = sizes[9];
        free(sizes);
        uint8_t *reference = read_named(&fixture, "reference_u8", index,
            H3_DTYPE_U8, (size_t)3 * (size_t)height * (size_t)width);
        snprintf(paths[index], sizeof(paths[index]), "%s/reference%d.png",
                 directory, index);
        write_reference(paths[index], reference, width, height);
        free(reference);
        references[index].path = paths[index];
        posterior[index] = read_named(&fixture, "noise", index, H3_DTYPE_F32,
            (size_t)LATENT_CHANNELS * (size_t)moments_h * (size_t)moments_w);
        aug[index] = read_named(&fixture, "reference_noise", index,
            H3_DTYPE_F32,
            (size_t)LATENT_CHANNELS * (size_t)latent_h * (size_t)latent_w);
        row_elements[index] = (size_t)latent_h * (size_t)latent_w / 4 * 96;
        want_rows[index] = read_named(&fixture, "noised_rows", index,
                                      H3_DTYPE_F32, row_elements[index]);
    }
    h3_image_params params = H3_IMAGE_PARAMS_DEFAULT;
    params.width = params.height = 0;
    params.min_generation_area = H3_IMAGE_MIN_AREA_LOW;
    params.references = references;
    params.reference_count = REFERENCE_CASES;
    h3_image_hooks hooks;
    memset(&hooks, 0, sizeof(hooks));
    hooks.references_only = 1;
    hooks.reference_posterior_noise = posterior;
    hooks.reference_aug_noise = aug;
    h3_result *result = h3_generate_image_with_hooks(
        ctx, "<Picture 1> <Picture 2> <Picture 3>", &params, &hooks);
    if (!result) die(h3_last_error(ctx));
    size_t total = 0;
    for (int index = 0; index < REFERENCE_CASES; index++)
        total += row_elements[index];
    if (hooks.condition_elements != total) die("condition row count");
    size_t offset = 0;
    for (int index = 0; index < REFERENCE_CASES; index++) {
        double error_l2 = relative_l2(hooks.condition_rows + offset,
                                      want_rows[index], row_elements[index]);
        printf("reference %d condition rows vs diffusers: rel-L2 %.3g "
               "(bound %.3g)\n", index, error_l2, REFERENCE_ROWS_BOUND);
        if (!(error_l2 < REFERENCE_ROWS_BOUND))
            die("condition rows are over the bound");
        offset += row_elements[index];
        free((void *)posterior[index]);
        free((void *)aug[index]);
        free(want_rows[index]);
        remove(paths[index]);
    }
    free(hooks.condition_rows);
    h3_result_free(result);
    h3_st_free_header(&fixture);
}

typedef struct {
    uint32_t magic;
    uint32_t stream;
    uint64_t hash;
    uint64_t elements;
} exchange_header;

static uint64_t hash_floats(const float *values, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    const unsigned char *bytes = (const unsigned char *)values;
    for (size_t index = 0; index < count * sizeof(float); index++) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* The final latents of `stream` and `bf16` must match bit for bit. The first
 * of the two to run stores its latent, its mode and a hash of the noise; the
 * second compares. A file of the same mode or of other noise is stale. */
static void check_exchange(const char *path, int stream, const float *noise,
                           const float *latent, size_t elements) {
    exchange_header header = {0x45445432u, stream ? 1u : 0u,
                              hash_floats(noise, elements), elements};
    FILE *file = fopen(path, "rb");
    if (!file) {
        file = fopen(path, "wb");
        if (!file || fwrite(&header, sizeof(header), 1, file) != 1 ||
            fwrite(latent, sizeof(float), elements, file) != elements ||
            fclose(file) != 0) die("cannot write the latent exchange file");
        printf("final latent stored for the stream/bf16 comparison\n");
        return;
    }
    exchange_header stored;
    if (fread(&stored, sizeof(stored), 1, file) != 1 ||
        stored.magic != header.magic) die("latent exchange file is invalid");
    if (stored.stream == header.stream)
        die("latent exchange file was written by this same mode: remove it");
    if (stored.hash != header.hash || stored.elements != header.elements)
        die("latent exchange file was written for other noise");
    float *other = malloc(elements * sizeof(*other));
    if (!other || fread(other, sizeof(float), elements, file) != elements)
        die("latent exchange file is short");
    fclose(file);
    size_t different = 0;
    for (size_t index = 0; index < elements; index++)
        if (memcmp(&other[index], &latent[index], sizeof(float))) different++;
    printf("final latent stream vs bf16 bit-identical: %s (%zu of %zu "
           "elements differ)\n", different ? "NO" : "yes", different, elements);
    free(other);
    if (different) die("stream and bf16 final latents are not bit-identical");
}

typedef struct {
    double psnr, within2, within16, latent;
} scores;

static scores score(const h3_image_hooks *hooks, const uint8_t *want,
                    size_t pixels, const float *want_latent,
                    size_t latent_elements) {
    int exact = 0;
    long within2 = 0, within8 = 0, within16 = 0;
    int maximum = 0;
    double square_error = 0.0;
    for (size_t pixel = 0; pixel < pixels; pixel++)
        for (size_t channel = 0; channel < 3; channel++) {
            int got = hooks->rgb[pixel * 3 + channel];
            int expected = want[channel * pixels + pixel];
            int delta = abs(got - expected);
            if (delta > maximum) maximum = delta;
            if (delta == 0) exact++;
            if (delta <= 2) within2++;
            if (delta <= 8) within8++;
            if (delta <= 16) within16++;
            square_error += (double)delta * (double)delta;
        }
    double bytes = (double)(3 * pixels);
    scores result;
    result.psnr = square_error > 0.0 ?
        10.0 * log10(255.0 * 255.0 / (square_error / bytes)) : 99.0;
    result.within2 = (double)within2 / bytes;
    result.within16 = (double)within16 / bytes;
    result.latent = 0.0;
    printf("  image vs reference: PSNR %.2f dB, max abs diff %d\n",
           result.psnr, maximum);
    printf("  bytes within +-0: %.4f, +-2: %.4f, +-8: %.4f, +-16: %.4f\n",
           (double)exact / bytes, result.within2, (double)within8 / bytes,
           result.within16);
    if (want_latent) {
        result.latent = relative_l2(hooks->latent, want_latent,
                                    latent_elements);
        printf("  final latent vs reference: rel-L2 %.6g\n", result.latent);
    }
    return result;
}

/* H3_MEASURE=1: every score bound only reports and the run continues, so one
 * full run measures all of them. Exact checks always fail the run. */
static int measure;
static int over_count;

static void bound(const char *what, double value, double limit, int greater) {
    int ok = greater ? value >= limit : value < limit;
    if (!ok) over_count++;
    if (measure) {
        printf("  [%s] %s: %.6g, bound %s %.6g\n", ok ? "PASS" : "OVER", what,
               value, greater ? ">=" : "<", limit);
        return;
    }
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_real_image_edit_e2e.c: %s is %.6g, "
                "bound %s %.6g\n", what, value, greater ? ">=" : "<", limit);
        exit(1);
    }
}

static void check_scores(const scores *got, const bounds *limit,
                         int latent_checked) {
    bound("image PSNR", got->psnr, limit->psnr, 1);
    bound("bytes within +-16 of the reference", got->within16,
          limit->within16, 1);
    if (latent_checked) bound("final latent rel-L2", got->latent,
                              limit->latent, 0);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL_DIR EDIT_E2E_FIXTURE|- "
                "stream|bf16|int8|refs [EDIT_VAE_FIXTURE [EXCHANGE_FILE]]\n"
                "  refs: only the reference-row check against EDIT_VAE_FIXTURE "
                "(needs the VAE alone)\n", argv[0]);
        return 2;
    }
    const char *mode = argv[3];
    measure = getenv("H3_MEASURE") && *getenv("H3_MEASURE") &&
              strcmp(getenv("H3_MEASURE"), "0");
    int refs_only = !strcmp(mode, "refs");
    int stream = !strcmp(mode, "stream");
    int int8 = !strcmp(mode, "int8");
    if (!refs_only && !stream && !int8 && strcmp(mode, "bf16"))
        die("mode must be stream, bf16, int8 or refs");
    const char *vae_fixture = argc > 4 && strcmp(argv[4], "-") ? argv[4] : NULL;
    const char *exchange_path = argc > 5 ? argv[5] : NULL;
    if (refs_only && !vae_fixture) die("mode refs needs EDIT_VAE_FIXTURE");
    char error[512];
    char directory[256];
    const char *base = getenv("TMPDIR");
    snprintf(directory, sizeof(directory), "%s/h3_edit_e2e_XXXXXX",
             base && *base ? base : "/tmp");
    if (!mkdtemp(directory)) die("cannot create a temporary directory");

    setenv("H3_PROFILE", "1", 1);
    h3_ctx *ctx = h3_load_dir(argv[1]);
    if (!ctx) die(h3_last_error(NULL));
    if (vae_fixture) check_reference_rows(ctx, vae_fixture, directory);
    if (refs_only) {
        h3_free(ctx);
        rmdir(directory);
        puts("ok: reference rows match diffusers");
        return 0;
    }

    h3_st_header fixture;
    if (!h3_st_read_header(argv[2], &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *prompt_tensor = h3_st_find(&fixture, "prompt");
    if (!prompt_tensor) die("fixture has no prompt");
    size_t prompt_bytes = h3_st_tensor_elements(prompt_tensor);
    char *prompt = read_fixture(&fixture, "prompt", H3_DTYPE_U8, prompt_bytes);
    prompt = realloc(prompt, prompt_bytes + 1);
    if (!prompt) die("oom");
    prompt[prompt_bytes] = '\0';
    int width = read_scalar(&fixture, "width");
    int height = read_scalar(&fixture, "height");
    int min_area = read_scalar(&fixture, "min_generation_area");
    int canvas_height = read_scalar(&fixture, "canvas_height");
    int canvas_width = read_scalar(&fixture, "canvas_width");
    int seed = read_scalar(&fixture, "seed");
    if (canvas_height != height || canvas_width != width)
        die("the fixture must be generated on its own size");
    size_t latent_elements = (size_t)LATENT_CHANNELS *
                             (size_t)(height / 16) * (size_t)(width / 16);
    size_t pixels = (size_t)width * (size_t)height;
    float *noise = read_fixture(&fixture, "noise", H3_DTYPE_F32,
                                latent_elements);
    uint8_t *want = read_fixture(&fixture, "image_u8", H3_DTYPE_U8,
                                 3 * pixels);
    const h3_st_tensor *reference_tensor = h3_st_find(&fixture, "reference_u8");
    if (!reference_tensor || reference_tensor->ndim != 3 ||
        reference_tensor->shape[0] != 3) die("fixture has no reference_u8");
    int reference_height = (int)reference_tensor->shape[1];
    int reference_width = (int)reference_tensor->shape[2];
    uint8_t *reference = read_fixture(&fixture, "reference_u8", H3_DTYPE_U8,
        (size_t)3 * (size_t)reference_height * (size_t)reference_width);
    h3_image_reference_geometry geometry;
    if (!h3_image_edit_reference_geometry(reference_width, reference_height,
                                          &geometry))
        die("the fixture reference has no geometry");
    size_t posterior_elements = (size_t)LATENT_CHANNELS *
        (size_t)(geometry.crop_h / 16) * (size_t)(geometry.crop_w / 16);
    size_t aug_elements = (size_t)LATENT_CHANNELS * (size_t)geometry.latent_h *
                          (size_t)geometry.latent_w;
    float *posterior = read_fixture(&fixture, "reference_posterior_noise_0",
                                    H3_DTYPE_F32, posterior_elements);
    float *aug = read_fixture(&fixture, "reference_aug_noise_0", H3_DTYPE_F32,
                              aug_elements);
    /* vae_u8 is the VAE copy; vision_u8 must be the image the Qwen vision
     * tower receives, that is after smart_resize, not the first-stage image
     * that the processor still resizes. */
    require_shape(&fixture, "vae_u8", geometry.vae_h, geometry.vae_w,
                  "the VAE copy");
    require_shape(&fixture, "vision_u8", geometry.vision_h, geometry.vision_w,
                  "the vision tower input after smart_resize");
    uint8_t *vae_pixels = read_fixture(&fixture, "vae_u8", H3_DTYPE_U8,
        (size_t)3 * (size_t)geometry.vae_h * (size_t)geometry.vae_w);
    uint8_t *vision_pixels = read_fixture(&fixture, "vision_u8", H3_DTYPE_U8,
        (size_t)3 * (size_t)geometry.vision_h * (size_t)geometry.vision_w);
    size_t condition_elements = (size_t)geometry.dit_rows * 96;
    float *want_condition = h3_st_find(&fixture, "condition_rows") ?
        read_fixture(&fixture, "condition_rows", H3_DTYPE_F32,
                     condition_elements) : NULL;
    const h3_st_tensor *token_tensor = h3_st_find(&fixture, "token_ids");
    size_t want_tokens = token_tensor ? h3_st_tensor_elements(token_tensor) : 0;
    float *want_latent = NULL;
    if (h3_st_find(&fixture, "latents_final"))
        want_latent = read_fixture(&fixture, "latents_final", H3_DTYPE_F32,
                                   latent_elements);
    else
        printf("no final latent in the fixture: latent check skipped\n");

    char reference_path[600];
    snprintf(reference_path, sizeof(reference_path), "%s/reference.png",
             directory);
    write_reference(reference_path, reference, reference_width,
                    reference_height);
    h3_image_reference references[1] = {{reference_path}};
    h3_image_params params = H3_IMAGE_PARAMS_DEFAULT;
    params.width = width;
    params.height = height;
    params.min_generation_area = min_area;
    params.seed = (uint64_t)seed;
    params.ssd_streaming = stream;
    params.use_int8 = int8;
    params.on_progress = progress;
    params.references = references;
    params.reference_count = 1;
    printf("mode %s: \"%s\" at %dx%d with a %dx%d reference, min area %d, "
           "seed %d\n", mode, prompt, width, height, reference_width,
           reference_height, min_area, seed);

    const float *posterior_hook[1] = {posterior};
    const float *aug_hook[1] = {aug};
    const uint8_t *vae_hook[1] = {vae_pixels};
    const uint8_t *vision_hook[1] = {vision_pixels};
    for (int own = 0; own < 2; own++) {
        printf("%s\n", own ? "own resampling of the reference:" :
                             "parity run (fixture resized copies):");
        h3_image_hooks hooks;
        memset(&hooks, 0, sizeof(hooks));
        hooks.noise = noise;
        hooks.reference_posterior_noise = posterior_hook;
        hooks.reference_aug_noise = aug_hook;
        if (!own) {
            hooks.vae_pixels = vae_hook;
            hooks.vision_pixels = vision_hook;
        }
        double started = seconds_now();
        h3_result *result = h3_generate_image_with_hooks(ctx, prompt, &params,
                                                         &hooks);
        if (!result) die(h3_last_error(ctx));
        printf("  generation: %.1f s\n", seconds_now() - started);
        printf("  weights that ran: %s%s\n", hooks.weight_mode,
               strcmp(hooks.weight_mode, mode) ? " (NOT the requested mode)" :
                                                 "");
        if (result->width != width || result->height != height ||
            !hooks.rgb || !hooks.latent) die("result has the wrong shape");
        if (hooks.condition_elements != condition_elements)
            die("condition row count");
        if (want_condition) {
            double rows_error = relative_l2(hooks.condition_rows,
                                            want_condition, condition_elements);
            printf("  condition rows vs diffusers: rel-L2 %.3g (bound %.3g)\n",
                   rows_error, REFERENCE_ROWS_BOUND);
            if (!own && !(rows_error < REFERENCE_ROWS_BOUND))
                die("condition rows are over the bound");
        }
        if (want_tokens) {
            printf("  text tokens %zu, fixture %zu\n", hooks.text_tokens,
                   want_tokens);
            if (hooks.text_tokens != want_tokens)
                die("text presentation length differs from the fixture");
        }
        scores got = score(&hooks, want, pixels, want_latent, latent_elements);
        int is_int8 = !strcmp(hooks.weight_mode, "int8");
        const bounds *limit = own ? (is_int8 ? &OWN_INT8_BOUNDS :
                                               &OWN_BF16_BOUNDS) :
                                    (is_int8 ? &INT8_BOUNDS : &BF16_BOUNDS);
        check_scores(&got, limit, want_latent != NULL);
        if (!own) {
            if (exchange_path && (stream || !strcmp(hooks.weight_mode, "bf16")))
                check_exchange(exchange_path, stream, noise, hooks.latent,
                               latent_elements);
        }
        free(hooks.condition_rows);
        free(hooks.rgb);
        free(hooks.latent);
        h3_result_free(result);
    }
    remove(reference_path);
    rmdir(directory);
    h3_free(ctx);
    free(noise);
    free(want);
    free(want_latent);
    free(want_condition);
    free(reference);
    free(posterior);
    free(aug);
    free(vae_pixels);
    free(vision_pixels);
    free(prompt);
    if (over_count) {
        printf("measure mode: %d bound(s) OVER\n", over_count);
        return 0;
    }
    printf("ok: image edit end to end %s matches the diffusers reference\n",
           mode);
    return 0;
}
