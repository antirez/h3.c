#include "h3_dit.h"
#include "h3_image.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 5376, TEXT_DIM = 5120, PATCH = 96 };

/* The reference is the diffusers transformer run on CPU in BF16. Against an
 * F32 run of the same checkpoint (not part of the fixtures) it is 2.2e-2 off
 * at step 0 and 1.9e-2 at step 3 in global rel-L2; this implementation in BF16
 * is 2.0e-2 and 1.4e-2 off, so it is closer to the truth than the reference
 * is, and 2.8e-2 and 1.8e-2 off the reference itself. The BF16 bounds are per
 * step, about 1.4x above that. The int8 run is 4.0e-2 and 3.4e-2 off the F32
 * truth and 4.4e-2 and 3.5e-2 off the reference; its bounds are 2x the latter.
 * The refined text of the fixture is 4.1e-3 from the F32 run of the refiner
 * and this implementation is 4.9e-3 from it, so its bound is about twice that.
 * The six heads are close to each other: fed the same hidden state, a wrong
 * head differs from the right one by 6e-2 to 9e-2 at step 0 (caught by the
 * step 0 bound) but only by 1.1e-2 to 5e-2 at step 3, which these bounds catch
 * only in part. Head selection is pinned by the host tests; the median over
 * rows is a second criterion that the rows with the largest values cannot
 * dominate. */
#define REFINED_TEXT_BOUND 1e-2
static const double BF16_VELOCITY_BOUND[2] = {4e-2, 2.5e-2};
static const double BF16_ROW_MEDIAN_BOUND[2] = {3.5e-2, 2e-2};
static const double INT8_VELOCITY_BOUND[2] = {9e-2, 7e-2};
static const double INT8_ROW_MEDIAN_BOUND[2] = {8.5e-2, 5e-2};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_dit.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_dit.c: fixture tensor %s "
                "is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static int32_t read_scalar(const h3_st_header *fixture, const char *name) {
    int32_t *data = read_fixture(fixture, name, H3_DTYPE_I32, 1);
    int32_t value = data[0];
    free(data);
    return value;
}

static float bf16_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

typedef struct {
    double l2, maximum, row_median, row_worst;
} comparison;

/* Per-row relative L2 error: the global L2 is dominated by a few large
 * elements, the median over rows is not. */
static comparison compare(const float *got, const float *want, size_t rows,
                          size_t width) {
    double *row_errors = malloc(rows * sizeof(*row_errors));
    require(row_errors != NULL, "comparison allocation failed");
    comparison result = {0.0, 0.0, 0.0, 0.0};
    double square_error = 0.0, square_value = 0.0;
    for (size_t row = 0; row < rows; row++) {
        double row_error = 0.0, row_value = 0.0;
        for (size_t index = row * width; index < (row + 1) * width; index++) {
            double delta = (double)got[index] - (double)want[index];
            if (fabs(delta) > result.maximum) result.maximum = fabs(delta);
            row_error += delta * delta;
            row_value += (double)want[index] * (double)want[index];
        }
        square_error += row_error;
        square_value += row_value;
        row_errors[row] = sqrt(row_error / (row_value > 1e-24 ? row_value :
                                                                 1e-24));
    }
    qsort(row_errors, rows, sizeof(*row_errors), compare_doubles);
    result.row_median = row_errors[rows / 2];
    result.row_worst = row_errors[rows - 1];
    result.l2 = sqrt(square_error /
                     (square_value > 1e-24 ? square_value : 1e-24));
    free(row_errors);
    return result;
}

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void progress(const char *phase, int completed, int total,
                     void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "  %s %d/%d\n", phase, completed, total);
}

/* The velocity of `stream` and `bf16` must match bit for bit: the first of the
 * two to run stores its velocities in `path`, the second compares. */
static void check_exchange(const char *path, const float *velocity,
                           size_t elements, const char *mode, int step) {
    char name[1024];
    snprintf(name, sizeof(name), "%s.step%d", path, step);
    FILE *file = fopen(name, "rb");
    if (!file) {
        file = fopen(name, "wb");
        require(file != NULL, "cannot create the velocity exchange file");
        require(fwrite(velocity, sizeof(*velocity), elements, file) ==
                    elements, "cannot write the velocity exchange file");
        fclose(file);
        printf("step %d: velocity stored for the %s comparison\n", step,
               strcmp(mode, "stream") ? "stream" : "bf16");
        return;
    }
    float *other = malloc(elements * sizeof(*other));
    require(other != NULL, "exchange allocation failed");
    require(fread(other, sizeof(*other), elements, file) == elements,
            "velocity exchange file is short");
    fclose(file);
    size_t different = 0;
    for (size_t index = 0; index < elements; index++)
        if (memcmp(&other[index], &velocity[index], sizeof(float)))
            different++;
    free(other);
    printf("step %d: stream vs bf16 velocity bit-identical: %s "
           "(%zu of %zu elements differ)\n", step,
           different ? "NO" : "yes", different, elements);
    require(different == 0, "stream and bf16 velocities are not bit-identical");
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s WEIGHT_DIR STEP_FIXTURE TEXT_FIXTURE "
                "stream|bf16|int8 [EXCHANGE_FILE]\n", argv[0]);
        return 2;
    }
    const char *weight_directory = argv[1];
    const char *mode = argv[2 + 2];
    const char *exchange = argc > 5 ? argv[5] : NULL;
    int stream = !strcmp(mode, "stream");
    int bf16 = !strcmp(mode, "bf16");
    int int8 = !strcmp(mode, "int8");
    require(stream || bf16 || int8, "mode must be stream, bf16 or int8");
    char error[512];

    h3_st_header text_fixture, step_fixture;
    if (!h3_st_read_header(argv[3], &text_fixture, error, sizeof(error)))
        die(error);
    if (!h3_st_read_header(argv[2], &step_fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *prompt = h3_st_find(&text_fixture, "prompt_embeds");
    require(prompt && prompt->ndim == 2 && prompt->shape[1] == TEXT_DIM,
            "bad prompt_embeds fixture");
    size_t tokens = (size_t)prompt->shape[0];
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    text.tokens = tokens;
    text.width = TEXT_DIM;
    text.values = read_fixture(&text_fixture, "prompt_embeds", H3_DTYPE_BF16,
                               tokens * TEXT_DIM);

    int latent_height = read_scalar(&step_fixture, "latent_height");
    int latent_width = read_scalar(&step_fixture, "latent_width");
    require(latent_height > 0 && latent_width > 0 &&
            latent_height % 2 == 0 && latent_width % 2 == 0,
            "bad latent size in the step fixture");
    size_t rows = (size_t)(latent_height / 2) * (size_t)(latent_width / 2);
    size_t elements = rows * PATCH;
    uint16_t *refined_want = read_fixture(&step_fixture, "refined_text",
                                          H3_DTYPE_BF16, tokens * HIDDEN);
    int32_t steps[2] = {read_scalar(&step_fixture, "step0"),
                        read_scalar(&step_fixture, "step3")};
    const char *input_names[2] = {"latents_in_step0", "latents_in_step3"};
    const char *output_names[2] = {"noise_pred_step0", "noise_pred_step3"};
    float *inputs[2], *wanted[2];
    for (int index = 0; index < 2; index++) {
        inputs[index] = read_fixture(&step_fixture, input_names[index],
                                     H3_DTYPE_F32, elements);
        wanted[index] = read_fixture(&step_fixture, output_names[index],
                                     H3_DTYPE_F32, elements);
    }

    printf("mode %s: %zu text tokens, latent %dx%d, %zu image rows\n", mode,
           tokens, latent_height, latent_width, rows);
    double started = seconds_now();
    h3_dit *dit = h3_dit_load_image(
        weight_directory, "h3_shaders.metal", &text, latent_height,
        latent_width, 50, stream, bf16, bf16, bf16, 0, 0, 0, 0, 0, 0, 0, 0,
        progress, NULL, error, sizeof(error));
    if (!dit) die(error);
    printf("load: %.1f s\n", seconds_now() - started);
    require(h3_dit_image_rows(dit) == rows, "image row count mismatch");

    uint16_t *refined_got = malloc(tokens * HIDDEN * sizeof(*refined_got));
    float *refined_got_f32 = malloc(tokens * HIDDEN * sizeof(float));
    float *refined_want_f32 = malloc(tokens * HIDDEN * sizeof(float));
    require(refined_got && refined_got_f32 && refined_want_f32, "oom");
    require(h3_dit_read_refined_text(dit, refined_got, tokens * HIDDEN),
            "cannot read the refined text");
    for (size_t index = 0; index < tokens * HIDDEN; index++) {
        refined_got_f32[index] = bf16_value(refined_got[index]);
        refined_want_f32[index] = bf16_value(refined_want[index]);
    }
    comparison refined = compare(refined_got_f32, refined_want_f32, tokens,
                                 HIDDEN);
    printf("refined text: rel-L2 %.6g, max abs %.6g, median row %.6g\n",
           refined.l2, refined.maximum, refined.row_median);
    require(refined.l2 < REFINED_TEXT_BOUND, "refined text is off");

    float *velocity = malloc(elements * sizeof(*velocity));
    require(velocity != NULL, "oom");
    for (int index = 0; index < 2; index++) {
        double forward_started = seconds_now();
        if (!h3_dit_forward_image(dit, steps[index], inputs[index], velocity,
                                  error, sizeof(error))) die(error);
        double elapsed = seconds_now() - forward_started;
        for (size_t element = 0; element < elements; element++)
            require(isfinite(velocity[element]), "non-finite velocity");
        comparison result = compare(velocity, wanted[index], rows, PATCH);
        printf("step %d: velocity rel-L2 %.6g, max abs %.6g, median row "
               "%.6g, worst row %.6g (%.2f s)\n", steps[index], result.l2,
               result.maximum, result.row_median, result.row_worst, elapsed);
        require(result.l2 < (int8 ? INT8_VELOCITY_BOUND[index] :
                                    BF16_VELOCITY_BOUND[index]),
                "velocity rel-L2 over the bound");
        require(result.row_median < (int8 ? INT8_ROW_MEDIAN_BOUND[index] :
                                            BF16_ROW_MEDIAN_BOUND[index]),
                "velocity median row error over the bound");
        if (exchange && !int8)
            check_exchange(exchange, velocity, elements, mode, steps[index]);
    }
    h3_gpu_stats stats;
    if (h3_dit_get_gpu_stats(dit, &stats))
        printf("GPU buffers: peak live %.2f GiB, %.3f GPU seconds\n",
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               stats.gpu_seconds);
    h3_dit_free(dit);
    printf("ok: image DiT %s matches the diffusers reference\n", mode);
    return 0;
}
