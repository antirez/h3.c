#include "h3_dit.h"
#include "h3_image.h"
#include "h3_image_pipeline.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 5376, TEXT_DIM = 5120, PATCH = 96, FIXTURE_STEPS = 6 };

/* All six steps, each fed the BF16 reference's own latents. The reference is
 * the diffusers transformer run on CPU in BF16. Global rel-L2 per step:
 *   reference vs F32 truth:  2.2e-2 3.3e-2 4.0e-2 1.9e-2 2.0e-2 2.7e-2
 *   BF16 / streaming vs ref: 2.8e-2 4.4e-2 4.4e-2 1.8e-2 2.1e-2 2.6e-2
 *   BF16 / streaming vs truth: 2.0e-2 3.5e-2 2.8e-2 1.4e-2 1.4e-2 2.0e-2
 *   int8 vs reference:       4.4e-2 4.3e-2 8.7e-2 3.5e-2 2.9e-2 4.3e-2
 *   int8 vs truth:           4.0e-2 4.8e-2 7.9e-2 3.4e-2 2.7e-2 4.2e-2
 * so this implementation in BF16 is closer to the truth than the reference is,
 * and int8 is about twice as far (step 2 is the worst for every path). The
 * bounds are about 1.4x the achieved value, per step and mode, for the global
 * rel-L2 and the median over rows. The refined text of the fixture is 4.1e-3
 * from its F32 run and this implementation is 4.9e-3 from it.
 * The six output heads are close to each other, so no velocity bound tells
 * them apart. The head binding is checked exactly instead: the velocity must
 * equal the head of its step applied to the final hidden state (rel-L2 <
 * HEAD_MATCH_BOUND) and differ from every other head's. */
#define REFINED_TEXT_BOUND 1e-2
#define REFINED_TRUTH_BOUND 7e-3
#define HEAD_MATCH_BOUND 1e-5
#define HEAD_MISMATCH_BOUND 1e-3
static const double BF16_VELOCITY_BOUND[FIXTURE_STEPS] = {
    4e-2, 6.2e-2, 6.2e-2, 2.5e-2, 3e-2, 3.7e-2};
static const double BF16_ROW_MEDIAN_BOUND[FIXTURE_STEPS] = {
    3.5e-2, 4.5e-2, 4.7e-2, 1.8e-2, 1.9e-2, 1.7e-2};
static const double INT8_VELOCITY_BOUND[FIXTURE_STEPS] = {
    6.2e-2, 6.1e-2, 1.22e-1, 4.9e-2, 4.1e-2, 6.1e-2};
static const double INT8_ROW_MEDIAN_BOUND[FIXTURE_STEPS] = {
    5.9e-2, 4.8e-2, 7.5e-2, 3.5e-2, 3.4e-2, 5.2e-2};
static const double BF16_TRUTH_BOUND[FIXTURE_STEPS] = {
    2.9e-2, 4.9e-2, 4e-2, 1.9e-2, 2e-2, 2.8e-2};
static const double BF16_TRUTH_MEDIAN_BOUND[FIXTURE_STEPS] = {
    2.2e-2, 4e-2, 2.6e-2, 1.3e-2, 1.3e-2, 1.5e-2};
static const double INT8_TRUTH_BOUND[FIXTURE_STEPS] = {
    5.6e-2, 6.8e-2, 1.12e-1, 4.8e-2, 3.8e-2, 5.9e-2};
static const double INT8_TRUTH_MEDIAN_BOUND[FIXTURE_STEPS] = {
    5.3e-2, 5.4e-2, 7e-2, 3.5e-2, 3.2e-2, 5.2e-2};

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

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= p[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

typedef struct {
    uint32_t magic;
    uint32_t mode;
    uint32_t tokens;
    uint32_t latent_height;
    uint32_t latent_width;
    uint32_t input_hash_low;
    uint32_t input_hash_high;
    uint32_t elements;
} exchange_header;

#define EXCHANGE_MAGIC 0x56583348u

/* The velocities of `stream` and `bf16` must match bit for bit. The first of
 * the two to run stores its velocities, with its mode and a hash of the
 * inputs, in `path`; the second compares. A file of the same mode or of other
 * inputs is a stale leftover and fails. */
static void check_exchange(const char *path, const exchange_header *header,
                           float *const velocity[FIXTURE_STEPS]) {
    size_t elements = header->elements;
    FILE *file = fopen(path, "rb");
    if (!file) {
        file = fopen(path, "wb");
        require(file != NULL, "cannot create the velocity exchange file");
        int ok = fwrite(header, sizeof(*header), 1, file) == 1;
        for (int index = 0; ok && index < FIXTURE_STEPS; index++)
            ok = fwrite(velocity[index], sizeof(float), elements, file) ==
                 elements;
        require(fclose(file) == 0 && ok,
                "cannot write the velocity exchange file");
        printf("velocities stored for the %s comparison\n",
               header->mode ? "bf16" : "stream");
        return;
    }
    exchange_header stored;
    require(fread(&stored, sizeof(stored), 1, file) == 1 &&
            stored.magic == EXCHANGE_MAGIC,
            "velocity exchange file has no valid header");
    require(stored.mode != header->mode,
            "velocity exchange file was written by this same mode: "
            "remove the stale file");
    require(stored.tokens == header->tokens &&
            stored.latent_height == header->latent_height &&
            stored.latent_width == header->latent_width &&
            stored.input_hash_low == header->input_hash_low &&
            stored.input_hash_high == header->input_hash_high &&
            stored.elements == header->elements,
            "velocity exchange file was written for other inputs");
    float *other = malloc(elements * sizeof(*other));
    require(other != NULL, "exchange allocation failed");
    for (int step = 0; step < FIXTURE_STEPS; step++) {
        require(fread(other, sizeof(*other), elements, file) == elements,
                "velocity exchange file is short");
        size_t different = 0;
        for (size_t index = 0; index < elements; index++)
            if (memcmp(&other[index], &velocity[step][index], sizeof(float)))
                different++;
        printf("fixture step %d: stream vs bf16 velocity bit-identical: %s "
               "(%zu of %zu elements differ)\n", step,
               different ? "NO" : "yes", different, elements);
        require(different == 0,
                "stream and bf16 velocities are not bit-identical");
    }
    free(other);
    fclose(file);
}

/* Relative L2 distance of the Metal velocity from head `head` applied in
 * double precision to the final hidden state. */
static double head_distance(const float *hidden, const float *weight,
                            const float *bias, const float *velocity,
                            size_t rows) {
    double square_error = 0.0, square_value = 0.0;
    for (size_t row = 0; row < rows; row++)
        for (size_t out = 0; out < PATCH; out++) {
            double sum = bias[out];
            for (size_t k = 0; k < HIDDEN; k++)
                sum += (double)hidden[row * HIDDEN + k] *
                       (double)weight[out * HIDDEN + k];
            double delta = (double)velocity[row * PATCH + out] - sum;
            square_error += delta * delta;
            square_value += sum * sum;
        }
    return sqrt(square_error / square_value);
}

static float *read_head_tensor(const h3_weight_store *weights,
                               const char *name, size_t elements) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(weights, name, &header);
    char error[512];
    require(tensor && header && tensor->dtype == H3_DTYPE_F32 &&
            h3_st_tensor_elements(tensor) == elements,
            "output head tensor is missing or has the wrong type or size");
    float *data = malloc(elements * sizeof(*data));
    require(data != NULL, "oom");
    if (!h3_st_read_data(header, tensor, data, elements * sizeof(*data),
                         error, sizeof(error))) die(error);
    return data;
}

typedef struct {
    uint32_t magic;
    uint32_t mode;
    uint32_t tokens;
    uint32_t latent_size;
    uint64_t hash;
    uint64_t elements;
} random_header;

#define RANDOM_MAGIC 0x52444d33u
#define RANDOM_STEPS 2
#define RANDOM_SEED 20241112u
#define RANDOM_BOUND 7e-2
#define RANDOM_DEFECT 1e-1

/* The same checkpoint at a larger row count with seeded random latents,
 * written by one mode and compared by the next: stream against bf16 must be
 * bit-identical, int8 against either is reported and bounded. mode_id is
 * 0 bf16, 1 stream, 2 int8. */
static void run_random(h3_dit *dit, const char *mode, uint32_t mode_id,
                       size_t tokens, int latent_size, const char *exchange) {
    char error[512];
    size_t rows = (size_t)(latent_size / 2) * (size_t)(latent_size / 2);
    size_t elements = rows * PATCH;
    float *latent = malloc(elements * sizeof(*latent));
    float *input = malloc(elements * sizeof(*input));
    require(latent && input, "oom");
    h3_image_noise(RANDOM_SEED, latent_size, latent_size, latent);
    require(h3_dit_patchify_video(latent, H3_IMAGE_LATENT_CHANNELS, 1,
                                  latent_size, latent_size, input, elements),
            "cannot patchify the random latent");
    free(latent);
    static const int steps[RANDOM_STEPS] = {0, 5};
    float *velocity[RANDOM_STEPS];
    for (int index = 0; index < RANDOM_STEPS; index++) {
        velocity[index] = malloc(elements * sizeof(float));
        require(velocity[index] != NULL, "oom");
        double started = seconds_now();
        require(h3_dit_forward_image(dit, steps[index], input, velocity[index],
                                     error, sizeof(error)), error);
        printf("random latents, step %d: %.2f s\n", steps[index],
               seconds_now() - started);
        for (size_t element = 0; element < elements; element++)
            require(isfinite(velocity[index][element]),
                    "non-finite velocity on random latents");
    }
    random_header header = {RANDOM_MAGIC, mode_id, (uint32_t)tokens,
                            (uint32_t)latent_size,
                            hash_bytes(1469598103934665603ull, input,
                                       elements * sizeof(float)), elements};
    free(input);
    FILE *file = exchange ? fopen(exchange, "rb") : NULL;
    if (!file) {
        if (exchange) {
            file = fopen(exchange, "wb");
            int ok = file && fwrite(&header, sizeof(header), 1, file) == 1;
            for (int index = 0; ok && index < RANDOM_STEPS; index++)
                ok = fwrite(velocity[index], sizeof(float), elements, file) ==
                     elements;
            require(file && fclose(file) == 0 && ok,
                    "cannot write the random exchange file");
            printf("velocities stored for the comparison with the next mode\n");
        }
        return;
    }
    random_header stored;
    require(fread(&stored, sizeof(stored), 1, file) == 1 &&
            stored.magic == RANDOM_MAGIC, "random exchange file is invalid");
    require(stored.mode != header.mode,
            "random exchange file was written by this same mode");
    require(stored.tokens == header.tokens &&
            stored.latent_size == header.latent_size &&
            stored.hash == header.hash && stored.elements == header.elements,
            "random exchange file was written for other inputs");
    int int8_involved = stored.mode == 2 || header.mode == 2;
    float *other = malloc(elements * sizeof(*other));
    require(other != NULL, "oom");
    for (int index = 0; index < RANDOM_STEPS; index++) {
        require(fread(other, sizeof(float), elements, file) == elements,
                "random exchange file is short");
        if (!int8_involved) {
            size_t different = 0;
            for (size_t element = 0; element < elements; element++)
                if (memcmp(&other[element], &velocity[index][element],
                           sizeof(float))) different++;
            printf("random latents, step %d: stream vs bf16 bit-identical: %s "
                   "(%zu of %zu differ)\n", steps[index],
                   different ? "NO" : "yes", different, elements);
            require(different == 0,
                    "stream and bf16 velocities are not bit-identical");
            continue;
        }
        const float *int8_values = header.mode == 2 ? velocity[index] : other;
        const float *exact = header.mode == 2 ? other : velocity[index];
        comparison result = compare(int8_values, exact, rows, PATCH);
        printf("random latents, %zu rows, step %d: int8 vs %s: rel-L2 %.6g, "
               "max abs %.6g, median row %.6g, worst row %.6g\n", rows,
               steps[index], header.mode == 2 ? (stored.mode ? "stream" : "bf16")
               : mode, result.l2, result.maximum, result.row_median,
               result.row_worst);
        require(isfinite(result.l2), "non-finite int8 comparison");
        require(result.l2 < RANDOM_DEFECT,
                "int8 differs from the exact path by 1e-1 or more: defect");
        require(result.l2 < RANDOM_BOUND,
                "int8 differs from the exact path by more than 7e-2");
    }
    free(other);
    fclose(file);
    for (int index = 0; index < RANDOM_STEPS; index++) free(velocity[index]);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s WEIGHT_DIR STEP_FIXTURE TEXT_FIXTURE "
                "stream|bf16|int8 [EXCHANGE_FILE|- [TRUTH_FIXTURE]]\n",
                argv[0]);
        return 2;
    }
    const char *weight_directory = argv[1];
    const char *mode = argv[4];
    const char *exchange = argc > 5 && strcmp(argv[5], "-") ? argv[5] : NULL;
    const char *truth_path = argc > 6 ? argv[6] : NULL;
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

    if (argc > 7) {
        int random_size = atoi(argv[7]);
        require(random_size >= 2 && random_size % 2 == 0,
                "latent size must be a positive even number");
        printf("mode %s: %zu text tokens, random latents %dx%d\n", mode,
               tokens, random_size, random_size);
        h3_dit *random_dit = h3_dit_load_image(
            weight_directory, "h3_shaders.metal", &text, random_size,
            random_size, NULL, 0, NULL, 0, 50, stream, bf16, bf16, bf16, 0, 0,
            0, 0, 0, 0, 0, 0, progress, NULL, error, sizeof(error));
        require(random_dit != NULL, error);
        run_random(random_dit, mode, bf16 ? 0u : stream ? 1u : 2u, tokens,
                   random_size, exchange);
        h3_dit_free(random_dit);
        printf("ok: random-latent run of %s at %dx%d\n", mode, random_size,
               random_size);
        return 0;
    }
    int latent_height = read_scalar(&step_fixture, "latent_height");
    int latent_width = read_scalar(&step_fixture, "latent_width");
    require(latent_height > 0 && latent_width > 0 &&
            latent_height % 2 == 0 && latent_width % 2 == 0,
            "bad latent size in the step fixture");
    size_t rows = (size_t)(latent_height / 2) * (size_t)(latent_width / 2);
    size_t elements = rows * PATCH;
    uint16_t *refined_want = read_fixture(&step_fixture, "refined_text",
                                          H3_DTYPE_BF16, tokens * HIDDEN);
    int32_t steps[FIXTURE_STEPS];
    static const char *const input_names[FIXTURE_STEPS] = {
        "latents_in_step0", "latents_in_step1", "latents_in_step2",
        "latents_in_step3", "latents_in_step4", "latents_in_step5"};
    static const char *const output_names[FIXTURE_STEPS] = {
        "noise_pred_step0", "noise_pred_step1", "noise_pred_step2",
        "noise_pred_step3", "noise_pred_step4", "noise_pred_step5"};
    static const char *const truth_names[FIXTURE_STEPS] = {
        "noise_pred_f32_step0", "noise_pred_f32_step1",
        "noise_pred_f32_step2", "noise_pred_f32_step3",
        "noise_pred_f32_step4", "noise_pred_f32_step5"};
    for (int index = 0; index < FIXTURE_STEPS; index++) {
        char name[16];
        snprintf(name, sizeof(name), "step%d", index);
        steps[index] = read_scalar(&step_fixture, name);
        require(steps[index] == index, "step fixture is not the six-step run");
    }
    float *inputs[FIXTURE_STEPS], *wanted[FIXTURE_STEPS];
    float *truth[FIXTURE_STEPS] = {NULL, NULL, NULL, NULL, NULL, NULL};
    float *refined_truth = NULL;
    for (int index = 0; index < FIXTURE_STEPS; index++) {
        inputs[index] = read_fixture(&step_fixture, input_names[index],
                                     H3_DTYPE_F32, elements);
        wanted[index] = read_fixture(&step_fixture, output_names[index],
                                     H3_DTYPE_F32, elements);
    }
    h3_st_header truth_fixture;
    if (truth_path && h3_st_read_header(truth_path, &truth_fixture, error,
                                        sizeof(error))) {
        refined_truth = read_fixture(&truth_fixture, "refined_text_f32",
                                     H3_DTYPE_F32, tokens * HIDDEN);
        for (int index = 0; index < FIXTURE_STEPS; index++)
            truth[index] = read_fixture(&truth_fixture, truth_names[index],
                                        H3_DTYPE_F32, elements);
    } else {
        printf("no F32 truth fixture: truth checks skipped\n");
    }

    uint64_t hash = 1469598103934665603ull;
    hash = hash_bytes(hash, text.values, tokens * TEXT_DIM * sizeof(uint16_t));
    for (int index = 0; index < FIXTURE_STEPS; index++)
        hash = hash_bytes(hash, inputs[index], elements * sizeof(float));
    exchange_header header = {
        EXCHANGE_MAGIC, stream ? 1u : 0u, (uint32_t)tokens,
        (uint32_t)latent_height, (uint32_t)latent_width,
        (uint32_t)hash, (uint32_t)(hash >> 32), (uint32_t)elements
    };

    char weights_error[512];
    h3_weight_store *weights = h3_weight_store_open(
        weight_directory, weights_error, sizeof(weights_error));
    if (!weights) die(weights_error);
    float *head_weight[H3_IMAGE_STEPS], *head_bias[H3_IMAGE_STEPS];
    for (unsigned head = 0; head < H3_IMAGE_STEPS; head++) {
        char name[96];
        snprintf(name, sizeof(name), "pdd_proj_out.%u.weight", head);
        head_weight[head] = read_head_tensor(weights, name,
                                             (size_t)PATCH * HIDDEN);
        snprintf(name, sizeof(name), "pdd_proj_out.%u.bias", head);
        head_bias[head] = read_head_tensor(weights, name, PATCH);
    }

    printf("mode %s: %zu text tokens, latent %dx%d, %zu image rows\n", mode,
           tokens, latent_height, latent_width, rows);
    double started = seconds_now();
    h3_dit *dit = h3_dit_load_image(
        weight_directory, "h3_shaders.metal", &text, latent_height,
        latent_width, NULL, 0, NULL, 0, 50, stream, bf16, bf16, bf16, 0, 0, 0,
        0, 0, 0, 0, 0, progress, NULL, error, sizeof(error));
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
    printf("refined text vs reference: rel-L2 %.6g, max abs %.6g, median row "
           "%.6g\n", refined.l2, refined.maximum, refined.row_median);
    require(refined.l2 < REFINED_TEXT_BOUND, "refined text is off");
    if (refined_truth) {
        comparison against = compare(refined_got_f32, refined_truth, tokens,
                                     HIDDEN);
        printf("refined text vs F32 truth: rel-L2 %.6g, median row %.6g\n",
               against.l2, against.row_median);
        require(against.l2 < REFINED_TRUTH_BOUND,
                "refined text is far from the F32 truth");
    }

    float *velocity[FIXTURE_STEPS];
    float *hidden = malloc(rows * HIDDEN * sizeof(*hidden));
    require(hidden != NULL, "oom");
    for (int index = 0; index < FIXTURE_STEPS; index++) {
        velocity[index] = malloc(elements * sizeof(float));
        require(velocity[index] != NULL, "oom");
        double forward_started = seconds_now();
        if (!h3_dit_forward_image(dit, steps[index], inputs[index],
                                  velocity[index], error, sizeof(error)))
            die(error);
        double elapsed = seconds_now() - forward_started;
        for (size_t element = 0; element < elements; element++)
            require(isfinite(velocity[index][element]),
                    "non-finite velocity");
        comparison result = compare(velocity[index], wanted[index], rows,
                                    PATCH);
        printf("step %d vs reference: rel-L2 %.6g, max abs %.6g, median row "
               "%.6g, worst row %.6g (%.2f s)\n", steps[index], result.l2,
               result.maximum, result.row_median, result.row_worst, elapsed);
        require(result.l2 < (int8 ? INT8_VELOCITY_BOUND[index] :
                                    BF16_VELOCITY_BOUND[index]),
                "velocity rel-L2 over the bound");
        require(result.row_median < (int8 ? INT8_ROW_MEDIAN_BOUND[index] :
                                            BF16_ROW_MEDIAN_BOUND[index]),
                "velocity median row error over the bound");
        if (truth[index]) {
            comparison against = compare(velocity[index], truth[index], rows,
                                         PATCH);
            printf("step %d vs F32 truth: rel-L2 %.6g, max abs %.6g, median "
                   "row %.6g, worst row %.6g\n", steps[index], against.l2,
                   against.maximum, against.row_median, against.row_worst);
            require(against.l2 < (int8 ? INT8_TRUTH_BOUND[index] :
                                         BF16_TRUTH_BOUND[index]),
                    "velocity rel-L2 against the F32 truth over the bound");
            require(against.row_median < (int8 ?
                        INT8_TRUTH_MEDIAN_BOUND[index] :
                        BF16_TRUTH_MEDIAN_BOUND[index]),
                    "velocity median row error against the truth over the "
                    "bound");
        }
        require(h3_dit_read_final_hidden(dit, hidden, rows * HIDDEN),
                "cannot read the final hidden state");
        for (unsigned head = 0; head < H3_IMAGE_STEPS; head++) {
            double distance = head_distance(hidden, head_weight[head],
                                            head_bias[head], velocity[index],
                                            rows);
            int right = (int)head == steps[index];
            printf("step %d velocity vs head %u: rel-L2 %.3g%s\n",
                   steps[index], head, distance, right ? " (its head)" : "");
            require(right ? distance < HEAD_MATCH_BOUND :
                            distance > HEAD_MISMATCH_BOUND,
                    right ? "velocity does not match the head of its step" :
                            "velocity matches a head of another step");
        }
    }

    /* A second pass in the reverse order must reproduce the first bit for
     * bit: nothing may carry over between forwards. */
    float *again = malloc(elements * sizeof(*again));
    require(again != NULL, "oom");
    for (int index = FIXTURE_STEPS - 1; index >= 0; index--) {
        require(h3_dit_forward_image(dit, steps[index], inputs[index], again,
                                     error, sizeof(error)), error);
        int same = !memcmp(again, velocity[index], elements * sizeof(float));
        printf("step %d repeated on the same DiT: bit-identical: %s\n",
               steps[index], same ? "yes" : "NO");
        require(same, "a repeated forward changed the velocity");
    }
    free(again);

    h3_gpu_stats stats;
    if (h3_dit_get_gpu_stats(dit, &stats))
        printf("GPU buffers: peak live %.2f GiB, %.3f GPU seconds\n",
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               stats.gpu_seconds);
    h3_dit_free(dit);
    if (exchange && !int8) check_exchange(exchange, &header, velocity);
    printf("ok: image DiT %s matches the diffusers reference\n", mode);
    return 0;
}
