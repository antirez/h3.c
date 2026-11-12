#include "h3_dit.h"
#include "h3_dit_names.h"
#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_image.h"
#include "h3_image_vae_encode.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    HIDDEN = 5376, TEXT_DIM = 5120, PATCH = 96, FIXTURE_STEPS = 6,
    HEADS = 56, HEAD_DIM = 128, INNER = HEADS * HEAD_DIM, FFN = 14336,
    TIME_INPUT = 256, TIME_DIM = 2688, ROPE_FREQS = 16, ROPE_HALF = 48,
    MODALITIES = 3, SLOTS = 6, MAX_TENSORS = 64, TIME_ROWS = 7,
    REFERENCE_TIME_ROW = 6, LATENT_CHANNELS = 24
};

/* Every bound below is a placeholder copied from the text-to-image test
 * (global rel-L2 and median over rows, about 1.4x the achieved values there)
 * and marked for the gated window, which measures the edit values. The
 * exact checks (the AdaLN index mapping, the condition rows, head binding,
 * bf16 = stream) have no bounds. */
#define REFINED_TEXT_BOUND 1e-2        /* measured in the gated window */
#define REFINED_TRUTH_BOUND 7e-3       /* measured in the gated window */
#define TIME_EMBEDDING_BOUND 1e-4      /* measured in the gated window */
#define BLOCK_MEDIAN_BOUND 2e-2        /* measured in the gated window */
#define ROW_Q99_BOUND 0.5              /* measured in the gated window */
#define HEAD_MATCH_BOUND 1e-5
#define HEAD_MISMATCH_BOUND 1e-3
#define CONDITION_MOVE_MIN 1e-2
static const double BF16_VELOCITY_BOUND[FIXTURE_STEPS] = {
    4e-2, 6.2e-2, 6.2e-2, 2.5e-2, 3e-2, 3.7e-2};   /* measured in the gated window */
static const double BF16_ROW_MEDIAN_BOUND[FIXTURE_STEPS] = {
    3.5e-2, 4.5e-2, 4.7e-2, 1.8e-2, 1.9e-2, 1.7e-2};   /* measured in the gated window */
static const double INT8_VELOCITY_BOUND[FIXTURE_STEPS] = {
    6.2e-2, 6.1e-2, 1.22e-1, 4.9e-2, 4.1e-2, 6.1e-2};   /* measured in the gated window */
static const double INT8_ROW_MEDIAN_BOUND[FIXTURE_STEPS] = {
    5.9e-2, 4.8e-2, 7.5e-2, 3.5e-2, 3.4e-2, 5.2e-2};   /* measured in the gated window */
static const double BF16_TRUTH_BOUND[FIXTURE_STEPS] = {
    2.9e-2, 4.9e-2, 4e-2, 1.9e-2, 2e-2, 2.8e-2};   /* measured in the gated window */
static const double BF16_TRUTH_MEDIAN_BOUND[FIXTURE_STEPS] = {
    2.2e-2, 4e-2, 2.6e-2, 1.3e-2, 1.3e-2, 1.5e-2};   /* measured in the gated window */
static const double INT8_TRUTH_BOUND[FIXTURE_STEPS] = {
    5.6e-2, 6.8e-2, 1.12e-1, 4.8e-2, 3.8e-2, 5.9e-2};   /* measured in the gated window */
static const double INT8_TRUTH_MEDIAN_BOUND[FIXTURE_STEPS] = {
    5.3e-2, 5.4e-2, 7e-2, 3.5e-2, 3.2e-2, 5.2e-2};   /* measured in the gated window */

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_edit_dit.c: %s\n", message);
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
        fprintf(stderr, "FAIL tests/test_real_image_edit_dit.c: fixture "
                "tensor %s is missing or has the wrong type or size\n", name);
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
    double l2, maximum, row_median, row_q99, row_worst;
} comparison;

/* Per-row relative L2 error: the global L2 is dominated by a few large
 * elements, the median and the 99th percentile over rows are not. */
static comparison compare(const float *got, const float *want, size_t rows,
                          size_t width) {
    double *row_errors = malloc(rows * sizeof(*row_errors));
    require(row_errors != NULL, "comparison allocation failed");
    comparison result = {0.0, 0.0, 0.0, 0.0, 0.0};
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
    result.row_q99 = row_errors[(rows * 99) / 100 < rows ?
                                (rows * 99) / 100 : rows - 1];
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
    uint32_t reference_rows;
    uint32_t latent_height;
    uint32_t latent_width;
    uint32_t input_hash_low;
    uint32_t input_hash_high;
    uint32_t elements;
} exchange_header;

#define EXCHANGE_MAGIC 0x45443348u

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
            stored.reference_rows == header->reference_rows &&
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
 * double precision to the final hidden state of the generated rows. */
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

/* ---- block 0 of the edit sequence, with two time rows ------------------ */

static h3_gpu *gpu;
static h3_gpu_tensor *owned[MAX_TENSORS];
static size_t owned_count;

static h3_gpu_tensor *own(h3_gpu_tensor *tensor, const char *what) {
    if (!tensor) {
        fprintf(stderr, "FAIL tests/test_real_image_edit_dit.c: %s: %s\n",
                what, gpu ? h3_gpu_error(gpu) : "no GPU");
        exit(1);
    }
    require(owned_count < MAX_TENSORS, "tensor registry overflow");
    owned[owned_count++] = tensor;
    return tensor;
}

static void release(h3_gpu_tensor **tensor) {
    for (size_t index = 0; index < owned_count; index++) {
        if (owned[index] != *tensor) continue;
        owned[index] = owned[--owned_count];
        break;
    }
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void gpu_call(int ok, const char *operation) {
    if (ok) return;
    fprintf(stderr, "FAIL tests/test_real_image_edit_dit.c: %s: %s\n",
            operation, h3_gpu_error(gpu));
    exit(1);
}

static void weight_name(char *out, size_t size, const h3_dit_names *names,
                        const char *suffix) {
    if (!h3_dit_block_name(out, size, names, 0, 0, suffix))
        die("cannot format a tensor name");
}

static h3_gpu_tensor *load_bf16(const h3_weight_store *weights,
                                const char *name, int ndim, uint64_t rows,
                                uint64_t columns) {
    uint64_t shape[2] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(
        weights, gpu, name, ndim, ndim == 1 ? &shape[0] : shape, error,
        sizeof(error));
    if (!tensor) die(error);
    return own(tensor, name);
}

static h3_gpu_tensor *new_bf16(size_t elements) {
    return own(h3_gpu_tensor_new_bf16(gpu, elements), "BF16 allocation");
}

typedef struct {
    h3_gpu_tensor *norm1, *norm2, *qkv, *q_norm, *k_norm, *out, *fc1, *fc2;
    h3_gpu_tensor *adaln_weight, *adaln_bias;
} block_weights;

static block_weights load_block(const h3_weight_store *weights,
                                const h3_dit_names *names) {
    block_weights result;
    char name[192], second[192], third[192], error[512];
    weight_name(name, sizeof(name), names, names->norm1);
    result.norm1 = load_bf16(weights, name, 1, HIDDEN, 0);
    weight_name(name, sizeof(name), names, names->norm2);
    result.norm2 = load_bf16(weights, name, 1, HIDDEN, 0);
    weight_name(name, sizeof(name), names, names->q);
    weight_name(second, sizeof(second), names, names->k);
    weight_name(third, sizeof(third), names, names->v);
    result.qkv = own(h3_weight_load_bf16_qkv(
        weights, gpu, name, second, third, HEADS, HEAD_DIM, HIDDEN, error,
        sizeof(error)), error);
    weight_name(name, sizeof(name), names, names->q_norm);
    result.q_norm = load_bf16(weights, name, 1, HEAD_DIM, 0);
    weight_name(name, sizeof(name), names, names->k_norm);
    result.k_norm = load_bf16(weights, name, 1, HEAD_DIM, 0);
    weight_name(name, sizeof(name), names, names->out);
    result.out = load_bf16(weights, name, 2, HIDDEN, INNER);
    weight_name(name, sizeof(name), names, names->fc2);
    result.fc2 = load_bf16(weights, name, 2, HIDDEN, FFN);
    weight_name(name, sizeof(name), names, names->adaln_weight);
    result.adaln_weight = load_bf16(weights, name, 2,
                                    MODALITIES * SLOTS * HIDDEN, TIME_DIM);
    weight_name(name, sizeof(name), names, names->adaln_bias);
    result.adaln_bias = load_bf16(weights, name, 1,
                                  MODALITIES * SLOTS * HIDDEN, 0);
    weight_name(name, sizeof(name), names, names->fc1);
    h3_gpu_tensor *fc1 = h3_weight_load_bf16_halves(
        weights, gpu, name, FFN * 2, HIDDEN, !H3_IMAGE_FC1_GATE_FIRST, error,
        sizeof(error));
    if (!fc1) die(error);
    result.fc1 = own(fc1, name);
    return result;
}

/* One block over `sequence` rows whose AdaLN entries index the table of
 * `time_rows` time embeddings: row_map = time row * 3 + modality tag. */
static void run_block(const block_weights *weight, h3_gpu_tensor *adaln_input,
                      uint32_t time_rows, h3_gpu_tensor *hidden,
                      h3_gpu_tensor *row_map, h3_gpu_tensor *rope_cos,
                      h3_gpu_tensor *rope_sin, uint32_t sequence) {
    size_t rows = sequence;
    h3_gpu_tensor *modulation = new_bf16((size_t)time_rows * MODALITIES *
                                         SLOTS * HIDDEN);
    h3_gpu_tensor *modulated = new_bf16(rows * HIDDEN);
    h3_gpu_tensor *qkv = new_bf16(rows * INNER * 3);
    h3_gpu_tensor *query = new_bf16(rows * INNER);
    h3_gpu_tensor *key = new_bf16(rows * INNER);
    h3_gpu_tensor *value = new_bf16(rows * INNER);
    h3_gpu_tensor *heads = new_bf16(rows * INNER);
    h3_gpu_tensor *branch = new_bf16(rows * HIDDEN);
    h3_gpu_tensor *fc1 = new_bf16(rows * FFN * 2);
    h3_gpu_tensor *activated = new_bf16(rows * FFN);
    gpu_call(h3_gpu_begin(gpu), "begin block");
    gpu_call(h3_gpu_linear_bf16(gpu, modulation, adaln_input,
                                weight->adaln_weight, weight->adaln_bias,
                                time_rows, TIME_DIM,
                                MODALITIES * SLOTS * HIDDEN),
             "AdaLN projection");
    gpu_call(h3_gpu_adaln_bf16(gpu, modulated, hidden, weight->norm1,
                               modulation, row_map, sequence, HIDDEN, SLOTS,
                               0, 1, 1e-5f), "attention AdaLN");
    gpu_call(h3_gpu_linear_bf16(gpu, qkv, modulated, weight->qkv, NULL,
                                sequence, HIDDEN, INNER * 3), "QKV");
    gpu_call(h3_gpu_grouped_qkv_rope_bf16(
                 gpu, query, key, value, qkv, weight->q_norm, weight->k_norm,
                 rope_cos, rope_sin, sequence, HEADS, HEAD_DIM, ROPE_HALF,
                 1e-5f), "QKV norm and RoPE");
    gpu_call(h3_gpu_sdpa_bf16(gpu, heads, query, key, value, sequence, HEADS,
                              HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
             "attention");
    gpu_call(h3_gpu_linear_bf16(gpu, branch, heads, weight->out, NULL,
                                sequence, INNER, HIDDEN), "attention output");
    gpu_call(h3_gpu_gate_bf16(gpu, hidden, hidden, branch, modulation,
                              row_map, sequence, HIDDEN, SLOTS, 2),
             "attention gate");
    gpu_call(h3_gpu_adaln_bf16(gpu, modulated, hidden, weight->norm2,
                               modulation, row_map, sequence, HIDDEN, SLOTS,
                               3, 4, 1e-5f), "MLP AdaLN");
    gpu_call(h3_gpu_linear_bf16(gpu, fc1, modulated, weight->fc1, NULL,
                                sequence, HIDDEN, FFN * 2), "FC1");
    gpu_call(h3_gpu_swiglu_bf16(gpu, activated, fc1, sequence, FFN),
             "SwiGLU");
    gpu_call(h3_gpu_linear_bf16(gpu, branch, activated, weight->fc2, NULL,
                                sequence, FFN, HIDDEN), "FC2");
    gpu_call(h3_gpu_gate_bf16(gpu, hidden, hidden, branch, modulation,
                              row_map, sequence, HIDDEN, SLOTS, 5),
             "MLP gate");
    gpu_call(h3_gpu_submit(gpu), "submit block");
    h3_gpu_tensor *scratch[] = {modulation, modulated, qkv, query, key, value,
                                heads, branch, fc1, activated};
    for (size_t index = 0; index < sizeof(scratch) / sizeof(*scratch); index++)
        release(&scratch[index]);
}

/* Block-0 output against the BF16 fixture (global) and the F32 run (median
 * over rows). */
static void compare_block(h3_gpu_tensor *got_tensor, const uint16_t *want,
                          const float *want_f32, size_t rows,
                          double *l2_reference, double *median_truth,
                          double *q99_truth) {
    size_t elements = rows * HIDDEN;
    uint16_t *got = malloc(elements * sizeof(*got));
    float *got_f32 = malloc(elements * sizeof(*got_f32));
    float *want_values = malloc(elements * sizeof(*want_values));
    require(got && got_f32 && want_values, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read the block output");
    for (size_t index = 0; index < elements; index++) {
        got_f32[index] = bf16_value(got[index]);
        want_values[index] = bf16_value(want[index]);
    }
    comparison reference = compare(got_f32, want_values, rows, HIDDEN);
    comparison truth = compare(got_f32, want_f32, rows, HIDDEN);
    *l2_reference = reference.l2;
    *median_truth = truth.row_median;
    *q99_truth = truth.row_q99;
    free(got); free(got_f32); free(want_values);
}

static double compare_f32(h3_gpu_tensor *got_tensor, const float *want,
                          size_t elements) {
    float *got = malloc(elements * sizeof(*got));
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_f32(got_tensor, got, elements),
            "cannot read the time embedding");
    double square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double delta = (double)got[index] - (double)want[index];
        square_error += delta * delta;
        square_value += (double)want[index] * (double)want[index];
    }
    free(got);
    return sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24));
}

/* Block 0 of step 0 with the fixture's two time rows [t_step, 0.999] and
 * its real adaln_indices (reference numbering): the time embeddings, then the
 * block output against the BF16 reference and the F32 run. */
static void check_block0(const char *weight_directory,
                         const h3_st_header *fixture, size_t sequence) {
    char error[512];
    float *positions = read_fixture(fixture, "position_ids", H3_DTYPE_F32,
                                    sequence * 3);
    int32_t *indices = read_fixture(fixture, "adaln_indices", H3_DTYPE_I32,
                                    sequence);
    float *temb = read_fixture(fixture, "temb_step0", H3_DTYPE_F32,
                               2 * TIME_DIM);
    float *timesteps = read_fixture(fixture, "timestep_step0", H3_DTYPE_F32,
                                    2);
    uint16_t *input = read_fixture(fixture, "hidden_in", H3_DTYPE_BF16,
                                   sequence * HIDDEN);
    uint16_t *want = read_fixture(fixture, "hidden_out", H3_DTYPE_BF16,
                                  sequence * HIDDEN);
    float *want_f32 = read_fixture(fixture, "hidden_out_f32", H3_DTYPE_F32,
                                   sequence * HIDDEN);
    h3_weight_store *weights = h3_weight_store_open(weight_directory, error,
                                                    sizeof(error));
    if (!weights) die(error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    const h3_dit_names *names = h3_dit_names_for(H3_DIT_FAMILY_IMAGE);

    h3_sigma_schedule schedule;
    require(h3_image_schedule_build(&schedule), "cannot build the schedule");
    require(timesteps[0] == 1.0f - schedule.video[0] && timesteps[1] == 0.999f,
            "the fixture's time values are not [1 - sigma, 0.999]");
    float features[2 * TIME_INPUT];
    h3_dit_time_features(timesteps[0], features);
    h3_dit_time_features(timesteps[1], features + TIME_INPUT);
    h3_gpu_tensor *computed = own(h3_dit_time_embed(
        weights, names, gpu, 2, features, error, sizeof(error)), error);
    double time_l2 = compare_f32(computed, temb, 2 * TIME_DIM);
    printf("block 0: time embeddings of [t_step, 0.999]: rel-L2 %.6g\n",
           time_l2);
    require(time_l2 < TIME_EMBEDDING_BOUND,
            "time embeddings differ from the reference");
    release(&computed);
    h3_gpu_tensor *time = own(h3_gpu_tensor_from_f32(gpu, temb, 2 * TIME_DIM),
                              "temb upload");
    h3_gpu_tensor *adaln_input = own(h3_dit_adaln_input(
        gpu, time, 2, H3_DIT_ADALN_SILU_THEN_CAST, error, sizeof(error)),
        error);

    uint32_t *rows = malloc(sequence * sizeof(*rows));
    float *cosines = malloc(sequence * ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(sequence * ROPE_HALF * sizeof(*sines));
    require(rows && cosines && sines, "host allocation failed");
    float inverse[ROPE_FREQS];
    h3_image_dit_inv_freq(inverse);
    for (size_t row = 0; row < sequence; row++) {
        require(indices[row] >= 0 && indices[row] < 2 * MODALITIES,
                "unexpected AdaLN row index");
        rows[row] = (uint32_t)indices[row];
        for (size_t axis = 0; axis < 3; axis++)
            for (size_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = row * ROPE_HALF + axis * ROPE_FREQS + frequency;
                float angle = positions[row * 3 + axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
    }
    h3_gpu_tensor *row_map = own(h3_gpu_tensor_from_u32(gpu, rows, sequence),
                                 "row map upload");
    h3_gpu_tensor *cos_f32 = own(h3_gpu_tensor_from_f32(
        gpu, cosines, sequence * ROPE_HALF), "RoPE cosine upload");
    h3_gpu_tensor *sin_f32 = own(h3_gpu_tensor_from_f32(
        gpu, sines, sequence * ROPE_HALF), "RoPE sine upload");
    h3_gpu_tensor *rope_cos = new_bf16(sequence * ROPE_HALF);
    h3_gpu_tensor *rope_sin = new_bf16(sequence * ROPE_HALF);
    gpu_call(h3_gpu_begin(gpu), "begin RoPE tables");
    gpu_call(h3_gpu_cast_f32_to_bf16(gpu, rope_cos, cos_f32,
                                     (uint32_t)(sequence * ROPE_HALF)),
             "RoPE cosine cast");
    gpu_call(h3_gpu_cast_f32_to_bf16(gpu, rope_sin, sin_f32,
                                     (uint32_t)(sequence * ROPE_HALF)),
             "RoPE sine cast");
    gpu_call(h3_gpu_submit(gpu), "submit RoPE tables");

    block_weights block = load_block(weights, names);
    h3_gpu_tensor *hidden = own(h3_gpu_tensor_from_bf16(
        gpu, input, sequence * HIDDEN), "hidden upload");
    run_block(&block, adaln_input, 2, hidden, row_map, rope_cos, rope_sin,
              (uint32_t)sequence);
    double l2_reference, median_truth, q99_truth;
    compare_block(hidden, want, want_f32, sequence, &l2_reference,
                  &median_truth, &q99_truth);
    printf("block 0 (%zu rows, two time rows): vs BF16 fixture rel-L2 %.6g; "
           "vs F32 run median row %.6g, q99 row %.6g\n", sequence,
           l2_reference, median_truth, q99_truth);
    require(median_truth < BLOCK_MEDIAN_BOUND,
            "block 0 median row error against the F32 run is over the bound");
    require(q99_truth < ROW_Q99_BOUND,
            "block 0 q99 row error against the F32 run is over the bound");

    free(positions); free(indices); free(temb); free(timesteps); free(input);
    free(want); free(want_f32); free(rows); free(cosines); free(sines);
    while (owned_count) h3_gpu_tensor_free(owned[--owned_count]);
    h3_gpu_free(gpu);
    gpu = NULL;
    h3_weight_store_free(weights);
}

/* ---- the transformer --------------------------------------------------- */

typedef struct {
    size_t tokens, reference_rows, rows, sequence;
    int latent_height, latent_width, reference_height, reference_width;
    h3_text_embedding text;
    float *condition_rows;
    int stream, bf16;
} edit_case;

static h3_dit *load_edit_dit(const char *weight_directory,
                             const edit_case *edit,
                             const float *condition_rows) {
    char error[512];
    h3_layout_ref reference = {H3_LAYOUT_REF_IMAGE, 1, edit->reference_height,
                               edit->reference_width, 0};
    h3_dit *dit = h3_dit_load_image(
        weight_directory, "h3_shaders.metal", &edit->text, edit->latent_height,
        edit->latent_width, &reference, 1, condition_rows,
        edit->reference_rows * PATCH, 50, edit->stream, edit->bf16,
        edit->bf16, edit->bf16, 0, 0, 0, 0, 0, 0, 0, 0, progress, NULL, error,
        sizeof(error));
    if (!dit) die(error);
    return dit;
}

/* (2) The AdaLN indices h3.c builds at every step, translated into the
 * reference's numbering (unique-timestep index * 3 + tag), equal the
 * fixture's adaln_indices; and the table has the six step rows plus one. */
static void check_adaln_mapping(const h3_dit *dit, const edit_case *edit,
                                const int32_t *adaln, const int32_t *indices,
                                const uint8_t *tags) {
    require(h3_dit_time_rows(dit) == TIME_ROWS,
            "an edit has the six step rows plus one reference row");
    uint32_t *rows = malloc(edit->sequence * sizeof(*rows));
    require(rows != NULL, "oom");
    size_t text = edit->tokens, reference_end = text + edit->reference_rows;
    for (int step = 0; step < FIXTURE_STEPS; step++) {
        require(h3_dit_read_row_map(dit, step, edit->text.tags, text, rows,
                                    edit->sequence),
                "cannot read the AdaLN row map");
        for (size_t row = 0; row < edit->sequence; row++) {
            uint32_t time_row = rows[row] / MODALITIES;
            uint32_t tag = rows[row] % MODALITIES;
            int is_reference = row >= text && row < reference_end;
            require(time_row == (is_reference ? REFERENCE_TIME_ROW :
                                                (uint32_t)step),
                    "a row has the wrong time row");
            require(tag == (row < text ? tags[row] : 0u),
                    "a row has the wrong modality tag");
            int32_t unique = is_reference ? 1 : 0;
            require(indices[row] == unique &&
                    adaln[row] == unique * MODALITIES + (int32_t)tag,
                    "the AdaLN index differs from the reference");
        }
    }
    free(rows);
    printf("AdaLN mapping: %d steps x %zu rows equal the fixture's "
           "adaln_indices (reference rows -> time row %d, table of %d rows)\n",
           FIXTURE_STEPS, edit->sequence, REFERENCE_TIME_ROW, TIME_ROWS);
}

/* (1) The condition rows: the noised latents of the reference at t = 0.999
 * with 1 - t in float32, patchified, equal the fixture's rows bit for bit. */
static void load_condition_rows(const h3_st_header *step_fixture,
                                edit_case *edit) {
    size_t reference_elements = edit->reference_rows * PATCH;
    size_t plane = (size_t)edit->reference_height *
                   (size_t)edit->reference_width;
    float *latents = read_fixture(step_fixture, "reference_latents_0",
                                  H3_DTYPE_F32, LATENT_CHANNELS * plane);
    float *aug_noise = read_fixture(step_fixture, "reference_aug_noise_0",
                                    H3_DTYPE_F32, LATENT_CHANNELS * plane);
    float *noised = malloc(LATENT_CHANNELS * plane * sizeof(*noised));
    edit->condition_rows = malloc(reference_elements * sizeof(float));
    require(noised && edit->condition_rows, "oom");
    h3_reference_scale_noise(latents, aug_noise, H3_REFERENCE_NOISE_AUG,
                             LATENT_CHANNELS * plane, noised);
    require(h3_dit_patchify_video(noised, LATENT_CHANNELS, 1,
                                  edit->reference_height,
                                  edit->reference_width, edit->condition_rows,
                                  reference_elements),
            "cannot patchify the reference latents");
    float *fixture_rows = read_fixture(step_fixture, "condition_rows",
                                       H3_DTYPE_F32, reference_elements);
    require(!memcmp(edit->condition_rows, fixture_rows,
                    reference_elements * sizeof(float)),
            "the condition rows differ from the fixture's: not "
            "t * x + (1.0f - t) * noise in float32");
    printf("condition rows: %zu x %d, bit-identical to the fixture's "
           "scale_noise rows\n", edit->reference_rows, PATCH);
    free(latents); free(aug_noise); free(noised); free(fixture_rows);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        printf("skip: usage %s WEIGHT_DIR EDIT_STEP_FIXTURE "
               "EDIT_TEXT_SMALL_FIXTURE stream|bf16|int8 [EXCHANGE_FILE|- "
               "[TRUTH_FIXTURE]]\n", argv[0]);
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s WEIGHT_DIR EDIT_STEP_FIXTURE "
                "EDIT_TEXT_SMALL_FIXTURE stream|bf16|int8 [EXCHANGE_FILE|- "
                "[TRUTH_FIXTURE]]\n", argv[0]);
        return 2;
    }
    if (access(argv[2], R_OK) != 0 || access(argv[3], R_OK) != 0) {
        printf("skip: fixture %s is not installed\n",
               access(argv[2], R_OK) != 0 ? argv[2] : argv[3]);
        return 0;
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
    edit_case edit;
    memset(&edit, 0, sizeof(edit));
    edit.stream = stream;
    edit.bf16 = bf16;
    edit.tokens = (size_t)prompt->shape[0];
    edit.text.tokens = edit.tokens;
    edit.text.width = TEXT_DIM;
    edit.text.values = read_fixture(&text_fixture, "prompt_embeds",
                                    H3_DTYPE_BF16, edit.tokens * TEXT_DIM);
    uint8_t *tags = read_fixture(&text_fixture, "token_tags", H3_DTYPE_U8,
                                 edit.tokens);
    edit.text.tags = tags;

    edit.latent_height = read_scalar(&step_fixture, "latent_height");
    edit.latent_width = read_scalar(&step_fixture, "latent_width");
    require(read_scalar(&step_fixture, "reference_count") == 1,
            "the step fixture holds one reference");
    require(edit.latent_height > 0 && edit.latent_width > 0 &&
            edit.latent_height % 2 == 0 && edit.latent_width % 2 == 0,
            "bad latent size in the step fixture");
    const h3_st_tensor *reference_tensor = h3_st_find(&step_fixture,
                                                      "reference_latents_0");
    require(reference_tensor && reference_tensor->ndim == 3 &&
            reference_tensor->shape[0] == LATENT_CHANNELS,
            "bad reference_latents_0 fixture");
    edit.reference_height = (int)reference_tensor->shape[1];
    edit.reference_width = (int)reference_tensor->shape[2];
    edit.reference_rows = (size_t)(edit.reference_height / 2) *
                          (size_t)(edit.reference_width / 2);
    edit.rows = (size_t)(edit.latent_height / 2) *
                (size_t)(edit.latent_width / 2);
    edit.sequence = edit.tokens + edit.reference_rows + edit.rows;
    size_t elements = edit.rows * PATCH;
    size_t reference_elements = edit.reference_rows * PATCH;

    load_condition_rows(&step_fixture, &edit);

    int32_t *adaln = read_fixture(&step_fixture, "adaln_indices", H3_DTYPE_I32,
                                  edit.sequence);
    int32_t *timestep_indices = read_fixture(&step_fixture, "timestep_indices",
                                             H3_DTYPE_I32, edit.sequence);
    uint8_t *sequence_tags = read_fixture(&step_fixture, "token_tags",
                                          H3_DTYPE_U8, edit.sequence);
    require(!memcmp(sequence_tags, tags, edit.tokens),
            "the text fixture's tags differ from the step fixture's");
    for (size_t row = edit.tokens; row < edit.sequence; row++)
        require(sequence_tags[row] == 0, "an image row has a nonzero tag");

    if (bf16) check_block0(weight_directory, &step_fixture, edit.sequence);

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
    uint16_t *refined_want = read_fixture(&step_fixture, "refined_text",
                                          H3_DTYPE_BF16, edit.tokens * HIDDEN);
    float *inputs[FIXTURE_STEPS], *wanted[FIXTURE_STEPS];
    float *truth[FIXTURE_STEPS] = {NULL, NULL, NULL, NULL, NULL, NULL};
    float *refined_truth = NULL;
    for (int index = 0; index < FIXTURE_STEPS; index++) {
        inputs[index] = read_fixture(&step_fixture, input_names[index],
                                     H3_DTYPE_F32, elements);
        /* The model returns the reference rows first; the generated rows
         * are the ones h3.c produces. */
        float *whole = read_fixture(&step_fixture, output_names[index],
                                    H3_DTYPE_F32,
                                    reference_elements + elements);
        wanted[index] = malloc(elements * sizeof(float));
        require(wanted[index] != NULL, "oom");
        memcpy(wanted[index], whole + reference_elements,
               elements * sizeof(float));
        free(whole);
    }
    h3_st_header truth_fixture;
    if (truth_path && h3_st_read_header(truth_path, &truth_fixture, error,
                                        sizeof(error))) {
        refined_truth = read_fixture(&truth_fixture, "refined_text_f32",
                                     H3_DTYPE_F32, edit.tokens * HIDDEN);
        for (int index = 0; index < FIXTURE_STEPS; index++)
            truth[index] = read_fixture(&truth_fixture, truth_names[index],
                                        H3_DTYPE_F32, elements);
    } else {
        printf("no F32 truth fixture: truth checks skipped\n");
    }

    uint64_t hash = 1469598103934665603ull;
    hash = hash_bytes(hash, edit.text.values,
                      edit.tokens * TEXT_DIM * sizeof(uint16_t));
    hash = hash_bytes(hash, edit.condition_rows,
                      reference_elements * sizeof(float));
    for (int index = 0; index < FIXTURE_STEPS; index++)
        hash = hash_bytes(hash, inputs[index], elements * sizeof(float));
    exchange_header header = {
        EXCHANGE_MAGIC, stream ? 1u : 0u, (uint32_t)edit.tokens,
        (uint32_t)edit.reference_rows, (uint32_t)edit.latent_height,
        (uint32_t)edit.latent_width, (uint32_t)hash, (uint32_t)(hash >> 32),
        (uint32_t)elements
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

    printf("mode %s: N=%zu text rows, R=%zu reference rows (%dx%d latent), "
           "P=%zu image rows (%dx%d latent), S=%zu\n", mode, edit.tokens,
           edit.reference_rows, edit.reference_height, edit.reference_width,
           edit.rows, edit.latent_height, edit.latent_width, edit.sequence);
    double started = seconds_now();
    h3_dit *dit = load_edit_dit(weight_directory, &edit, edit.condition_rows);
    printf("load: %.1f s\n", seconds_now() - started);
    require(h3_dit_image_rows(dit) == edit.rows, "image row count mismatch");
    check_adaln_mapping(dit, &edit, adaln, timestep_indices, tags);

    uint16_t *refined_got = malloc(edit.tokens * HIDDEN * sizeof(*refined_got));
    float *refined_got_f32 = malloc(edit.tokens * HIDDEN * sizeof(float));
    float *refined_want_f32 = malloc(edit.tokens * HIDDEN * sizeof(float));
    require(refined_got && refined_got_f32 && refined_want_f32, "oom");
    require(h3_dit_read_refined_text(dit, refined_got, edit.tokens * HIDDEN),
            "cannot read the refined text");
    for (size_t index = 0; index < edit.tokens * HIDDEN; index++) {
        refined_got_f32[index] = bf16_value(refined_got[index]);
        refined_want_f32[index] = bf16_value(refined_want[index]);
    }
    comparison refined = compare(refined_got_f32, refined_want_f32,
                                 edit.tokens, HIDDEN);
    printf("refined text vs reference: rel-L2 %.6g, max abs %.6g, median row "
           "%.6g\n", refined.l2, refined.maximum, refined.row_median);
    require(refined.l2 < REFINED_TEXT_BOUND, "refined text is off");
    if (refined_truth) {
        comparison against = compare(refined_got_f32, refined_truth,
                                     edit.tokens, HIDDEN);
        printf("refined text vs F32 truth: rel-L2 %.6g, median row %.6g\n",
               against.l2, against.row_median);
        require(against.l2 < REFINED_TRUTH_BOUND,
                "refined text is far from the F32 truth");
    }

    /* (3) Six teacher-forced steps: the velocity of the generated rows. */
    float *velocity[FIXTURE_STEPS];
    float *hidden = malloc(elements / PATCH * HIDDEN * sizeof(*hidden));
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
        comparison result = compare(velocity[index], wanted[index], edit.rows,
                                    PATCH);
        printf("step %d vs reference: rel-L2 %.6g, max abs %.6g, median row "
               "%.6g, q99 row %.6g, worst row %.6g (%.2f s)\n", steps[index],
               result.l2, result.maximum, result.row_median, result.row_q99,
               result.row_worst, elapsed);
        require(result.l2 < (int8 ? INT8_VELOCITY_BOUND[index] :
                                    BF16_VELOCITY_BOUND[index]),
                "velocity rel-L2 over the bound");
        require(result.row_median < (int8 ? INT8_ROW_MEDIAN_BOUND[index] :
                                            BF16_ROW_MEDIAN_BOUND[index]),
                "velocity median row error over the bound");
        require(result.row_q99 < ROW_Q99_BOUND,
                "velocity q99 row error over the bound");
        if (truth[index]) {
            comparison against = compare(velocity[index], truth[index],
                                         edit.rows, PATCH);
            printf("step %d vs F32 truth: rel-L2 %.6g, max abs %.6g, median "
                   "row %.6g, q99 row %.6g, worst row %.6g\n", steps[index],
                   against.l2, against.maximum, against.row_median,
                   against.row_q99, against.row_worst);
            require(against.l2 < (int8 ? INT8_TRUTH_BOUND[index] :
                                         BF16_TRUTH_BOUND[index]),
                    "velocity rel-L2 against the F32 truth over the bound");
            require(against.row_median < (int8 ?
                        INT8_TRUTH_MEDIAN_BOUND[index] :
                        BF16_TRUTH_MEDIAN_BOUND[index]),
                    "velocity median row error against the truth over the "
                    "bound");
            require(against.row_q99 < ROW_Q99_BOUND,
                    "velocity q99 row error against the truth over the "
                    "bound");
        }
        /* (4) Head binding on the generated rows' final hidden state. */
        require(h3_dit_read_final_hidden(dit, hidden, edit.rows * HIDDEN),
                "cannot read the final hidden state");
        for (unsigned head = 0; head < H3_IMAGE_STEPS; head++) {
            double distance = head_distance(hidden, head_weight[head],
                                            head_bias[head], velocity[index],
                                            edit.rows);
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
     * bit: the reference rows are written once and nothing carries over. */
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

    h3_gpu_stats stats;
    if (h3_dit_get_gpu_stats(dit, &stats))
        printf("GPU buffers: peak live %.2f GiB, %.3f GPU seconds\n",
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               stats.gpu_seconds);
    h3_dit_free(dit);
    if (exchange && !int8) check_exchange(exchange, &header, velocity);

    /* (5) Condition binding: with the reference rows zeroed the step-0
     * velocity must move, so the rows are in the sequence and attended. */
    float *zero_rows = calloc(reference_elements, sizeof(float));
    require(zero_rows != NULL, "oom");
    h3_dit *zeroed = load_edit_dit(weight_directory, &edit, zero_rows);
    require(h3_dit_forward_image(zeroed, steps[0], inputs[0], again, error,
                                 sizeof(error)), error);
    comparison moved = compare(again, velocity[0], edit.rows, PATCH);
    printf("step 0 with zeroed reference rows vs with them: rel-L2 %.6g "
           "(must exceed %g)\n", moved.l2, CONDITION_MOVE_MIN);
    require(moved.l2 > CONDITION_MOVE_MIN,
            "zeroing the reference rows did not move the velocity: the "
            "condition rows do not reach the attention");
    h3_dit_free(zeroed);
    printf("ok: image edit DiT %s matches the diffusers reference\n", mode);
    return 0;
}
