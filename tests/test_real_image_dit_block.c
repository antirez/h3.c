#include "h3_dit_names.h"
#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_image.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376, HEADS = 56, HEAD_DIM = 128, INNER = HEADS * HEAD_DIM,
    FFN = 14336, TIME_INPUT = 256, TIME_DIM = 2688, ROPE_FREQS = 16,
    ROPE_HALF = 48, MODALITIES = 3, SLOTS = 6, MAX_TENSORS = 64
};

/* Median over rows of the per-row relative L2 against the F32 run of the
 * block. The BF16 reference is 9.8e-3 from that run as a whole; the Metal
 * output is closer to it than the reference is. A row-wise median ignores the
 * few rows that carry the 1e5 massive activations. */
#define BOUND 6.5e-3

static h3_gpu *gpu;
static h3_gpu_tensor *owned[MAX_TENSORS];
static size_t owned_count;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_dit_block.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static h3_gpu_tensor *own(h3_gpu_tensor *tensor, const char *what) {
    if (!tensor) {
        fprintf(stderr, "FAIL tests/test_real_image_dit_block.c: %s: %s\n",
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
    fprintf(stderr, "FAIL tests/test_real_image_dit_block.c: %s: %s\n",
            operation, h3_gpu_error(gpu));
    exit(1);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_dit_block.c: fixture "
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

static h3_gpu_tensor *load_fc1(const h3_weight_store *weights,
                               const h3_dit_names *names, int gate_first) {
    char name[192], error[512];
    weight_name(name, sizeof(name), names, names->fc1);
    h3_gpu_tensor *tensor = h3_weight_load_bf16_halves(
        weights, gpu, name, FFN * 2, HIDDEN, !gate_first, error,
        sizeof(error));
    if (!tensor) die(error);
    return own(tensor, name);
}

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
    result.fc1 = NULL;
    return result;
}

static void run_block(const block_weights *weight, h3_gpu_tensor *adaln_input,
                      h3_gpu_tensor *hidden, h3_gpu_tensor *row_map,
                      h3_gpu_tensor *rope_cos, h3_gpu_tensor *rope_sin,
                      uint32_t sequence) {
    size_t rows = sequence;
    h3_gpu_tensor *modulation = new_bf16(MODALITIES * SLOTS * HIDDEN);
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
                                weight->adaln_weight, weight->adaln_bias, 1,
                                TIME_DIM, MODALITIES * SLOTS * HIDDEN),
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

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

typedef struct {
    double l2, maximum, row_median;
} comparison;

static comparison compare(h3_gpu_tensor *got_tensor, const uint16_t *want,
                          const float *want_f32, size_t rows) {
    size_t elements = rows * HIDDEN;
    uint16_t *got = malloc(elements * sizeof(*got));
    double *row_errors = malloc(rows * sizeof(*row_errors));
    require(got && row_errors, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read the block output");
    comparison result = {0.0, 0.0, 0.0};
    double square_error = 0.0, square_value = 0.0;
    for (size_t row = 0; row < rows; row++) {
        double row_error = 0.0, row_value = 0.0;
        for (size_t index = row * HIDDEN; index < (row + 1) * HIDDEN;
             index++) {
            double expected = bf16_value(want[index]);
            double delta = (double)bf16_value(got[index]) - expected;
            if (fabs(delta) > result.maximum) result.maximum = fabs(delta);
            square_error += delta * delta;
            square_value += expected * expected;
            double truth = want_f32[index];
            double truth_delta = (double)bf16_value(got[index]) - truth;
            row_error += truth_delta * truth_delta;
            row_value += truth * truth;
        }
        row_errors[row] = sqrt(row_error / (row_value > 1e-24 ? row_value :
                                                                1e-24));
    }
    qsort(row_errors, rows, sizeof(*row_errors), compare_doubles);
    result.row_median = row_errors[rows / 2];
    result.l2 = sqrt(square_error /
                     (square_value > 1e-24 ? square_value : 1e-24));
    free(got);
    free(row_errors);
    return result;
}

static uint16_t *read_source(const h3_weight_store *weights, const char *name,
                             size_t elements) {
    char error[512];
    const h3_st_header *header;
    const h3_st_tensor *tensor = h3_weight_find(weights, name, &header);
    require(tensor && tensor->dtype == H3_DTYPE_BF16 &&
            h3_st_tensor_elements(tensor) == elements, "bad source tensor");
    uint16_t *data = malloc(elements * sizeof(*data));
    if (!data || !h3_st_read_data(header, tensor, data,
                                  elements * sizeof(*data), error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static uint16_t *read_gpu(h3_gpu_tensor *tensor, size_t elements) {
    uint16_t *data = malloc(elements * sizeof(*data));
    require(data && h3_gpu_tensor_read_bf16(tensor, data, elements),
            "cannot read back a weight");
    return data;
}

/* Row h * 384 + p * 128 + d of the grouped matrix is row h * 128 + d of
 * to_{q,k,v}[p], bit for bit. */
static void check_qkv_layout(const h3_weight_store *weights,
                             const h3_dit_names *names, h3_gpu_tensor *qkv) {
    const char *members[3] = {names->q, names->k, names->v};
    size_t matrix = (size_t)INNER * HIDDEN;
    uint16_t *grouped = read_gpu(qkv, 3 * matrix);
    for (int projection = 0; projection < 3; projection++) {
        char name[192];
        weight_name(name, sizeof(name), names, members[projection]);
        uint16_t *source = read_source(weights, name, matrix);
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t d = 0; d < HEAD_DIM; d++) {
                size_t destination = head * 384 + (size_t)projection * 128 + d;
                require(memcmp(grouped + destination * HIDDEN,
                               source + (head * 128 + d) * HIDDEN,
                               HIDDEN * sizeof(*source)) == 0,
                        "grouped QKV row differs from its source row");
            }
        }
        free(source);
    }
    free(grouped);
}

static void check_fc1_layout(const h3_weight_store *weights,
                             const h3_dit_names *names, h3_gpu_tensor *fc1,
                             int gate_first) {
    char name[192];
    weight_name(name, sizeof(name), names, names->fc1);
    size_t elements = (size_t)FFN * 2 * HIDDEN;
    uint16_t *source = read_source(weights, name, elements);
    uint16_t *loaded = read_gpu(fc1, elements);
    for (size_t row = 0; row < (size_t)FFN * 2; row++) {
        size_t from = gate_first ? row : (row + FFN) % (2 * (size_t)FFN);
        require(memcmp(loaded + row * HIDDEN, source + from * HIDDEN,
                       HIDDEN * sizeof(*source)) == 0,
                "FC1 row differs from its source row");
    }
    free(source);
    free(loaded);
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

int main(int argc, char **argv) {
    const char *weight_directory = argc > 1 ? argv[1] :
        "MiniMax-H3-Image/transformer_diffusers";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_image_dit_block0.safetensors";
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *position_tensor = h3_st_find(&fixture, "position_ids");
    require(position_tensor && position_tensor->ndim == 2 &&
            position_tensor->shape[1] == 3, "bad position_ids fixture");
    size_t sequence = (size_t)position_tensor->shape[0];
    float *positions = read_fixture(&fixture, "position_ids", H3_DTYPE_F32,
                                    sequence * 3);
    int32_t *indices = read_fixture(&fixture, "adaln_indices", H3_DTYPE_I32,
                                    sequence);
    float *temb = read_fixture(&fixture, "temb", H3_DTYPE_F32, TIME_DIM);
    uint16_t *want = read_fixture(&fixture, "hidden_out", H3_DTYPE_BF16,
                                  sequence * HIDDEN);
    float *want_f32 = read_fixture(&fixture, "hidden_out_f32", H3_DTYPE_F32,
                                   sequence * HIDDEN);
    int32_t step = read_scalar(&fixture, "step");
    require(step >= 0 && step < H3_IMAGE_STEPS, "fixture step out of range");

    h3_weight_store *weights = h3_weight_store_open(weight_directory, error,
                                                    sizeof(error));
    if (!weights) die(error);
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    const h3_dit_names *names = h3_dit_names_for(H3_DIT_FAMILY_IMAGE);

    h3_sigma_schedule schedule;
    require(h3_image_schedule_build(&schedule), "cannot build the schedule");
    float features[TIME_INPUT];
    h3_dit_time_features(1.0f - schedule.video[step], features);
    h3_gpu_tensor *computed_time = own(h3_dit_time_embed(
        weights, names, gpu, 1, features, error, sizeof(error)), error);
    double time_l2 = compare_f32(computed_time, temb, TIME_DIM);
    printf("timestep embedding of step %d: rel-L2 %.6g\n", step, time_l2);
    require(time_l2 < 1e-4, "timestep embedding differs from the reference");
    release(&computed_time);

    h3_gpu_tensor *time = own(h3_gpu_tensor_from_f32(gpu, temb, TIME_DIM),
                              "temb upload");
    h3_gpu_tensor *adaln_input = own(h3_dit_adaln_input(
        gpu, time, 1, H3_DIT_ADALN_SILU_THEN_CAST, error, sizeof(error)),
        error);

    uint32_t *rows = malloc(sequence * sizeof(*rows));
    float *cosines = malloc(sequence * ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(sequence * ROPE_HALF * sizeof(*sines));
    require(rows && cosines && sines, "host allocation failed");
    float inverse[ROPE_FREQS];
    h3_image_dit_inv_freq(inverse);
    for (size_t row = 0; row < sequence; row++) {
        require(indices[row] >= 0 && indices[row] < MODALITIES,
                "unexpected AdaLN row index");
        rows[row] = (uint32_t)indices[row];
        for (size_t axis = 0; axis < 3; axis++) {
            for (size_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = row * ROPE_HALF + axis * ROPE_FREQS + frequency;
                float angle = positions[row * 3 + axis] * inverse[frequency];
                cosines[index] = cosf(angle);
                sines[index] = sinf(angle);
            }
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
    check_qkv_layout(weights, names, block.qkv);
    uint16_t *input = read_fixture(&fixture, "hidden_in", H3_DTYPE_BF16,
                                   sequence * HIDDEN);
    comparison result[2];
    for (int gate_first = 1; gate_first >= 0; gate_first--) {
        block.fc1 = load_fc1(weights, names, gate_first);
        check_fc1_layout(weights, names, block.fc1, gate_first);
        h3_gpu_tensor *hidden = own(h3_gpu_tensor_from_bf16(
            gpu, input, sequence * HIDDEN), "hidden upload");
        run_block(&block, adaln_input, hidden, row_map, rope_cos, rope_sin,
                  (uint32_t)sequence);
        result[gate_first] = compare(hidden, want, want_f32, sequence);
        printf("fc1_gate_first=%d: median row rel-L2 vs F32 %.6g; vs BF16 "
               "fixture rel-L2 %.6g, max abs %.6g\n", gate_first,
               result[gate_first].row_median, result[gate_first].l2,
               result[gate_first].maximum);
        release(&hidden);
        release(&block.fc1);
    }
    int passes[2] = {result[0].row_median < BOUND,
                     result[1].row_median < BOUND};
    require(passes[0] + passes[1] == 1,
            "exactly one FC1 half order must pass the bound");
    int winner = passes[1];
    printf("FC1 half order that matches the reference: fc1_gate_first=%d\n",
           winner);
    require(winner == H3_IMAGE_FC1_GATE_FIRST,
            "the image loader default disagrees with the passing FC1 order");
    printf("ok: block 0 matches the diffusers reference (median row rel-L2 %.6g < %g)\n",
           result[winner].row_median, BOUND);

    free(input);
    free(rows);
    free(cosines);
    free(sines);
    free(positions);
    free(indices);
    free(temb);
    free(want);
    free(want_f32);
    while (owned_count) h3_gpu_tensor_free(owned[--owned_count]);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    h3_st_free_header(&fixture);
    return 0;
}
