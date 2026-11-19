/* tests/test_mps_operands.c
 *
 * The engine can run its MPSGraph BF16 graphs with FP16 operands when the
 * device emulates the BF16 matrix kernels in software: BF16 values are
 * exactly representable in FP16, so the product and its FP32 accumulation
 * are unchanged while the kernel is faster.
 *
 * This test builds one Metal context that keeps BF16 operands and one that
 * uses FP16 operands, runs the same BF16 data through both, and compares
 * the results. The linear path must be bit identical, because both graphs
 * accumulate in FP32. The attention path may differ by the rounding of the
 * attention weights, so it is checked against a relative bound.
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MPSGraph requires every operand buffer to be at least 512 KiB, so the
 * shapes below are the smallest ones that exercise the real paths. */
enum {
    ROWS = 256,
    K_DIM = 1024,
    N_DIM = 1024,
    SEQ = 256,
    HEADS = 8,
    HEAD_DIM = 128
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_mps_operands.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exponent = (int32_t)((bits >> 23) & 0xff);
    uint32_t mantissa = bits & 0x7fffff;
    uint32_t rounded = mantissa >> 16;
    uint32_t rest = mantissa & 0xffff;
    if (rest > 0x8000 || (rest == 0x8000 && (rounded & 1))) rounded++;
    if (rounded == 0x80) {
        rounded = 0;
        exponent++;
    }
    if (exponent == 0xff)
        return (uint16_t)((sign << 15) | (0xff << 7) | (rounded & 0x7f));
    return (uint16_t)((sign << 15) | ((uint32_t)exponent << 7) |
                       (rounded & 0x7f));
}

static float bf16_to_float(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

static uint32_t random_state = 12345;

static float random_value(void) {
    random_state = random_state * 1103515245u + 12345u;
    return ((float)((random_state >> 9) & 0xffff) / 32768.0f - 1.0f);
}

/* MPSGraph refuses buffers that are too small for the temporaries of the
 * graph it encodes, so every tensor is allocated with a generous size.
 * The engine itself never hits this limit: its tensors are megabytes. */
#define MIN_TENSOR_BYTES (8 * 1024 * 1024)

static h3_gpu_tensor *allocate_bf16(h3_gpu *gpu, size_t elements) {
    size_t minimum = MIN_TENSOR_BYTES / sizeof(uint16_t);
    if (elements < minimum) elements = minimum;
    return h3_gpu_tensor_new_bf16(gpu, elements);
}

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *input;
    h3_gpu_tensor *weight;
    h3_gpu_tensor *output;
} operands;

static operands make_operands(const char *mode, uint16_t *input_values,
                              uint16_t *weight_values) {
    operands result;
    char error[256];
    error[0] = 0;
    setenv("H3_FP16_OPERANDS", mode, 1);
    result.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!result.gpu) fail(error[0] ? error : "cannot create the Metal context");
    size_t input_count = (size_t)ROWS * K_DIM;
    size_t weight_count = (size_t)N_DIM * K_DIM;
    result.input = h3_gpu_tensor_new_bf16(result.gpu,
        input_count > MIN_TENSOR_BYTES / sizeof(uint16_t) ?
            input_count : MIN_TENSOR_BYTES / sizeof(uint16_t));
    result.weight = h3_gpu_tensor_new_bf16(result.gpu,
        weight_count > MIN_TENSOR_BYTES / sizeof(uint16_t) ?
            weight_count : MIN_TENSOR_BYTES / sizeof(uint16_t));
    result.output = allocate_bf16(result.gpu, (size_t)ROWS * N_DIM);
    if (!result.input || !result.weight || !result.output)
        fail("cannot allocate the Metal tensors");
    require(h3_gpu_tensor_write_bf16(result.input, input_values, input_count),
            "cannot write the linear input");
    require(h3_gpu_tensor_write_bf16(result.weight, weight_values, weight_count),
            "cannot write the linear weight");
    return result;
}

static void free_operands(operands *value) {
    h3_gpu_tensor_free(value->input);
    h3_gpu_tensor_free(value->weight);
    h3_gpu_tensor_free(value->output);
    h3_gpu_free(value->gpu);
}

static void run_linear(operands *value) {
    if (!h3_gpu_begin(value->gpu)) fail("cannot begin the command stream");
    if (!h3_gpu_linear_bf16(value->gpu, value->output, value->input,
                            value->weight, NULL, ROWS, K_DIM, N_DIM))
        fail(h3_gpu_error(value->gpu));
    if (!h3_gpu_submit(value->gpu)) fail("cannot submit the command stream");
}

static uint16_t *run_sdpa(operands *value, uint16_t *values) {
    size_t count = (size_t)SEQ * HEADS * HEAD_DIM;
    h3_gpu_tensor *q = allocate_bf16(value->gpu, count);
    h3_gpu_tensor *k = allocate_bf16(value->gpu, count);
    h3_gpu_tensor *v = allocate_bf16(value->gpu, count);
    h3_gpu_tensor *out = allocate_bf16(value->gpu, count);
    require(h3_gpu_tensor_write_bf16(q, values, count),
            "cannot write the attention query");
    require(h3_gpu_tensor_write_bf16(k, values, count),
            "cannot write the attention key");
    require(h3_gpu_tensor_write_bf16(v, values, count),
            "cannot write the attention value");
    if (!q || !k || !v || !out) fail("cannot allocate the attention tensors");
    if (!h3_gpu_begin(value->gpu)) fail("cannot begin the command stream");
    if (!h3_gpu_sdpa_bf16(value->gpu, out, q, k, v, SEQ, HEADS, HEAD_DIM,
                          1.0f / sqrtf((float)HEAD_DIM)))
        fail(h3_gpu_error(value->gpu));
    if (!h3_gpu_submit(value->gpu)) fail("cannot submit the command stream");
    uint16_t *result = malloc(count * sizeof(uint16_t));
    if (!result) fail("out of memory");
    require(h3_gpu_tensor_read_bf16(out, result, count),
            "cannot read the attention result");
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(out);
    return result;
}

int main(void) {
    size_t input_count = (size_t)ROWS * K_DIM;
    size_t weight_count = (size_t)N_DIM * K_DIM;
    uint16_t *input_values = malloc(input_count * sizeof(uint16_t));
    uint16_t *weight_values = malloc(weight_count * sizeof(uint16_t));
    size_t attention_count = (size_t)SEQ * HEADS * HEAD_DIM;
    uint16_t *attention_values = malloc(attention_count * sizeof(uint16_t));
    if (!input_values || !weight_values || !attention_values)
        fail("out of memory");
    for (size_t i = 0; i < input_count; i++)
        input_values[i] = float_to_bf16(random_value() * 0.5f);
    for (size_t i = 0; i < weight_count; i++)
        weight_values[i] = float_to_bf16(random_value() * 0.5f);
    for (size_t i = 0; i < attention_count; i++)
        attention_values[i] = float_to_bf16(random_value() * 0.5f);

    operands bf16 = make_operands("0", input_values, weight_values);
    operands fp16 = make_operands("1", input_values, weight_values);
    require(h3_gpu_uses_fp16_linear(bf16.gpu) == 0,
            "the BF16 context selected FP16 operands");
    require(h3_gpu_uses_fp16_linear(fp16.gpu) == 1,
            "the FP16 context selected BF16 operands");

    run_linear(&bf16);
    run_linear(&fp16);
    size_t output_count = (size_t)ROWS * N_DIM;
    uint16_t *bf16_output = malloc(output_count * sizeof(uint16_t));
    uint16_t *fp16_output = malloc(output_count * sizeof(uint16_t));
    if (!bf16_output || !fp16_output) fail("out of memory");
    require(h3_gpu_tensor_read_bf16(bf16.output, bf16_output, output_count),
            "cannot read the BF16 linear result");
    require(h3_gpu_tensor_read_bf16(fp16.output, fp16_output, output_count),
            "cannot read the FP16 linear result");
    size_t differences = 0;
    double worst = 0;
    for (size_t i = 0; i < output_count; i++) {
        if (bf16_output[i] != fp16_output[i]) differences++;
        double delta = fabs(bf16_to_float(bf16_output[i]) -
                            bf16_to_float(fp16_output[i]));
        if (delta > worst) worst = delta;
    }
    printf("linear: %zu of %zu results differ, max delta %.3e\n",
           differences, output_count, worst);
    require(differences == 0,
            "the FP16 operand linear path is not bit identical");

    /* The attention weights are rounded to FP16 before the kernel runs, so
     * the two paths agree to the precision of the format instead of bit
     * for bit. */
    uint16_t *bf16_attention = run_sdpa(&bf16, attention_values);
    uint16_t *fp16_attention = run_sdpa(&fp16, attention_values);
    size_t attention_output_count = (size_t)SEQ * HEADS * HEAD_DIM;
    double energy = 0, error = 0;
    for (size_t i = 0; i < attention_output_count; i++) {
        double reference = bf16_to_float(bf16_attention[i]);
        double delta = reference - bf16_to_float(fp16_attention[i]);
        energy += reference * reference;
        error += delta * delta;
    }
    double relative = energy > 0 ? sqrt(error / energy) : 0;
    printf("attention: relative L2 difference %.3e\n", relative);
    require(relative < 0.01, "the FP16 operand attention path diverged");
    free(bf16_attention);
    free(fp16_attention);

    free(bf16_output);
    free(fp16_output);
    free_operands(&bf16);
    free_operands(&fp16);
    free(input_values);
    free(weight_values);
    free(attention_values);
    printf("ok: the MPSGraph FP16 operand path matches the BF16 path\n");
    return 0;
}
