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

/* The bias is added inside the same graph, so it has to follow the operand
 * format of the product: MPSGraph rejects an addition whose operands differ.
 * The DiT passes no bias, but the video VAE decoder's MLP does. */
static uint16_t *run_linear_biased(operands *value, uint16_t *input_values,
                                   uint16_t *weight_values,
                                   uint16_t *bias_values) {
    size_t input_count = (size_t)ROWS * K_DIM;
    size_t weight_count = (size_t)N_DIM * K_DIM;
    size_t output_count = (size_t)ROWS * N_DIM;
    h3_gpu_tensor *input = allocate_bf16(value->gpu, input_count);
    h3_gpu_tensor *weight = allocate_bf16(value->gpu, weight_count);
    h3_gpu_tensor *bias = allocate_bf16(value->gpu, N_DIM);
    h3_gpu_tensor *output = allocate_bf16(value->gpu, output_count);
    if (!input || !weight || !bias || !output)
        fail("cannot allocate the biased linear tensors");
    require(h3_gpu_tensor_write_bf16(input, input_values, input_count),
            "cannot write the biased linear input");
    require(h3_gpu_tensor_write_bf16(weight, weight_values, weight_count),
            "cannot write the biased linear weight");
    require(h3_gpu_tensor_write_bf16(bias, bias_values, N_DIM),
            "cannot write the biased linear bias");
    if (!h3_gpu_begin(value->gpu)) fail("cannot begin the command stream");
    if (!h3_gpu_linear_bf16(value->gpu, output, input, weight, bias,
                            ROWS, K_DIM, N_DIM))
        fail(h3_gpu_error(value->gpu));
    if (!h3_gpu_submit(value->gpu)) fail("cannot submit the command stream");
    uint16_t *result = malloc(output_count * sizeof(uint16_t));
    if (!result) fail("out of memory");
    require(h3_gpu_tensor_read_bf16(output, result, output_count),
            "cannot read the biased linear result");
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias); h3_gpu_tensor_free(output);
    return result;
}

/* The fused fc1 -> SwiGLU -> fc2 graph. Its intermediate is cast to the
 * operand format before the second product, so the two paths are compared
 * rather than assumed identical. */
static uint16_t *run_mlp(operands *value, uint16_t *input_values,
                         uint16_t *fc1_values, uint16_t *fc2_values) {
    enum { MLP_ROWS = 128, MLP_IN = 512, MLP_HIDDEN = 256, MLP_OUT = 512 };
    size_t input_count = (size_t)MLP_ROWS * MLP_IN;
    size_t fc1_count = (size_t)MLP_HIDDEN * 2 * MLP_IN;
    size_t fc2_count = (size_t)MLP_OUT * MLP_HIDDEN;
    size_t output_count = (size_t)MLP_ROWS * MLP_OUT;
    h3_gpu_tensor *input = allocate_bf16(value->gpu, input_count);
    h3_gpu_tensor *fc1 = allocate_bf16(value->gpu, fc1_count);
    h3_gpu_tensor *fc2 = allocate_bf16(value->gpu, fc2_count);
    h3_gpu_tensor *output = allocate_bf16(value->gpu, output_count);
    if (!input || !fc1 || !fc2 || !output) fail("cannot allocate the MLP tensors");
    require(h3_gpu_tensor_write_bf16(input, input_values, input_count),
            "cannot write the MLP input");
    require(h3_gpu_tensor_write_bf16(fc1, fc1_values, fc1_count),
            "cannot write the MLP fc1 weight");
    require(h3_gpu_tensor_write_bf16(fc2, fc2_values, fc2_count),
            "cannot write the MLP fc2 weight");
    if (!h3_gpu_begin(value->gpu)) fail("cannot begin the command stream");
    if (!h3_gpu_mlp_bf16(value->gpu, output, input, fc1, fc2, MLP_ROWS,
                         MLP_IN, MLP_HIDDEN, MLP_OUT))
        fail(h3_gpu_error(value->gpu));
    if (!h3_gpu_submit(value->gpu)) fail("cannot submit the command stream");
    uint16_t *result = malloc(output_count * sizeof(uint16_t));
    if (!result) fail("out of memory");
    require(h3_gpu_tensor_read_bf16(output, result, output_count),
            "cannot read the MLP result");
    h3_gpu_tensor_free(input); h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(fc2); h3_gpu_tensor_free(output);
    return result;
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

    /* The same comparison with a bias, which the graph adds to the product
     * and which therefore has to follow the same operand format. */
    uint16_t *bias_values = malloc(N_DIM * sizeof(uint16_t));
    if (!bias_values) fail("out of memory");
    for (size_t i = 0; i < N_DIM; i++)
        bias_values[i] = float_to_bf16(random_value() * 0.5f);
    uint16_t *bf16_biased = run_linear_biased(&bf16, input_values,
                                              weight_values, bias_values);
    uint16_t *fp16_biased = run_linear_biased(&fp16, input_values,
                                              weight_values, bias_values);
    size_t biased_differences = 0;
    double biased_worst = 0;
    for (size_t i = 0; i < output_count; i++) {
        if (bf16_biased[i] != fp16_biased[i]) biased_differences++;
        double delta = fabs(bf16_to_float(bf16_biased[i]) -
                            bf16_to_float(fp16_biased[i]));
        if (delta > biased_worst) biased_worst = delta;
    }
    printf("linear with bias: %zu of %zu results differ, max delta %.3e\n",
           biased_differences, output_count, biased_worst);
    require(biased_differences == 0,
            "the FP16 operand biased linear path is not bit identical");
    free(bf16_biased);
    free(fp16_biased);
    free(bias_values);

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

    /* The same comparison against a double precision attention reference
     * shows that the spread above is the rounding of the BF16 path rather
     * than error introduced by the FP16 one. */
    {
        double bf16_sq = 0, fp16_sq = 0, reference_sq = 0;
        double scale = 1.0 / sqrt((double)HEAD_DIM);
        double *scores = malloc((size_t)SEQ * sizeof(double));
        if (!scores) fail("out of memory");
        for (int head = 0; head < HEADS; head++) {
            for (int row = 0; row < SEQ; row++) {
                double maximum = -1e30, total = 0;
                for (int column = 0; column < SEQ; column++) {
                    double dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++)
                        dot += bf16_to_float(attention_values[
                                   ((size_t)row * HEADS + head) * HEAD_DIM + d]) *
                               bf16_to_float(attention_values[
                                   ((size_t)column * HEADS + head) * HEAD_DIM + d]);
                    scores[column] = dot * scale;
                    if (scores[column] > maximum) maximum = scores[column];
                }
                for (int column = 0; column < SEQ; column++) {
                    scores[column] = exp(scores[column] - maximum);
                    total += scores[column];
                }
                for (int d = 0; d < HEAD_DIM; d++) {
                    double expected = 0;
                    for (int column = 0; column < SEQ; column++)
                        expected += (scores[column] / total) *
                            bf16_to_float(attention_values[
                                ((size_t)column * HEADS + head) * HEAD_DIM + d]);
                    double b = bf16_to_float(bf16_attention[
                        ((size_t)row * HEADS + head) * HEAD_DIM + d]);
                    double f = bf16_to_float(fp16_attention[
                        ((size_t)row * HEADS + head) * HEAD_DIM + d]);
                    reference_sq += expected * expected;
                    bf16_sq += (b - expected) * (b - expected);
                    fp16_sq += (f - expected) * (f - expected);
                }
            }
        }
        free(scores);
        double bf16_relative = sqrt(bf16_sq / reference_sq);
        double fp16_relative = sqrt(fp16_sq / reference_sq);
        printf("attention vs double reference: BF16 %.3e, FP16 %.3e\n",
               bf16_relative, fp16_relative);
        require(fp16_relative <= bf16_relative * 1.05,
                "the FP16 operand attention path is less accurate than the "
                "BF16 one");
    }
    free(bf16_attention);
    free(fp16_attention);

    /* The MLP intermediate is cast to the operand format before the second
     * product, which keeps more significand bits than the BF16 cast. */
    /* Both contexts must see the same MLP weights, so they are generated
     * once here instead of inside the run. */
    enum { MLP_ROWS = 128, MLP_IN = 512, MLP_HIDDEN = 256, MLP_OUT = 512 };
    size_t mlp_input_count = (size_t)MLP_ROWS * MLP_IN;
    size_t mlp_fc1_count = (size_t)MLP_HIDDEN * 2 * MLP_IN;
    size_t mlp_fc2_count = (size_t)MLP_OUT * MLP_HIDDEN;
    uint16_t *mlp_input = malloc(mlp_input_count * sizeof(uint16_t));
    uint16_t *mlp_fc1 = malloc(mlp_fc1_count * sizeof(uint16_t));
    uint16_t *mlp_fc2 = malloc(mlp_fc2_count * sizeof(uint16_t));
    if (!mlp_input || !mlp_fc1 || !mlp_fc2) fail("out of memory");
    for (size_t i = 0; i < mlp_input_count; i++)
        mlp_input[i] = float_to_bf16(random_value() * 0.25f);
    for (size_t i = 0; i < mlp_fc1_count; i++)
        mlp_fc1[i] = float_to_bf16(random_value() * 0.05f);
    for (size_t i = 0; i < mlp_fc2_count; i++)
        mlp_fc2[i] = float_to_bf16(random_value() * 0.05f);
    uint16_t *bf16_mlp = run_mlp(&bf16, mlp_input, mlp_fc1, mlp_fc2);
    uint16_t *fp16_mlp = run_mlp(&fp16, mlp_input, mlp_fc1, mlp_fc2);
    /* The fused MLP keeps BF16 operands in both contexts on purpose: its
     * SwiGLU intermediate is the product of two unbounded projections and on
     * the released checkpoint it reaches values two orders of magnitude
     * above FP16's +-65504 range. The two contexts must therefore agree bit
     * for bit here, which is what pins the MLP to BF16. */
    size_t mlp_count = (size_t)MLP_ROWS * MLP_OUT;
    size_t mlp_differences = 0;
    double mlp_worst = 0;
    for (size_t i = 0; i < mlp_count; i++) {
        if (bf16_mlp[i] != fp16_mlp[i]) mlp_differences++;
        double delta = fabs(bf16_to_float(bf16_mlp[i]) -
                            bf16_to_float(fp16_mlp[i]));
        if (delta > mlp_worst) mlp_worst = delta;
    }
    printf("mlp: %zu of %zu results differ, max delta %.3e\n",
           mlp_differences, mlp_count, mlp_worst);
    require(mlp_differences == 0,
            "the MLP followed the operand selection instead of staying BF16");

    /* The MLP is compared against a double precision reference of the same
     * inputs, so a regression in the BF16 graph itself cannot pass either. */
    {
        double *activated = malloc((size_t)MLP_ROWS * MLP_HIDDEN *
                                   sizeof(double));
        if (!activated) fail("out of memory");
        for (int r = 0; r < MLP_ROWS; r++) {
            for (int h = 0; h < MLP_HIDDEN; h++) {
                double gate = 0, up = 0;
                for (int i = 0; i < MLP_IN; i++) {
                    double input_value =
                        bf16_to_float(mlp_input[(size_t)r * MLP_IN + i]);
                    gate += input_value *
                        bf16_to_float(mlp_fc1[(size_t)h * MLP_IN + i]);
                    up += input_value *
                        bf16_to_float(mlp_fc1[
                            (size_t)(h + MLP_HIDDEN) * MLP_IN + i]);
                }
                double silu = gate / (1.0 + exp(-gate));
                activated[(size_t)r * MLP_HIDDEN + h] = silu * up;
            }
        }
        double reference_energy = 0, bf16_error = 0;
        for (int r = 0; r < MLP_ROWS; r++) {
            for (int o = 0; o < MLP_OUT; o++) {
                double expected = 0;
                for (int h = 0; h < MLP_HIDDEN; h++)
                    expected += activated[(size_t)r * MLP_HIDDEN + h] *
                        bf16_to_float(mlp_fc2[(size_t)o * MLP_HIDDEN + h]);
                double b = bf16_to_float(bf16_mlp[(size_t)r * MLP_OUT + o]);
                reference_energy += expected * expected;
                bf16_error += (b - expected) * (b - expected);
            }
        }
        double bf16_relative = sqrt(bf16_error / reference_energy);
        printf("mlp vs double reference: BF16 %.3e\n", bf16_relative);
        require(bf16_relative < 0.01, "the MLP diverged from the reference");
        free(activated);
    }
    free(bf16_mlp);
    free(fp16_mlp);
    free(mlp_input);
    free(mlp_fc1);
    free(mlp_fc2);

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
