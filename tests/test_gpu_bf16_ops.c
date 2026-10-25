/* Numerical agreement between the BF16 video-VAE primitives and their F32
 * counterparts.
 *
 * The BF16 kernels are a transcription of the F32 ones, so the only tolerated
 * difference is the BF16 quantization of the tensors that cross the GPU
 * boundary. Three protocols are used:
 *
 *   A. accuracy   - the F32 path consumes the reference floats while the BF16
 *                   path consumes their BF16 truncation. This is what the VAE
 *                   decoder actually pays, and it is reported as relative L2
 *                   error plus the largest absolute deviation.
 *   B. structural - both paths consume numerically identical values (the F32
 *                   path receives the BF16 truncation widened back to F32).
 *                   Every intermediate is then bit-identical, so the BF16
 *                   output must equal the BF16 rounding of the F32 output
 *                   exactly. This is the check that catches a mis-copied RoPE
 *                   sign, a wrong cosine/sine index or a misplaced epsilon.
 *   C. reference  - an independent host implementation in double precision is
 *                   compared against both GPU paths.
 *
 * The RoPE half-split rotation is additionally verified analytically: the
 * cosine and sine actually applied by the kernel are recovered from the
 * output pair and compared against the cosine/sine table.
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    /* Match the released video VAE decoder geometry. */
    ROWS = 3,
    WIDTH = 2048,
    HEADS = 32,
    HEAD_DIM = 64,
    ROPE_HALF = 24,
    INNER = HEADS * HEAD_DIM,
    COUNT = ROWS * INNER,
    QKV_COUNT = COUNT * 3,
    ROPE_COUNT = ROWS * ROPE_HALF,
    MAX_TENSORS = 64
};

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} gpu_test;

typedef struct {
    double rel_l2;
    double max_abs;
    size_t mismatches;
} comparison;

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_gpu_bf16_ops.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void require_gpu(gpu_test *test, int condition,
                        const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_gpu_bf16_ops.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *own(gpu_test *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry is full");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void release(gpu_test *test) {
    while (test->owned_count)
        h3_gpu_tensor_free(test->owned[--test->owned_count]);
}

/* BF16 conversion must mirror h3_shaders.metal exactly: round to nearest,
 * ties to even, and an exact widening on the way back. */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Spacing of the BF16 grid at the given magnitude, used to tell a rounding
 * difference apart from a genuine disagreement. */
static double bf16_spacing(float value) {
    double magnitude = fabs((double)value);
    int exponent = 0;
    if (magnitude == 0.0) return 0.0;
    frexp(magnitude, &exponent);
    return ldexp(1.0, exponent - 8);
}

static uint32_t random_state = 0x9e3779b9u;

static float random_value(void) {
    random_state = random_state * 1664525u + 1013904223u;
    uint32_t bits = random_state >> 8;
    float unit = (float)bits * (1.0f / 16777216.0f);
    return unit * 2.0f - 1.0f;
}

/* Relative L2 error, largest absolute deviation and the number of elements
 * that sit more than one BF16 step away from the reference. */
static comparison compare(const float *reference, const float *candidate,
                          size_t count) {
    comparison result = {0.0, 0.0, 0};
    double reference_norm = 0.0;
    double difference_norm = 0.0;
    for (size_t i = 0; i < count; i++) {
        double reference_value = (double)reference[i];
        double difference = (double)candidate[i] - reference_value;
        reference_norm += reference_value * reference_value;
        difference_norm += difference * difference;
        double magnitude = fabs(difference);
        if (magnitude > result.max_abs) result.max_abs = magnitude;
        if (magnitude > bf16_spacing(reference[i])) result.mismatches++;
    }
    if (reference_norm > 0.0)
        result.rel_l2 = sqrt(difference_norm / reference_norm);
    return result;
}

/* Independent host implementation of the video QKV kernel, in double
 * precision. It follows h3_video_qkv_rope_f32: a QK RMS norm over the whole
 * head with the epsilon inside the square root, then a half-split rotation
 * whose second half reuses the cosine/sine of the first half with the sine
 * term added instead of subtracted. */
static void reference_qkv_rope(const float *qkv, const float *rope_cos,
                                const float *rope_sin, float *query,
                                float *key, float *value, uint32_t sequence,
                                uint32_t heads, uint32_t head_dim,
                                uint32_t rope_half, double epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t base = ((size_t)row * heads + head) * head_dim * 3;
            double q_sum = 0.0;
            double k_sum = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                double q = (double)qkv[base + d];
                double k = (double)qkv[base + head_dim + d];
                q_sum += q * q;
                k_sum += k * k;
            }
            double q_inverse = 1.0 / sqrt(q_sum / (double)head_dim + epsilon);
            double k_inverse = 1.0 / sqrt(k_sum / (double)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                double q0 = (double)qkv[base + dimension] * q_inverse;
                double k0 = (double)qkv[base + head_dim + dimension] *
                            k_inverse;
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    double q1 = (double)qkv[base + pair] * q_inverse;
                    double k1 = (double)qkv[base + head_dim + pair] *
                                k_inverse;
                    double cosine = (double)rope_cos[
                        (size_t)row * rope_half + dimension];
                    double sine = (double)rope_sin[
                        (size_t)row * rope_half + dimension];
                    q0 = q0 * cosine - q1 * sine;
                    k0 = k0 * cosine - k1 * sine;
                } else if (dimension < rope_half * 2) {
                    uint32_t pair = dimension - rope_half;
                    double q1 = (double)qkv[base + pair] * q_inverse;
                    double k1 = (double)qkv[base + head_dim + pair] *
                                k_inverse;
                    double cosine = (double)rope_cos[
                        (size_t)row * rope_half + pair];
                    double sine = (double)rope_sin[
                        (size_t)row * rope_half + pair];
                    q0 = q0 * cosine + q1 * sine;
                    k0 = k0 * cosine + k1 * sine;
                }
                size_t output = ((size_t)row * heads + head) * head_dim +
                                dimension;
                query[output] = (float)q0;
                key[output] = (float)k0;
                value[output] = qkv[base + (size_t)head_dim * 2 + dimension];
            }
        }
    }
}

/* Recover the cosine and sine that the kernel applied to one half-split
 * pair. For a rotation by an angle t the pair (x0, x1) maps to
 * (x0*c - x1*s, x1*c + x0*s), so the dot and cross products against the
 * input recover c and s exactly. This validates the rotation direction, the
 * sign convention and the cosine/sine index without restating the kernel.
 *
 * The tolerance accounts for the precision of the output being inspected: an
 * F32 output recovers the coefficients to full precision, a BF16 output only
 * to the BF16 grid. A wrong sign or a wrong index shifts the recovered
 * coefficient by an order of magnitude, far outside both tolerances. */
static void check_rope_rotation(const char *label, const float *qkv,
                                const float *query, const float *rope_cos,
                                const float *rope_sin, uint32_t sequence,
                                uint32_t heads, uint32_t head_dim,
                                uint32_t rope_half, double epsilon,
                                double tolerance) {
    double worst_cosine = 0.0;
    double worst_sine = 0.0;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t base = ((size_t)row * heads + head) * head_dim * 3;
            size_t output_base = ((size_t)row * heads + head) * head_dim;
            double q_sum = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                double q = (double)qkv[base + d];
                q_sum += q * q;
            }
            double q_inverse = 1.0 / sqrt(q_sum / (double)head_dim + epsilon);
            for (uint32_t d = 0; d < rope_half; d++) {
                double x0 = (double)qkv[base + d] * q_inverse;
                double x1 = (double)qkv[base + d + rope_half] * q_inverse;
                double o0 = (double)query[output_base + d];
                double o1 = (double)query[output_base + d + rope_half];
                double norm = x0 * x0 + x1 * x1;
                if (norm < 1e-12) continue;
                double cosine = (x0 * o0 + x1 * o1) / norm;
                double sine = (x0 * o1 - x1 * o0) / norm;
                double expected_cosine = (double)rope_cos[
                    (size_t)row * rope_half + d];
                double expected_sine = (double)rope_sin[
                    (size_t)row * rope_half + d];
                double cosine_error = fabs(cosine - expected_cosine);
                double sine_error = fabs(sine - expected_sine);
                if (cosine_error > worst_cosine)
                    worst_cosine = cosine_error;
                if (sine_error > worst_sine) worst_sine = sine_error;
            }
        }
    }
    printf("  rope rotation (%s): worst |cos| error %.3g, worst |sin| error "
           "%.3g\n", label, worst_cosine, worst_sine);
    require(worst_cosine < tolerance,
            "RoPE cosine does not match the table");
    require(worst_sine < tolerance, "RoPE sine does not match the table");
}

/* The RMS inverse must be rsqrt(sum / head_dim + epsilon) over the whole
 * head, epsilon included inside the square root. Dimensions beyond the
 * rotated range keep the plain normalized value, so they expose the scale
 * factor directly. */
static void check_rms_inverse(const float *qkv, const float *query,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              double epsilon) {
    double worst = 0.0;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            size_t base = ((size_t)row * heads + head) * head_dim * 3;
            size_t output_base = ((size_t)row * heads + head) * head_dim;
            double q_sum = 0.0;
            for (uint32_t d = 0; d < head_dim; d++) {
                double q = (double)qkv[base + d];
                q_sum += q * q;
            }
            double expected = 1.0 / sqrt(q_sum / (double)head_dim + epsilon);
            for (uint32_t d = rope_half * 2; d < head_dim; d++) {
                double source = (double)qkv[base + d];
                if (fabs(source) < 1e-6) continue;
                double actual = (double)query[output_base + d] / source;
                double error = fabs(actual - expected) / expected;
                if (error > worst) worst = error;
            }
        }
    }
    printf("  rms inverse: worst relative error %.3g\n", worst);
    require(worst < 1e-3, "QK RMS inverse disagrees with the reference");
}

static void test_scale_add(gpu_test *test) {
    float *residual = malloc((size_t)ROWS * WIDTH * sizeof(float));
    float *branch = malloc((size_t)ROWS * WIDTH * sizeof(float));
    float *scale = malloc((size_t)WIDTH * sizeof(float));
    float *reference = malloc((size_t)ROWS * WIDTH * sizeof(float));
    uint16_t *residual_bf16 = malloc((size_t)ROWS * WIDTH * sizeof(uint16_t));
    uint16_t *branch_bf16 = malloc((size_t)ROWS * WIDTH * sizeof(uint16_t));
    uint16_t *scale_bf16 = malloc((size_t)WIDTH * sizeof(uint16_t));
    float *candidate = malloc((size_t)ROWS * WIDTH * sizeof(float));
    require(residual && branch && scale && reference && residual_bf16 &&
            branch_bf16 && scale_bf16 && candidate, "host allocation failed");
    for (size_t i = 0; i < (size_t)ROWS * WIDTH; i++) {
        residual[i] = random_value();
        branch[i] = random_value();
        residual_bf16[i] = f32_to_bf16(residual[i]);
        branch_bf16[i] = f32_to_bf16(branch[i]);
    }
    for (size_t i = 0; i < WIDTH; i++) {
        scale[i] = random_value();
        scale_bf16[i] = f32_to_bf16(scale[i]);
    }

    h3_gpu_tensor *f32_residual = own(test, h3_gpu_tensor_from_f32(
        test->gpu, residual, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *f32_branch = own(test, h3_gpu_tensor_from_f32(
        test->gpu, branch, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *f32_scale = own(test, h3_gpu_tensor_from_f32(
        test->gpu, scale, WIDTH));
    h3_gpu_tensor *f32_output = own(test, h3_gpu_tensor_new_f32(
        test->gpu, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *bf16_residual = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, residual_bf16, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *bf16_branch = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, branch_bf16, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *bf16_scale = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, scale_bf16, WIDTH));
    h3_gpu_tensor *bf16_output = own(test, h3_gpu_tensor_new_bf16(
        test->gpu, (size_t)ROWS * WIDTH));

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin scale-add command stream");
    require_gpu(test, h3_gpu_scale_add_f32(test->gpu, f32_output, f32_residual,
               f32_branch, f32_scale, ROWS, WIDTH),
               "encode F32 scale-add");
    require_gpu(test, h3_gpu_scale_add_bf16(test->gpu, bf16_output,
               bf16_residual, bf16_branch, bf16_scale, ROWS, WIDTH),
               "encode BF16 scale-add");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit scale-add command stream");

    require(h3_gpu_tensor_read_f32(f32_output, reference,
                                   (size_t)ROWS * WIDTH),
            "cannot read F32 scale-add output");
    uint16_t *bf16_result = malloc((size_t)ROWS * WIDTH * sizeof(uint16_t));
    require(bf16_result != NULL, "host allocation failed");
    require(h3_gpu_tensor_read_bf16(bf16_output, bf16_result,
                                    (size_t)ROWS * WIDTH),
            "cannot read BF16 scale-add output");
    for (size_t i = 0; i < (size_t)ROWS * WIDTH; i++)
        candidate[i] = bf16_to_f32(bf16_result[i]);

    comparison accuracy = compare(reference, candidate,
                                   (size_t)ROWS * WIDTH);
    printf("scale-add (BF16 input vs F32 reference): rel-L2 %.4g, "
           "max |delta| %.4g, mismatches %zu/%d\n",
           accuracy.rel_l2, accuracy.max_abs, accuracy.mismatches,
           ROWS * WIDTH);
    require(accuracy.rel_l2 < 1e-2, "scale-add relative L2 exceeds 1e-2");

    /* Protocol B: feed the F32 path the same values the BF16 path sees. */
    float *widened_residual = malloc((size_t)ROWS * WIDTH * sizeof(float));
    float *widened_branch = malloc((size_t)ROWS * WIDTH * sizeof(float));
    float *widened_scale = malloc((size_t)WIDTH * sizeof(float));
    require(widened_residual && widened_branch && widened_scale,
            "host allocation failed");
    for (size_t i = 0; i < (size_t)ROWS * WIDTH; i++) {
        widened_residual[i] = bf16_to_f32(residual_bf16[i]);
        widened_branch[i] = bf16_to_f32(branch_bf16[i]);
    }
    for (size_t i = 0; i < WIDTH; i++)
        widened_scale[i] = bf16_to_f32(scale_bf16[i]);

    h3_gpu_tensor *wide_residual = own(test, h3_gpu_tensor_from_f32(
        test->gpu, widened_residual, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *wide_branch = own(test, h3_gpu_tensor_from_f32(
        test->gpu, widened_branch, (size_t)ROWS * WIDTH));
    h3_gpu_tensor *wide_scale = own(test, h3_gpu_tensor_from_f32(
        test->gpu, widened_scale, WIDTH));
    h3_gpu_tensor *wide_output = own(test, h3_gpu_tensor_new_f32(
        test->gpu, (size_t)ROWS * WIDTH));
    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin scale-add identity command stream");
    require_gpu(test, h3_gpu_scale_add_f32(test->gpu, wide_output,
               wide_residual, wide_branch, wide_scale, ROWS, WIDTH),
               "encode F32 scale-add with identical values");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit scale-add identity command stream");
    require(h3_gpu_tensor_read_f32(wide_output, reference,
                                   (size_t)ROWS * WIDTH),
            "cannot read F32 scale-add identity output");

    size_t exact = 0;
    for (size_t i = 0; i < (size_t)ROWS * WIDTH; i++) {
        if (f32_to_bf16(reference[i]) == bf16_result[i]) exact++;
    }
    printf("scale-add (identical values): %zu/%d bit-exact against the F32 "
           "path\n", exact, ROWS * WIDTH);
    require(exact == (size_t)ROWS * WIDTH,
            "scale-add BF16 output is not the BF16 rounding of the F32 output");

    free(residual); free(branch); free(scale); free(reference);
    free(residual_bf16); free(branch_bf16); free(scale_bf16);
    free(candidate); free(bf16_result);
    free(widened_residual); free(widened_branch); free(widened_scale);
}

static void test_qkv_rope(gpu_test *test) {
    const float epsilon = 1e-5f;
    float *qkv = malloc((size_t)QKV_COUNT * sizeof(float));
    float *rope_cos = malloc((size_t)ROPE_COUNT * sizeof(float));
    float *rope_sin = malloc((size_t)ROPE_COUNT * sizeof(float));
    uint16_t *qkv_bf16 = malloc((size_t)QKV_COUNT * sizeof(uint16_t));
    uint16_t *rope_cos_bf16 = malloc((size_t)ROPE_COUNT * sizeof(uint16_t));
    uint16_t *rope_sin_bf16 = malloc((size_t)ROPE_COUNT * sizeof(uint16_t));
    float *reference_query = malloc((size_t)COUNT * sizeof(float));
    float *reference_key = malloc((size_t)COUNT * sizeof(float));
    float *reference_value = malloc((size_t)COUNT * sizeof(float));
    float *candidate_query = malloc((size_t)COUNT * sizeof(float));
    float *candidate_key = malloc((size_t)COUNT * sizeof(float));
    float *candidate_value = malloc((size_t)COUNT * sizeof(float));
    require(qkv && rope_cos && rope_sin && qkv_bf16 && rope_cos_bf16 &&
            rope_sin_bf16 && reference_query && reference_key &&
            reference_value && candidate_query && candidate_key &&
            candidate_value, "host allocation failed");

    for (size_t i = 0; i < (size_t)QKV_COUNT; i++) {
        qkv[i] = random_value();
        qkv_bf16[i] = f32_to_bf16(qkv[i]);
    }
    for (size_t i = 0; i < (size_t)ROPE_COUNT; i++) {
        double angle = (double)random_value() * 3.14159265358979323846;
        rope_cos[i] = (float)cos(angle);
        rope_sin[i] = (float)sin(angle);
        rope_cos_bf16[i] = f32_to_bf16(rope_cos[i]);
        rope_sin_bf16[i] = f32_to_bf16(rope_sin[i]);
    }

    h3_gpu_tensor *f32_qkv = own(test, h3_gpu_tensor_from_f32(
        test->gpu, qkv, QKV_COUNT));
    h3_gpu_tensor *f32_cos = own(test, h3_gpu_tensor_from_f32(
        test->gpu, rope_cos, ROPE_COUNT));
    h3_gpu_tensor *f32_sin = own(test, h3_gpu_tensor_from_f32(
        test->gpu, rope_sin, ROPE_COUNT));
    h3_gpu_tensor *f32_query = own(test, h3_gpu_tensor_new_f32(
        test->gpu, COUNT));
    h3_gpu_tensor *f32_key = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *f32_value = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *bf16_qkv = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, qkv_bf16, QKV_COUNT));
    h3_gpu_tensor *bf16_cos = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, rope_cos_bf16, ROPE_COUNT));
    h3_gpu_tensor *bf16_sin = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, rope_sin_bf16, ROPE_COUNT));
    h3_gpu_tensor *bf16_query = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));
    h3_gpu_tensor *bf16_key = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));
    h3_gpu_tensor *bf16_value = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin video QKV command stream");
    require_gpu(test, h3_gpu_video_qkv_rope_f32(test->gpu, f32_query,
               f32_key, f32_value, f32_qkv, f32_cos, f32_sin, ROWS, HEADS,
               HEAD_DIM, ROPE_HALF, epsilon), "encode F32 video QKV");
    require_gpu(test, h3_gpu_video_qkv_rope_bf16(test->gpu, bf16_query,
               bf16_key, bf16_value, bf16_qkv, bf16_cos, bf16_sin, ROWS,
               HEADS, HEAD_DIM, ROPE_HALF, epsilon), "encode BF16 video QKV");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit video QKV command stream");

    uint16_t *bf16_query_result = malloc((size_t)COUNT * sizeof(uint16_t));
    uint16_t *bf16_key_result = malloc((size_t)COUNT * sizeof(uint16_t));
    uint16_t *bf16_value_result = malloc((size_t)COUNT * sizeof(uint16_t));
    require(bf16_query_result && bf16_key_result && bf16_value_result,
            "host allocation failed");
    require(h3_gpu_tensor_read_f32(f32_query, reference_query, COUNT),
            "cannot read F32 query");
    require(h3_gpu_tensor_read_f32(f32_key, reference_key, COUNT),
            "cannot read F32 key");
    require(h3_gpu_tensor_read_f32(f32_value, reference_value, COUNT),
            "cannot read F32 value");
    require(h3_gpu_tensor_read_bf16(bf16_query, bf16_query_result, COUNT),
            "cannot read BF16 query");
    require(h3_gpu_tensor_read_bf16(bf16_key, bf16_key_result, COUNT),
            "cannot read BF16 key");
    require(h3_gpu_tensor_read_bf16(bf16_value, bf16_value_result, COUNT),
            "cannot read BF16 value");
    for (size_t i = 0; i < COUNT; i++) {
        candidate_query[i] = bf16_to_f32(bf16_query_result[i]);
        candidate_key[i] = bf16_to_f32(bf16_key_result[i]);
        candidate_value[i] = bf16_to_f32(bf16_value_result[i]);
    }

    comparison query_accuracy = compare(reference_query, candidate_query, COUNT);
    comparison key_accuracy = compare(reference_key, candidate_key, COUNT);
    printf("video qkv (BF16 input vs F32 reference): query rel-L2 %.4g, "
           "max |delta| %.4g, mismatches %zu/%d\n",
           query_accuracy.rel_l2, query_accuracy.max_abs,
           query_accuracy.mismatches, COUNT);
    printf("video qkv (BF16 input vs F32 reference): key   rel-L2 %.4g, "
           "max |delta| %.4g, mismatches %zu/%d\n",
           key_accuracy.rel_l2, key_accuracy.max_abs,
           key_accuracy.mismatches, COUNT);
    require(query_accuracy.rel_l2 < 1e-2, "query relative L2 exceeds 1e-2");
    require(key_accuracy.rel_l2 < 1e-2, "key relative L2 exceeds 1e-2");

    /* The value channel is a pure copy, so it must match the BF16 rounding
     * of the F32 value exactly. */
    size_t value_exact = 0;
    for (size_t i = 0; i < COUNT; i++) {
        if (f32_to_bf16(reference_value[i]) == bf16_value_result[i])
            value_exact++;
    }
    printf("video qkv value channel: %zu/%d bit-exact against the F32 path\n",
           value_exact, COUNT);
    require(value_exact == COUNT,
            "video qkv value channel is not a verbatim copy");

    /* Protocol B: identical values on both paths. */
    float *wide_qkv = malloc((size_t)QKV_COUNT * sizeof(float));
    float *wide_cos = malloc((size_t)ROPE_COUNT * sizeof(float));
    float *wide_sin = malloc((size_t)ROPE_COUNT * sizeof(float));
    require(wide_qkv && wide_cos && wide_sin, "host allocation failed");
    for (size_t i = 0; i < (size_t)QKV_COUNT; i++)
        wide_qkv[i] = bf16_to_f32(qkv_bf16[i]);
    for (size_t i = 0; i < (size_t)ROPE_COUNT; i++) {
        wide_cos[i] = bf16_to_f32(rope_cos_bf16[i]);
        wide_sin[i] = bf16_to_f32(rope_sin_bf16[i]);
    }
    h3_gpu_tensor *wide_qkv_tensor = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_qkv, QKV_COUNT));
    h3_gpu_tensor *wide_cos_tensor = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_cos, ROPE_COUNT));
    h3_gpu_tensor *wide_sin_tensor = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_sin, ROPE_COUNT));
    h3_gpu_tensor *wide_query = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *wide_key = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *wide_value = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin video QKV identity command stream");
    require_gpu(test, h3_gpu_video_qkv_rope_f32(test->gpu, wide_query,
               wide_key, wide_value, wide_qkv_tensor, wide_cos_tensor,
               wide_sin_tensor, ROWS, HEADS, HEAD_DIM, ROPE_HALF, epsilon),
               "encode F32 video QKV with identical values");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit video QKV identity command stream");
    require(h3_gpu_tensor_read_f32(wide_query, reference_query, COUNT),
            "cannot read F32 identity query");
    require(h3_gpu_tensor_read_f32(wide_key, reference_key, COUNT),
            "cannot read F32 identity key");
    require(h3_gpu_tensor_read_f32(wide_value, reference_value, COUNT),
            "cannot read F32 identity value");

    size_t query_exact = 0;
    size_t key_exact = 0;
    for (size_t i = 0; i < COUNT; i++) {
        if (f32_to_bf16(reference_query[i]) == bf16_query_result[i])
            query_exact++;
        if (f32_to_bf16(reference_key[i]) == bf16_key_result[i]) key_exact++;
    }
    printf("video qkv (identical values): query %zu/%d bit-exact, key %zu/%d "
           "bit-exact against the F32 path\n",
           query_exact, COUNT, key_exact, COUNT);
    require(query_exact == COUNT,
            "BF16 query is not the BF16 rounding of the F32 query");
    require(key_exact == COUNT,
            "BF16 key is not the BF16 rounding of the F32 key");

    /* Protocol C: the independent double precision reference. */
    float *host_query = malloc((size_t)COUNT * sizeof(float));
    float *host_key = malloc((size_t)COUNT * sizeof(float));
    float *host_value = malloc((size_t)COUNT * sizeof(float));
    require(host_query && host_key && host_value, "host allocation failed");
    reference_qkv_rope(wide_qkv, wide_cos, wide_sin, host_query, host_key,
                       host_value, ROWS, HEADS, HEAD_DIM, ROPE_HALF,
                       (double)epsilon);
    comparison host_query_error = compare(host_query, reference_query, COUNT);
    comparison host_key_error = compare(host_key, reference_key, COUNT);
    comparison host_value_error = compare(host_value, reference_value, COUNT);
    printf("video qkv (host reference vs F32 path): query rel-L2 %.4g, "
           "key rel-L2 %.4g, value rel-L2 %.4g\n",
           host_query_error.rel_l2, host_key_error.rel_l2,
           host_value_error.rel_l2);
    require(host_query_error.rel_l2 < 1e-5,
            "F32 query disagrees with the host reference");
    require(host_key_error.rel_l2 < 1e-5,
            "F32 key disagrees with the host reference");
    require(host_value_error.max_abs == 0.0,
            "F32 value is not a verbatim copy");

    /* The BF16 path must also track the host reference. */
    comparison bf16_query_error = compare(host_query, candidate_query, COUNT);
    comparison bf16_key_error = compare(host_key, candidate_key, COUNT);
    printf("video qkv (host reference vs BF16 path): query rel-L2 %.4g, "
           "key rel-L2 %.4g\n",
           bf16_query_error.rel_l2, bf16_key_error.rel_l2);
    require(bf16_query_error.rel_l2 < 1e-2,
            "BF16 query diverges from the host reference");
    require(bf16_key_error.rel_l2 < 1e-2,
            "BF16 key diverges from the host reference");

    check_rope_rotation("F32 path", wide_qkv, reference_query, wide_cos,
                        wide_sin, ROWS, HEADS, HEAD_DIM, ROPE_HALF,
                        (double)epsilon, 1e-3);
    /* The same probe on the BF16 output, whose precision only allows a
     * looser bound. */
    check_rope_rotation("BF16 path", wide_qkv, candidate_query, wide_cos,
                        wide_sin, ROWS, HEADS, HEAD_DIM, ROPE_HALF,
                        (double)epsilon, 1e-2);
    check_rms_inverse(wide_qkv, reference_query, ROWS, HEADS, HEAD_DIM,
                      ROPE_HALF, (double)epsilon);

    free(qkv); free(rope_cos); free(rope_sin);
    free(qkv_bf16); free(rope_cos_bf16); free(rope_sin_bf16);
    free(reference_query); free(reference_key); free(reference_value);
    free(candidate_query); free(candidate_key); free(candidate_value);
    free(bf16_query_result); free(bf16_key_result); free(bf16_value_result);
    free(wide_qkv); free(wide_cos); free(wide_sin);
    free(host_query); free(host_key); free(host_value);
}

/* The epsilon of the QK RMS norm has to sit inside the square root, on the
 * mean of the squares. With the production value of 1e-5 the effect is below
 * the noise floor of the data, so the placement is probed with a large
 * epsilon instead: a norm that ignores the epsilon, divides by
 * (head_dim + epsilon) or adds the epsilon after the square root all give a
 * visibly different scale factor. */
static void test_epsilon_placement(gpu_test *test) {
    const float epsilon = 1.0f;
    float *qkv = malloc((size_t)QKV_COUNT * sizeof(float));
    float *rope_cos = malloc((size_t)ROPE_COUNT * sizeof(float));
    float *rope_sin = malloc((size_t)ROPE_COUNT * sizeof(float));
    uint16_t *qkv_bf16 = malloc((size_t)QKV_COUNT * sizeof(uint16_t));
    uint16_t *rope_cos_bf16 = malloc((size_t)ROPE_COUNT * sizeof(uint16_t));
    uint16_t *rope_sin_bf16 = malloc((size_t)ROPE_COUNT * sizeof(uint16_t));
    float *wide_qkv = malloc((size_t)QKV_COUNT * sizeof(float));
    float *wide_cos = malloc((size_t)ROPE_COUNT * sizeof(float));
    float *wide_sin = malloc((size_t)ROPE_COUNT * sizeof(float));
    float *reference_query = malloc((size_t)COUNT * sizeof(float));
    float *reference_key = malloc((size_t)COUNT * sizeof(float));
    float *reference_value = malloc((size_t)COUNT * sizeof(float));
    float *host_query = malloc((size_t)COUNT * sizeof(float));
    float *host_key = malloc((size_t)COUNT * sizeof(float));
    float *host_value = malloc((size_t)COUNT * sizeof(float));
    require(qkv && rope_cos && rope_sin && qkv_bf16 && rope_cos_bf16 &&
            rope_sin_bf16 && wide_qkv && wide_cos && wide_sin &&
            reference_query && reference_key && reference_value &&
            host_query && host_key && host_value, "host allocation failed");

    for (size_t i = 0; i < (size_t)QKV_COUNT; i++) {
        qkv[i] = random_value();
        qkv_bf16[i] = f32_to_bf16(qkv[i]);
        wide_qkv[i] = bf16_to_f32(qkv_bf16[i]);
    }
    for (size_t i = 0; i < (size_t)ROPE_COUNT; i++) {
        double angle = (double)random_value() * 3.14159265358979323846;
        rope_cos[i] = (float)cos(angle);
        rope_sin[i] = (float)sin(angle);
        rope_cos_bf16[i] = f32_to_bf16(rope_cos[i]);
        rope_sin_bf16[i] = f32_to_bf16(rope_sin[i]);
        wide_cos[i] = bf16_to_f32(rope_cos_bf16[i]);
        wide_sin[i] = bf16_to_f32(rope_sin_bf16[i]);
    }

    /* Both paths receive the same values, the F32 path the BF16 truncation
     * widened back, so the two must agree bit for bit. */
    h3_gpu_tensor *f32_qkv = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_qkv, QKV_COUNT));
    h3_gpu_tensor *f32_cos = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_cos, ROPE_COUNT));
    h3_gpu_tensor *f32_sin = own(test, h3_gpu_tensor_from_f32(
        test->gpu, wide_sin, ROPE_COUNT));
    h3_gpu_tensor *f32_query = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *f32_key = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *f32_value = own(test, h3_gpu_tensor_new_f32(test->gpu, COUNT));
    h3_gpu_tensor *bf16_qkv = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, qkv_bf16, QKV_COUNT));
    h3_gpu_tensor *bf16_cos = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, rope_cos_bf16, ROPE_COUNT));
    h3_gpu_tensor *bf16_sin = own(test, h3_gpu_tensor_from_bf16(
        test->gpu, rope_sin_bf16, ROPE_COUNT));
    h3_gpu_tensor *bf16_query = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));
    h3_gpu_tensor *bf16_key = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));
    h3_gpu_tensor *bf16_value = own(test, h3_gpu_tensor_new_bf16(test->gpu, COUNT));

    require_gpu(test, h3_gpu_begin(test->gpu),
                "begin epsilon command stream");
    require_gpu(test, h3_gpu_video_qkv_rope_f32(test->gpu, f32_query,
               f32_key, f32_value, f32_qkv, f32_cos, f32_sin, ROWS, HEADS,
               HEAD_DIM, ROPE_HALF, epsilon), "encode F32 video QKV");
    require_gpu(test, h3_gpu_video_qkv_rope_bf16(test->gpu, bf16_query,
               bf16_key, bf16_value, bf16_qkv, bf16_cos, bf16_sin, ROWS,
               HEADS, HEAD_DIM, ROPE_HALF, epsilon),
               "encode BF16 video QKV");
    require_gpu(test, h3_gpu_submit(test->gpu),
                "submit epsilon command stream");

    uint16_t *bf16_query_result = malloc((size_t)COUNT * sizeof(uint16_t));
    uint16_t *bf16_key_result = malloc((size_t)COUNT * sizeof(uint16_t));
    require(bf16_query_result && bf16_key_result, "host allocation failed");
    require(h3_gpu_tensor_read_f32(f32_query, reference_query, COUNT),
            "cannot read F32 query");
    require(h3_gpu_tensor_read_f32(f32_key, reference_key, COUNT),
            "cannot read F32 key");
    require(h3_gpu_tensor_read_f32(f32_value, reference_value, COUNT),
            "cannot read F32 value");
    require(h3_gpu_tensor_read_bf16(bf16_query, bf16_query_result, COUNT),
            "cannot read BF16 query");
    require(h3_gpu_tensor_read_bf16(bf16_key, bf16_key_result, COUNT),
            "cannot read BF16 key");

    /* The F32 path must reproduce the reference with the epsilon inside the
     * square root of the mean of the squares. */
    reference_qkv_rope(wide_qkv, wide_cos, wide_sin, host_query, host_key,
                       host_value, ROWS, HEADS, HEAD_DIM, ROPE_HALF,
                       (double)epsilon);
    comparison host_error = compare(host_query, reference_query, COUNT);
    printf("epsilon %.3g: F32 path vs host reference rel-L2 %.4g\n",
           (double)epsilon, host_error.rel_l2);
    require(host_error.rel_l2 < 1e-5,
            "F32 path ignores the epsilon or places it wrongly");

    /* A norm that dropped the epsilon would scale the query by a visibly
     * different factor, so the two must differ. */
    float *host_no_epsilon = malloc((size_t)COUNT * sizeof(float));
    float *host_no_epsilon_key = malloc((size_t)COUNT * sizeof(float));
    float *host_no_epsilon_value = malloc((size_t)COUNT * sizeof(float));
    require(host_no_epsilon && host_no_epsilon_key && host_no_epsilon_value,
            "host allocation failed");
    reference_qkv_rope(wide_qkv, wide_cos, wide_sin, host_no_epsilon,
                       host_no_epsilon_key, host_no_epsilon_value, ROWS,
                       HEADS, HEAD_DIM, ROPE_HALF, 0.0);
    comparison epsilon_effect = compare(host_no_epsilon, reference_query,
                                        COUNT);
    printf("epsilon %.3g: effect on the query rel-L2 %.4g\n",
           (double)epsilon, epsilon_effect.rel_l2);
    require(epsilon_effect.rel_l2 > 1e-3,
            "the epsilon probe is not sensitive enough to be conclusive");

    /* The BF16 path must be the BF16 rounding of the F32 path. */
    size_t query_exact = 0;
    size_t key_exact = 0;
    for (size_t i = 0; i < COUNT; i++) {
        if (f32_to_bf16(reference_query[i]) == bf16_query_result[i])
            query_exact++;
        if (f32_to_bf16(reference_key[i]) == bf16_key_result[i]) key_exact++;
    }
    printf("epsilon %.3g: BF16 query %zu/%d bit-exact, key %zu/%d bit-exact "
           "against the F32 path\n",
           (double)epsilon, query_exact, COUNT, key_exact, COUNT);
    require(query_exact == COUNT, "BF16 query ignores the epsilon");
    require(key_exact == COUNT, "BF16 key ignores the epsilon");

    free(qkv); free(rope_cos); free(rope_sin);
    free(qkv_bf16); free(rope_cos_bf16); free(rope_sin_bf16);
    free(wide_qkv); free(wide_cos); free(wide_sin);
    free(reference_query); free(reference_key); free(reference_value);
    free(host_query); free(host_key); free(host_value);
    free(bf16_query_result); free(bf16_key_result);
    free(host_no_epsilon); free(host_no_epsilon_key);
    free(host_no_epsilon_value);
}

int main(int argc, char **argv) {
    /* The shader path may be given explicitly so the test does not depend on
     * the working directory. */
    const char *shader = argc > 1 ? argv[1] : "h3_shaders.metal";
    char error[512];
    gpu_test test;
    memset(&test, 0, sizeof(test));
    test.gpu = h3_gpu_create(shader, error, sizeof(error));
    if (!test.gpu) {
        fprintf(stderr, "FAIL tests/test_gpu_bf16_ops.c: %s\n", error);
        return 1;
    }
    printf("video qkv geometry: rows %d, heads %d, head_dim %d, rope_half %d\n",
           ROWS, HEADS, HEAD_DIM, ROPE_HALF);
    test_scale_add(&test);
    test_qkv_rope(&test);
    test_epsilon_placement(&test);
    release(&test);
    h3_gpu_free(test.gpu);
    printf("ok: BF16 video VAE primitives match their F32 counterparts\n");
    return 0;
}
