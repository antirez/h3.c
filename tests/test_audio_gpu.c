#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TENSORS = 32 };

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t count;
} test_context;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_audio_gpu.c: %s\n", message);
    exit(1);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    if (!tensor || test->count == MAX_TENSORS) die("tensor allocation failed");
    test->owned[test->count++] = tensor;
    return tensor;
}

static void gpu_ok(test_context *test, int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_audio_gpu.c: %s: %s\n", operation,
                h3_gpu_error(test->gpu));
        exit(1);
    }
}

static void compare(h3_gpu_tensor *tensor, const float *expected, size_t count,
                    float tolerance, const char *label) {
    float *got = malloc(count * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got, count))
        die("cannot read result");
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(got[index] - expected[index]);
        if (error > maximum) maximum = error;
    }
    free(got);
    printf("audio primitive %-16s max abs %.7g\n", label, maximum);
    if (maximum > tolerance) die(label);
}

static void conv_reference(float *output, const float *input,
                           const float *weight, const float *bias, int length,
                           int input_channels, int output_channels, int kernel,
                           int padding, int dilation) {
    for (int time = 0; time < length; time++) {
        for (int out = 0; out < output_channels; out++) {
            float sum = bias[out];
            for (int k = 0; k < kernel; k++) {
                int source = time + k * dilation - padding;
                if (source < 0 || source >= length) continue;
                for (int in = 0; in < input_channels; in++) {
                    sum += input[source * input_channels + in] *
                           weight[(out * input_channels + in) * kernel + k];
                }
            }
            output[time * output_channels + out] = sum;
        }
    }
}

static void transpose_reference(float *output, const float *input,
                                const float *weight, const float *bias,
                                int length, int input_channels,
                                int output_channels, int kernel, int stride,
                                int padding) {
    int output_length = (length - 1) * stride + kernel - 2 * padding;
    for (int time = 0; time < output_length; time++) {
        for (int out = 0; out < output_channels; out++) {
            float sum = bias[out];
            for (int k = 0; k < kernel; k++) {
                int numerator = time + padding - k;
                if (numerator < 0 || numerator % stride) continue;
                int source = numerator / stride;
                if (source >= length) continue;
                for (int in = 0; in < input_channels; in++) {
                    sum += input[source * input_channels + in] *
                           weight[(in * output_channels + out) * kernel + k];
                }
            }
            output[time * output_channels + out] = sum;
        }
    }
}

static float upsample_at(const float *input, const float *filter, int length,
                         int channels, int channel, int up_time) {
    int raw = up_time + 15;
    float result = 0.0f;
    for (int k = 0; k < 12; k++) {
        int numerator = raw - k;
        if (numerator < 0 || numerator % 2) continue;
        int source = numerator / 2 - 5;
        if (source < 0) source = 0;
        if (source >= length) source = length - 1;
        result += input[source * channels + channel] * 2.0f * filter[k];
    }
    return result;
}

static void activation_reference(float *output, const float *input,
                                 const float *alpha_log,
                                 const float *beta_log,
                                 const float *up_filter,
                                 const float *down_filter, int length,
                                 int channels) {
    for (int time = 0; time < length; time++) {
        for (int channel = 0; channel < channels; channel++) {
            float alpha = expf(alpha_log[channel]);
            float beta = expf(beta_log[channel]);
            float result = 0.0f;
            for (int k = 0; k < 12; k++) {
                int up_time = time * 2 + k - 5;
                if (up_time < 0) up_time = 0;
                if (up_time >= length * 2) up_time = length * 2 - 1;
                float value = upsample_at(input, up_filter, length, channels,
                                          channel, up_time);
                float sine = sinf(alpha * value);
                value += sine * sine / (beta + 1e-9f);
                result += value * down_filter[k];
            }
            output[time * channels + channel] = result;
        }
    }
}

int main(void) {
    test_context test = {0};
    char error[512];
    test.gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!test.gpu) die(error);

    const float input_values[] = {0.2f, -0.3f, 0.5f, 0.1f,
                                  -0.4f, 0.7f, 0.8f, -0.2f};
    const float vector_values[] = {
        0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f,
        -0.2f, 0.7f, 0.1f, -0.4f, 0.3f, 0.5f,
    };
    const float magnitudes[] = {1.5f, 0.75f};
    const float bias_values[] = {0.1f, -0.2f};
    float normalized[12];
    for (int outer = 0; outer < 2; outer++) {
        float square = 0.0f;
        for (int index = 0; index < 6; index++) {
            float value = vector_values[outer * 6 + index];
            square += value * value;
        }
        float scale = magnitudes[outer] / sqrtf(square);
        for (int index = 0; index < 6; index++)
            normalized[outer * 6 + index] =
                vector_values[outer * 6 + index] * scale;
    }
    float conv_expected[8];
    conv_reference(conv_expected, input_values, normalized, bias_values,
                   4, 2, 2, 3, 1, 1);

    const float transpose_weight[] = {0.2f, 0.3f, -0.1f, 0.4f,
                                      -0.5f, 0.1f, 0.25f, 0.2f};
    const float transpose_bias[] = {0.05f};
    float transpose_expected[6];
    transpose_reference(transpose_expected, input_values, transpose_weight,
                        transpose_bias, 3, 2, 1, 4, 2, 1);

    float filter[12] = {0};
    filter[5] = filter[6] = 0.5f;
    const float alpha[] = {0.0f, 0.1f};
    const float beta[] = {0.0f, -0.2f};
    float activation_expected[8];
    activation_reference(activation_expected, input_values, alpha, beta,
                         filter, filter, 4, 2);
    const float right_values[] = {2.0f, -4.0f, 0.2f, 0.4f};
    float add_expected[4], clip_expected[4];
    for (int index = 0; index < 4; index++) {
        add_expected[index] = 0.25f * input_values[index] +
                              0.5f * right_values[index];
        clip_expected[index] = fminf(0.4f, fmaxf(-0.4f, add_expected[index]));
    }

    h3_gpu_tensor *input = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, input_values, 8));
    h3_gpu_tensor *vector = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, vector_values, 12));
    h3_gpu_tensor *magnitude = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, magnitudes, 2));
    h3_gpu_tensor *weight = own(&test, h3_gpu_tensor_new_f32(test.gpu, 12));
    h3_gpu_tensor *bias = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, bias_values, 2));
    h3_gpu_tensor *conv = own(&test, h3_gpu_tensor_new_f32(test.gpu, 8));
    h3_gpu_tensor *trans_weight = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, transpose_weight, 8));
    h3_gpu_tensor *trans_bias = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, transpose_bias, 1));
    h3_gpu_tensor *trans = own(&test, h3_gpu_tensor_new_f32(test.gpu, 6));
    h3_gpu_tensor *alpha_gpu = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, alpha, 2));
    h3_gpu_tensor *beta_gpu = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, beta, 2));
    h3_gpu_tensor *filter_gpu = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, filter, 12));
    h3_gpu_tensor *activated = own(&test, h3_gpu_tensor_new_f32(test.gpu, 8));
    h3_gpu_tensor *right = own(&test, h3_gpu_tensor_from_f32(
        test.gpu, right_values, 4));
    h3_gpu_tensor *added = own(&test, h3_gpu_tensor_new_f32(test.gpu, 4));
    h3_gpu_tensor *clipped = own(&test, h3_gpu_tensor_new_f32(test.gpu, 4));

    gpu_ok(&test, h3_gpu_begin(test.gpu), "begin");
    gpu_ok(&test, h3_gpu_weight_norm_f32(test.gpu, weight, vector, magnitude,
                                         2, 6), "weight norm");
    gpu_ok(&test, h3_gpu_conv1d_f32(test.gpu, conv, input, weight, bias,
                                    1, 4, 2, 2, 3, 1, 1), "Conv1d");
    gpu_ok(&test, h3_gpu_conv_transpose1d_f32(
        test.gpu, trans, input, trans_weight, trans_bias,
        1, 3, 2, 1, 4, 2, 1), "ConvTranspose1d");
    gpu_ok(&test, h3_gpu_alias_free_snake_f32(
        test.gpu, activated, input, alpha_gpu, beta_gpu, filter_gpu,
        filter_gpu, 1, 4, 2), "alias-free SnakeBeta");
    gpu_ok(&test, h3_gpu_add_scaled_f32(test.gpu, added, input, right,
                                        0.25f, 0.5f, 4), "scaled add");
    gpu_ok(&test, h3_gpu_clip_f32(test.gpu, clipped, added, 4, -0.4f, 0.4f),
           "clip");
    gpu_ok(&test, h3_gpu_submit(test.gpu), "submit");

    compare(weight, normalized, 12, 2e-6f, "weight norm");
    compare(conv, conv_expected, 8, 2e-5f, "Conv1d");
    compare(trans, transpose_expected, 6, 2e-5f, "ConvTranspose1d");
    compare(activated, activation_expected, 8, 2e-5f, "SnakeBeta");
    compare(added, add_expected, 4, 1e-7f, "scaled add");
    compare(clipped, clip_expected, 4, 1e-7f, "clip");

    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(test.gpu, &stats) || stats.mps_conv_dispatches != 2)
        die("MPS Conv1d dispatch count mismatch");
    for (size_t index = 0; index < test.count; index++)
        h3_gpu_tensor_free(test.owned[index]);
    h3_gpu_free(test.gpu);
    puts("ok: native AudioVAE Metal primitives match host references");
    return 0;
}
