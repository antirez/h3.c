#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static int close_values(const float *actual, const float *expected,
                        size_t count, float tolerance) {
    for (size_t index = 0; index < count; index++)
        if (fabsf(actual[index] - expected[index]) > tolerance) return 0;
    return 1;
}

static double wall_time(void) {
    struct timespec value;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &value) == 0);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int benchmark_shape(h3_gpu *gpu, const char *name, uint32_t rows,
                           uint32_t input_dim, uint32_t output_dim) {
    h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * input_dim);
    h3_gpu_tensor *weight = h3_gpu_tensor_new_bf16(
        gpu, (size_t)output_dim * input_dim);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * output_dim);
    CHECK(input && weight && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows,
                             input_dim, output_dim));
    CHECK(h3_gpu_submit(gpu));
    for (int run = 0; run < 3; run++) {
        double start = wall_time();
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows,
                                 input_dim, output_dim));
        CHECK(h3_gpu_submit(gpu));
        printf("benchmark %s run=%d seconds=%.6f\n", name, run + 1,
               wall_time() - start);
    }
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(input);
    return 0;
}

static int run_benchmark(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    CHECK(!benchmark_shape(gpu, "qkv", 2048, 5376, 21504));
    CHECK(!benchmark_shape(gpu, "mlp", 2048, 5376, 14336));
    CHECK(!benchmark_shape(gpu, "output", 2048, 5376, 5376));
    h3_gpu_free(gpu);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "benchmark") == 0)
        return run_benchmark();
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    const float input_values[] = {1, 2, 3, -1, 0.5f, 2};
    const float weight_values[] = {2, -1, 0.5f, -3, 4, 1};
    const float bias_values[] = {0.25f, -0.5f};
    const float expected[] = {1.75f, 7.5f, -1.25f, 6.5f};
    float actual[4];

    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, input_values, 6);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, weight_values, 6);
    h3_gpu_tensor *bias = h3_gpu_tensor_from_f32(gpu, bias_values, 2);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(input && weight && bias && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_linear_f32(gpu, output, input, weight, bias, 2, 3, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output, actual, 4));
    CHECK(close_values(actual, expected, 4, 1e-6f));

    h3_gpu_tensor *bf_input = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *bf_weight = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *bf_bias = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *bf_output = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(bf_input && bf_weight && bf_bias && bf_output);
    CHECK(h3_gpu_tensor_write_f32(bf_input, input_values, 6));
    CHECK(h3_gpu_tensor_write_f32(bf_weight, weight_values, 6));
    CHECK(h3_gpu_tensor_write_f32(bf_bias, bias_values, 2));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_linear_bf16(gpu, bf_output, bf_input, bf_weight, bf_bias,
                             2, 3, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bf_output, actual, 4));
    CHECK(close_values(actual, expected, 4, 0.04f));

    const float int8_input_values[] = {1, -2, 0.5f, 1};
    const float int8_weight_values[] = {2, -1, -1, 3};
    const float int8_expected[] = {4, -7, 0, 2.5f};
    h3_gpu_tensor *int8_input = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *int8_weight_source = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *int8_weight = h3_gpu_tensor_new_i8(gpu, 4);
    h3_gpu_tensor *int8_weight_scales = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *int8_quantized_input = h3_gpu_tensor_new_i8(gpu, 4);
    h3_gpu_tensor *int8_input_scales = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *int8_output = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(int8_input && int8_weight_source && int8_weight &&
          int8_weight_scales && int8_quantized_input && int8_input_scales &&
          int8_output);
    CHECK(h3_gpu_tensor_write_f32(int8_input, int8_input_values, 4));
    CHECK(h3_gpu_tensor_write_f32(int8_weight_source, int8_weight_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_quantize_weight_int8(gpu, int8_weight, int8_weight_scales,
                                      int8_weight_source, 2, 2));
    CHECK(h3_gpu_linear_int8_bf16(
        gpu, int8_output, int8_quantized_input, int8_input_scales, int8_input,
        int8_weight, int8_weight_scales, 2, 2, 2, 0));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(int8_output, actual, 4));
    CHECK(close_values(actual, int8_expected, 4, 0.06f));

    const float mlp_input_values[] = {1, 2};
    const float mlp_fc1_values[] = {1,0, 0,1, 2,0, 0,3};
    const float mlp_fc2_values[] = {1,0, 0,1};
    const float mlp_expected[] = {
        2.0f / (1.0f + expf(-1.0f)),
        12.0f / (1.0f + expf(-2.0f))
    };
    h3_gpu_tensor *mlp_input = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *mlp_fc1 = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *mlp_fc2 = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *mlp_activated = h3_gpu_tensor_new_bf16(gpu, 2);
    CHECK(mlp_input && mlp_fc1 && mlp_fc2 && mlp_output && mlp_activated);
    CHECK(h3_gpu_tensor_write_f32(mlp_input, mlp_input_values, 2));
    CHECK(h3_gpu_tensor_write_f32(mlp_fc1, mlp_fc1_values, 8));
    CHECK(h3_gpu_tensor_write_f32(mlp_fc2, mlp_fc2_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_mlp_bf16(gpu, mlp_output, mlp_input, mlp_fc1, mlp_fc2,
                          1, 2, 2, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(mlp_output, actual, 2));
    CHECK(close_values(actual, mlp_expected, 2, 0.08f));

    h3_gpu_tensor *mlp_fc1_i8 = h3_gpu_tensor_new_i8(gpu, 8);
    h3_gpu_tensor *mlp_fc1_scales = h3_gpu_tensor_new_f32(gpu, 4);
    h3_gpu_tensor *mlp_fc2_i8 = h3_gpu_tensor_new_i8(gpu, 4);
    h3_gpu_tensor *mlp_fc2_scales = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *mlp_quantized = h3_gpu_tensor_new_i8(gpu, 2);
    h3_gpu_tensor *mlp_scales = h3_gpu_tensor_new_f32(gpu, 1);
    CHECK(mlp_fc1_i8 && mlp_fc1_scales && mlp_fc2_i8 && mlp_fc2_scales &&
          mlp_quantized && mlp_scales);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_quantize_weight_int8(gpu, mlp_fc1_i8, mlp_fc1_scales,
                                      mlp_fc1, 4, 2));
    CHECK(h3_gpu_quantize_weight_int8(gpu, mlp_fc2_i8, mlp_fc2_scales,
                                      mlp_fc2, 2, 2));
    CHECK(h3_gpu_mlp_int8_bf16(
        gpu, mlp_output, mlp_activated, mlp_quantized, mlp_scales, mlp_input,
        mlp_fc1_i8, mlp_fc1_scales, mlp_fc2_i8, mlp_fc2_scales, mlp_fc1,
        mlp_fc2, 1, 2, 2, 2, 0, 0, 0, 0));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(mlp_output, actual, 2));
    CHECK(close_values(actual, mlp_expected, 2, 0.12f));

    const float head_values[] = {1,2, 3,4, 5,6, 7,8};
    const float identity_values[] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };
    const float head_expected[] = {1,2,5,6, 3,4,7,8};
    h3_gpu_tensor *head_input = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *head_weight_source = h3_gpu_tensor_new_bf16(gpu, 16);
    h3_gpu_tensor *head_weight = h3_gpu_tensor_new_i8(gpu, 16);
    h3_gpu_tensor *head_weight_scales = h3_gpu_tensor_new_f32(gpu, 4);
    h3_gpu_tensor *head_quantized = h3_gpu_tensor_new_i8(gpu, 8);
    h3_gpu_tensor *head_scales = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *head_output = h3_gpu_tensor_new_bf16(gpu, 8);
    float head_actual[8];
    CHECK(head_input && head_weight_source && head_weight &&
          head_weight_scales && head_quantized && head_scales && head_output);
    CHECK(h3_gpu_tensor_write_f32(head_input, head_values, 8));
    CHECK(h3_gpu_tensor_write_f32(head_weight_source, identity_values, 16));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_quantize_weight_int8(gpu, head_weight, head_weight_scales,
                                      head_weight_source, 4, 4));
    CHECK(h3_gpu_linear_int8_head_major_bf16(
        gpu, head_output, head_quantized, head_scales, head_input, head_weight,
        head_weight_scales, 2, 2, 2, 4));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(head_output, head_actual, 8));
    CHECK(close_values(head_actual, head_expected, 8, 0.08f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_mlp_nax_bf16(gpu, mlp_output, mlp_activated, mlp_input,
                              mlp_fc1, mlp_fc2, 1, 2, 2, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(mlp_activated, actual, 2));
    CHECK(close_values(actual, mlp_expected, 2, 0.08f));

    CHECK(h3_gpu_begin(gpu));
    CHECK(!h3_gpu_linear_bf16(gpu, bf_output, input, bf_weight, bf_bias,
                              2, 3, 2));
    CHECK(h3_gpu_error(gpu)[0] != '\0');
    CHECK(h3_gpu_submit(gpu));

    const uint32_t patch_input_dim = 32;
    const uint32_t patch_output_dim = 5376;
    float *patch_input_values = calloc(patch_input_dim + 1, sizeof(float));
    float *patch_weight_values = calloc(
        (size_t)patch_output_dim * patch_input_dim, sizeof(float));
    float *patch_bias_values = malloc((size_t)patch_output_dim * sizeof(float));
    float *patch_actual = malloc((size_t)(patch_output_dim * 2) * sizeof(float));
    CHECK(patch_input_values && patch_weight_values && patch_bias_values &&
          patch_actual);
    for (uint32_t column = 0; column < patch_input_dim; column++)
        patch_input_values[column + 1] = (float)(column + 1);
    for (uint32_t row = 0; row < patch_output_dim; row++)
        patch_bias_values[row] = 1.0f;
    patch_weight_values[0] = 2.0f;
    patch_weight_values[patch_input_dim + 1] = -1.0f;
    h3_gpu_tensor *patch_input = h3_gpu_tensor_from_f32(
        gpu, patch_input_values, patch_input_dim + 1);
    h3_gpu_tensor *patch_weight = h3_gpu_tensor_from_f32(
        gpu, patch_weight_values, (size_t)patch_output_dim * patch_input_dim);
    h3_gpu_tensor *patch_bias = h3_gpu_tensor_from_f32(
        gpu, patch_bias_values, patch_output_dim);
    h3_gpu_tensor *patch_output = h3_gpu_tensor_new_bf16(
        gpu, patch_output_dim + 2);
    CHECK(patch_input && patch_weight && patch_bias && patch_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_patch_linear_bf16_offset(
        gpu, patch_output, 2, patch_input, 1, patch_weight, patch_bias, 1,
        patch_input_dim, patch_output_dim));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32_range(
        patch_output, 2, patch_actual, patch_output_dim));
    CHECK(fabsf(patch_actual[0] - 3.0f) < 0.04f);
    CHECK(fabsf(patch_actual[1] + 1.0f) < 0.04f);
    CHECK(fabsf(patch_actual[patch_output_dim - 1] - 1.0f) < 0.04f);

    const uint32_t map_value[] = {1};
    h3_gpu_tensor *row_map = h3_gpu_tensor_from_u32(gpu, map_value, 1);
    h3_gpu_tensor *mapped_output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)patch_output_dim * 2);
    CHECK(row_map && mapped_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_patch_linear_bf16_map(
        gpu, mapped_output, patch_input, patch_weight, patch_bias, row_map,
        2, 1, patch_input_dim, patch_output_dim));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(mapped_output, patch_actual,
                                 (size_t)patch_output_dim * 2));
    CHECK(fabsf(patch_actual[patch_output_dim] - 1.0f) < 0.04f);
    CHECK(fabsf(patch_actual[patch_output_dim + 1]) < 0.04f);

    h3_gpu_tensor_free(mapped_output);
    h3_gpu_tensor_free(row_map);
    h3_gpu_tensor_free(patch_output);
    h3_gpu_tensor_free(patch_bias);
    h3_gpu_tensor_free(patch_weight);
    h3_gpu_tensor_free(patch_input);
    free(patch_actual);
    free(patch_bias_values);
    free(patch_weight_values);
    free(patch_input_values);

    h3_gpu_tensor_free(int8_output);
    h3_gpu_tensor_free(int8_input_scales);
    h3_gpu_tensor_free(int8_quantized_input);
    h3_gpu_tensor_free(int8_weight_scales);
    h3_gpu_tensor_free(int8_weight);
    h3_gpu_tensor_free(int8_weight_source);
    h3_gpu_tensor_free(int8_input);
    h3_gpu_tensor_free(mlp_activated);
    h3_gpu_tensor_free(mlp_output);
    h3_gpu_tensor_free(mlp_fc2);
    h3_gpu_tensor_free(mlp_fc1);
    h3_gpu_tensor_free(mlp_input);
    h3_gpu_tensor_free(mlp_scales);
    h3_gpu_tensor_free(mlp_quantized);
    h3_gpu_tensor_free(mlp_fc2_scales);
    h3_gpu_tensor_free(mlp_fc2_i8);
    h3_gpu_tensor_free(mlp_fc1_scales);
    h3_gpu_tensor_free(mlp_fc1_i8);
    h3_gpu_tensor_free(head_output);
    h3_gpu_tensor_free(head_scales);
    h3_gpu_tensor_free(head_quantized);
    h3_gpu_tensor_free(head_weight_scales);
    h3_gpu_tensor_free(head_weight);
    h3_gpu_tensor_free(head_weight_source);
    h3_gpu_tensor_free(head_input);

    h3_gpu_tensor_free(bf_output);
    h3_gpu_tensor_free(bf_bias);
    h3_gpu_tensor_free(bf_weight);
    h3_gpu_tensor_free(bf_input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(input);
    h3_gpu_free(gpu);
    puts("ok: CUDA cuBLASLt F32/BF16 linear");
    return 0;
}
