#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>

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

static float snake_oracle(const float *input, const float *up,
                          const float *down, unsigned length, unsigned time) {
    float result = 0.0f;
    for (int down_tap = 0; down_tap < 12; down_tap++) {
        int up_time = (int)time * 2 + down_tap - 5;
        if (up_time < 0) up_time = 0;
        if (up_time >= (int)length * 2) up_time = (int)length * 2 - 1;
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_tap = 0; up_tap < 12; up_tap++) {
            int numerator = raw_time - up_tap;
            if (numerator < 0 || (numerator & 1)) continue;
            int source = numerator / 2 - 5;
            if (source < 0) source = 0;
            if (source >= (int)length) source = (int)length - 1;
            upsampled += input[source] * 2.0f * up[up_tap];
        }
        float sine = sinf(upsampled);
        result += (upsampled + sine * sine) * down[down_tap];
    }
    return result;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    float actual[64];

    const float input1d_values[] = {1,2,3,4};
    const float weight1d_values[] = {1,0,-1};
    const float bias1d_values[] = {0.5f};
    h3_gpu_tensor *input1d = h3_gpu_tensor_from_f32(gpu, input1d_values, 4);
    h3_gpu_tensor *weight1d = h3_gpu_tensor_from_f32(gpu, weight1d_values, 3);
    h3_gpu_tensor *bias1d = h3_gpu_tensor_from_f32(gpu, bias1d_values, 1);
    h3_gpu_tensor *output1d = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(input1d && weight1d && bias1d && output1d);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_conv1d_f32(gpu, output1d, input1d, weight1d, bias1d,
                            1, 4, 1, 1, 3, 1, 1));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output1d, actual, 4));
    CHECK(close_values(actual, (const float[]){-1.5f,-1.5f,-1.5f,3.5f},
                       4, 1e-6f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_conv1d_stride_f32(gpu, output1d, input1d, weight1d, bias1d,
                                   1, 4, 1, 1, 3, 2, 1, 1));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output1d, actual, 2));
    CHECK(close_values(actual, (const float[]){-1.5f,-1.5f}, 2, 1e-6f));

    const float transpose_input_values[] = {1,2};
    const float transpose_weight_values[] = {1,2};
    h3_gpu_tensor *transpose_input = h3_gpu_tensor_from_f32(
        gpu, transpose_input_values, 2);
    h3_gpu_tensor *transpose_weight = h3_gpu_tensor_from_f32(
        gpu, transpose_weight_values, 2);
    CHECK(transpose_input && transpose_weight);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_conv_transpose1d_f32(gpu, output1d, transpose_input,
        transpose_weight, NULL, 1, 2, 1, 1, 2, 2, 0));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output1d, actual, 4));
    CHECK(close_values(actual, (const float[]){1,2,2,4}, 4, 1e-6f));

    h3_gpu_tensor *long_input = h3_gpu_tensor_new_f32(gpu, 65536);
    h3_gpu_tensor *long_output = h3_gpu_tensor_new_f32(gpu, 65536);
    CHECK(long_input && long_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_conv1d_f32(gpu, long_output, long_input, weight1d, NULL,
                            1, 65536, 1, 1, 1, 0, 1));
    CHECK(h3_gpu_conv_transpose1d_f32(gpu, long_output, long_input,
        transpose_weight, NULL, 1, 65536, 1, 1, 1, 1, 0));
    CHECK(h3_gpu_conv3d_f32(gpu, long_output, long_input, weight1d, NULL,
        1, 1, 256, 256, 1, 1, 1, 1, 1, 1, 1, 1));
    CHECK(h3_gpu_submit(gpu));

    const float volume[] = {1,2,3,4,5,6,7,8};
    const float volume_weight[] = {1,1,1,1,1,1,1,1};
    h3_gpu_tensor *volume_input = h3_gpu_tensor_from_f32(gpu, volume, 8);
    h3_gpu_tensor *volume_weights = h3_gpu_tensor_from_f32(
        gpu, volume_weight, 8);
    h3_gpu_tensor *volume_output = h3_gpu_tensor_new_f32(gpu, 1);
    CHECK(volume_input && volume_weights && volume_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_conv3d_f32(gpu, volume_output, volume_input, volume_weights,
        bias1d, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(volume_output, actual, 1));
    CHECK(fabsf(actual[0] - 36.5f) < 1e-6f);

    const float image[] = {1,2,3,4};
    h3_gpu_tensor *image_input = h3_gpu_tensor_from_f32(gpu, image, 4);
    h3_gpu_tensor *image_output = h3_gpu_tensor_new_f32(gpu, 32);
    CHECK(image_input && image_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_vae_encoder_pad_f32(gpu, image_output, image_input,
                                     1, 1, 2, 2, 1, 1, 1, 1, 1, 1));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(image_output, actual, 32));
    for (size_t index = 0; index < 16; index++) CHECK(actual[index] == 0.0f);
    const float reflected[] = {4,3,4,3, 2,1,2,1, 4,3,4,3, 2,1,2,1};
    CHECK(close_values(actual + 16, reflected, 16, 0.0f));

    const float norm_input_values[] = {1,3,5,7};
    const float norm_weight_values[] = {1,1};
    const float norm_bias_values[] = {0,0};
    h3_gpu_tensor *norm_input = h3_gpu_tensor_from_f32(gpu, norm_input_values, 4);
    h3_gpu_tensor *norm_weight = h3_gpu_tensor_from_f32(gpu, norm_weight_values, 2);
    h3_gpu_tensor *norm_bias = h3_gpu_tensor_from_f32(gpu, norm_bias_values, 2);
    h3_gpu_tensor *norm_output = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(norm_input && norm_weight && norm_bias && norm_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_vae_encoder_group_norm_silu_f32(gpu, norm_output, norm_input,
        norm_weight, norm_bias, 1, 1, 1, 2, 2, 1, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(norm_output, actual, 4));
    for (size_t index = 0; index < 4; index++) {
        float normalized = (norm_input_values[index] - 4.0f) /
                           sqrtf(5.0f + 1e-5f);
        float expected = normalized / (1.0f + expf(-normalized));
        CHECK(fabsf(actual[index] - expected) < 2e-6f);
    }

    const float qkv_values[] = {1,2, 3,4, 5,6};
    const float q_bias_values[] = {10,20};
    const float k_bias_values[] = {30,40};
    const float v_bias_values[] = {50,60};
    h3_gpu_tensor *qkv = h3_gpu_tensor_from_f32(gpu, qkv_values, 6);
    h3_gpu_tensor *q_bias = h3_gpu_tensor_from_f32(gpu, q_bias_values, 2);
    h3_gpu_tensor *k_bias = h3_gpu_tensor_from_f32(gpu, k_bias_values, 2);
    h3_gpu_tensor *v_bias = h3_gpu_tensor_from_f32(gpu, v_bias_values, 2);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, 2);
    CHECK(qkv && q_bias && k_bias && v_bias && q && k && v);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_audio_qkv_split_f32(gpu, q, k, v, qkv, q_bias, k_bias,
                                     v_bias, 1, 1, 1, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(q, actual, 2));
    CHECK(close_values(actual, (const float[]){11,22}, 2, 0.0f));

    const float audio_multi_values[] = {
        1,2,3,4, 0,0,0,0, 0,0,0,0,
        5,6,7,8, 0,0,0,0, 0,0,0,0};
    const float audio_zero_bias[] = {0,0,0,0};
    h3_gpu_tensor *audio_multi = h3_gpu_tensor_from_f32(
        gpu, audio_multi_values, 24);
    h3_gpu_tensor *audio_bias = h3_gpu_tensor_from_f32(
        gpu, audio_zero_bias, 4);
    h3_gpu_tensor *audio_q = h3_gpu_tensor_new_f32(gpu, 8);
    h3_gpu_tensor *audio_k = h3_gpu_tensor_new_f32(gpu, 8);
    h3_gpu_tensor *audio_v = h3_gpu_tensor_new_f32(gpu, 8);
    CHECK(audio_multi && audio_bias && audio_q && audio_k && audio_v);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_audio_qkv_split_f32(gpu, audio_q, audio_k, audio_v,
        audio_multi, audio_bias, audio_bias, audio_bias, 1, 2, 2, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(audio_q, actual, 8));
    CHECK(close_values(actual,
        (const float[]){1,2,5,6,3,4,7,8}, 8, 0.0f));

    const float attended_values[] = {1,2,3,4,5,6,7,8};
    h3_gpu_tensor *attended = h3_gpu_tensor_from_f32(gpu, attended_values, 8);
    h3_gpu_tensor *pooled = h3_gpu_tensor_new_f32(gpu, 2);
    CHECK(attended && pooled);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_audio_attention_pool_f32(gpu, pooled, attended,
                                          1, 1, 2, 4, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(pooled, actual, 2));
    CHECK(close_values(actual, (const float[]){3.5f,5.5f}, 2, 0.0f));

    const float filter[] = {0.02f,0.04f,0.06f,0.08f,0.1f,0.2f,
                            0.2f,0.1f,0.08f,0.06f,0.04f,0.02f};
    const float logs[] = {0};
    h3_gpu_tensor *filter_tensor = h3_gpu_tensor_from_f32(gpu, filter, 12);
    h3_gpu_tensor *log_tensor = h3_gpu_tensor_from_f32(gpu, logs, 1);
    h3_gpu_tensor *snake_output = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(filter_tensor && log_tensor && snake_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_alias_free_snake_f32(gpu, long_output, long_input,
        log_tensor, log_tensor, filter_tensor, filter_tensor, 1, 65536, 1));
    CHECK(h3_gpu_alias_free_snake_f32(gpu, snake_output, input1d, log_tensor,
        log_tensor, filter_tensor, filter_tensor, 1, 4, 1));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(snake_output, actual, 4));
    for (unsigned time = 0; time < 4; time++)
        CHECK(fabsf(actual[time] - snake_oracle(input1d_values, filter, filter,
                                                4, time)) < 2e-5f);

    h3_gpu_tensor_free(snake_output); h3_gpu_tensor_free(log_tensor);
    h3_gpu_tensor_free(filter_tensor); h3_gpu_tensor_free(pooled);
    h3_gpu_tensor_free(attended); h3_gpu_tensor_free(audio_v);
    h3_gpu_tensor_free(audio_k); h3_gpu_tensor_free(audio_q);
    h3_gpu_tensor_free(audio_bias); h3_gpu_tensor_free(audio_multi);
    h3_gpu_tensor_free(v); h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(q); h3_gpu_tensor_free(v_bias); h3_gpu_tensor_free(k_bias);
    h3_gpu_tensor_free(q_bias); h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(norm_output); h3_gpu_tensor_free(norm_bias);
    h3_gpu_tensor_free(norm_weight); h3_gpu_tensor_free(norm_input);
    h3_gpu_tensor_free(image_output); h3_gpu_tensor_free(image_input);
    h3_gpu_tensor_free(volume_output); h3_gpu_tensor_free(volume_weights);
    h3_gpu_tensor_free(volume_input); h3_gpu_tensor_free(transpose_weight);
    h3_gpu_tensor_free(transpose_input); h3_gpu_tensor_free(long_output);
    h3_gpu_tensor_free(long_input); h3_gpu_tensor_free(output1d);
    h3_gpu_tensor_free(bias1d); h3_gpu_tensor_free(weight1d);
    h3_gpu_tensor_free(input1d); h3_gpu_free(gpu);
    puts("ok: CUDA convolution, audio and VAE operators");
    return 0;
}
