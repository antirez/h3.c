#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 32,
    QUERY_HEADS = 8,
    KV_HEADS = 2,
    HEAD_DIM = 128,
    TRIALS = 8
};

static uint32_t random_state;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_gqa.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void require_gpu(h3_gpu *gpu, int condition, const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_gqa.c: %s: %s\n", operation,
            h3_gpu_error(gpu));
    exit(1);
}

static uint32_t random_u32(void) {
    uint32_t value = random_state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    random_state = value;
    return value;
}

static float random_signed(void) {
    return (float)((random_u32() >> 8) *
                   (1.0 / 16777216.0) * 2.0 - 1.0);
}

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

static void make_inputs(unsigned trial, uint16_t *query, uint16_t *key,
                        uint16_t *value) {
    static const float amplitudes[TRIALS] = {
        0.25f, 0.75f, 1.5f, 3.0f, 0.5f, 1.0f, 2.0f, 6.0f
    };
    float amplitude = amplitudes[trial];
    random_state = 0x9e3779b9u ^ (uint32_t)(trial * 0x85ebca6bu);
    for (size_t index = 0;
         index < (size_t)SEQUENCE * QUERY_HEADS * HEAD_DIM; index++) {
        query[index] = f32_to_bf16(random_signed() * amplitude);
    }
    for (size_t index = 0;
         index < (size_t)SEQUENCE * KV_HEADS * HEAD_DIM; index++) {
        key[index] = f32_to_bf16(random_signed() * amplitude);
        value[index] = f32_to_bf16(random_signed() * 2.0f);
    }
}

static void reference_attention(const uint16_t *query, const uint16_t *key,
                                const uint16_t *value, float *output) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float scores[SEQUENCE];
    for (unsigned row = 0; row < SEQUENCE; row++) {
        for (unsigned head = 0; head < QUERY_HEADS; head++) {
            unsigned kv_head = head / (QUERY_HEADS / KV_HEADS);
            float maximum = -INFINITY;
            for (unsigned key_row = 0; key_row <= row; key_row++) {
                float dot = 0.0f;
                for (unsigned d = 0; d < HEAD_DIM; d++) {
                    size_t query_index =
                        ((size_t)row * QUERY_HEADS + head) * HEAD_DIM + d;
                    size_t key_index =
                        ((size_t)key_row * KV_HEADS + kv_head) * HEAD_DIM + d;
                    float scaled_query =
                        bf16_to_f32(query[query_index]) * scale;
                    dot = fmaf(scaled_query, bf16_to_f32(key[key_index]), dot);
                }
                scores[key_row] = dot;
                if (dot > maximum) maximum = dot;
            }
            float sum = 0.0f;
            for (unsigned key_row = 0; key_row <= row; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                sum += scores[key_row];
            }
            for (unsigned d = 0; d < HEAD_DIM; d++) {
                float result = 0.0f;
                for (unsigned key_row = 0; key_row <= row; key_row++) {
                    size_t value_index =
                        ((size_t)key_row * KV_HEADS + kv_head) * HEAD_DIM + d;
                    result = fmaf(scores[key_row] / sum,
                                  bf16_to_f32(value[value_index]), result);
                }
                size_t output_index =
                    ((size_t)row * QUERY_HEADS + head) * HEAD_DIM + d;
                output[output_index] = result;
            }
        }
    }
}

int main(void) {
    const size_t query_count = (size_t)SEQUENCE * QUERY_HEADS * HEAD_DIM;
    const size_t kv_count = (size_t)SEQUENCE * KV_HEADS * HEAD_DIM;
    uint16_t *query = malloc(query_count * sizeof(*query));
    uint16_t *key = malloc(kv_count * sizeof(*key));
    uint16_t *value = malloc(kv_count * sizeof(*value));
    uint16_t *got = malloc(query_count * sizeof(*got));
    float *reference = malloc(query_count * sizeof(*reference));
    require(query && key && value && got && reference, "host allocation failed");

    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_gqa.c: Metal setup: %s\n", error);
        return 1;
    }

    double squared_error = 0.0;
    double maximum_absolute = 0.0;
    size_t mismatches = 0;
    size_t elements = 0;
    for (unsigned trial = 0; trial < TRIALS; trial++) {
        make_inputs(trial, query, key, value);
        reference_attention(query, key, value, reference);
        h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, query, query_count);
        h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, key, kv_count);
        h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, value, kv_count);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, query_count);
        require(q && k && v && output, "Metal tensor allocation failed");
        require_gpu(gpu, h3_gpu_begin(gpu), "begin command stream");
        require_gpu(gpu, h3_gpu_gqa_causal_bf16(
            gpu, output, q, k, v, SEQUENCE, QUERY_HEADS, KV_HEADS, HEAD_DIM,
            1.0f / sqrtf((float)HEAD_DIM)), "causal GQA");
        require_gpu(gpu, h3_gpu_submit(gpu), "submit command stream");
        require(h3_gpu_tensor_read_bf16(output, got, query_count),
                "cannot read GQA output");
        for (size_t index = 0; index < query_count; index++) {
            double delta = (double)bf16_to_f32(got[index]) - reference[index];
            double absolute = fabs(delta);
            if (absolute > maximum_absolute) maximum_absolute = absolute;
            squared_error += delta * delta;
            if (got[index] != f32_to_bf16(reference[index])) mismatches++;
        }
        elements += query_count;
        h3_gpu_tensor_free(output);
        h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(q);
    }

    double rmse = sqrt(squared_error / (double)elements);
    double mismatch_rate = (double)mismatches / (double)elements;
    printf("GQA F32-scaled reference: max abs %.7g, RMSE %.7g, "
           "BF16 mismatch %.5f%%\n", maximum_absolute, rmse,
           mismatch_rate * 100.0);
    require(maximum_absolute < 0.01,
            "scaled-query rounding exceeds maximum error tolerance");
    require(rmse < 0.0015,
            "scaled-query rounding exceeds RMSE tolerance");
    require(mismatch_rate < 0.001,
            "scaled-query rounding exceeds BF16 mismatch tolerance");

    h3_gpu_free(gpu);
    free(reference);
    free(got);
    free(value);
    free(key);
    free(query);
    puts("ok: production-width GQA keeps scaled queries at F32 precision");
    return 0;
}
