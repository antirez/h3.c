#include "h3_gpu.h"
#include "h3_host.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { QUERY_HEADS = 32, KV_HEADS = 8, HEAD_DIM = 128, RUNS = 30 };

static const double GQA_REL_L2_BOUND = 0.0063;
static const double GQA_SCALE = 0.08838834764831845;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_gpu_determinism.c: %s\n", message);
    exit(1);
}

static h3_gpu *gpu;
static int failures;

static void gpu_ok(int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_gpu_determinism.c: %s: %s\n",
                operation, h3_gpu_error(gpu));
        exit(1);
    }
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static double from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return (double)result;
}

static void fill_bf16(h3_rng *rng, uint16_t *values, size_t count,
                      float scale) {
    for (size_t index = 0; index < count; index++)
        values[index] = to_bf16(
            ((float)(h3_rng_u32(rng) >> 8) / 8388608.0f - 1.0f) * scale);
}

/* Reference rows are sampled: every `stride`-th row plus the last one. */
static double reference_error(const uint16_t *query, const uint16_t *key,
                              const uint16_t *value, const uint16_t *got,
                              uint32_t sequence, uint32_t stride) {
    double square_error = 0.0, square_value = 0.0;
    double *probability = malloc(sequence * sizeof(*probability));
    if (!probability) die("out of memory");
    for (uint32_t row = 0; row < sequence; row++) {
        if (row % stride && row != sequence - 1) continue;
        for (uint32_t head = 0; head < QUERY_HEADS; head++) {
            uint32_t kv_head = head / (QUERY_HEADS / KV_HEADS);
            const uint16_t *q = query + ((size_t)row * QUERY_HEADS + head) *
                                            HEAD_DIM;
            double maximum = -INFINITY, sum = 0.0;
            for (uint32_t k = 0; k <= row; k++) {
                const uint16_t *key_row =
                    key + ((size_t)k * KV_HEADS + kv_head) * HEAD_DIM;
                double dot = 0.0;
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    dot += from_bf16(q[d]) * GQA_SCALE * from_bf16(key_row[d]);
                probability[k] = dot;
                if (dot > maximum) maximum = dot;
            }
            for (uint32_t k = 0; k <= row; k++) {
                probability[k] = exp(probability[k] - maximum);
                sum += probability[k];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                double expected = 0.0;
                for (uint32_t k = 0; k <= row; k++)
                    expected += probability[k] / sum *
                                from_bf16(value[((size_t)k * KV_HEADS +
                                                 kv_head) * HEAD_DIM + d]);
                double actual = from_bf16(
                    got[((size_t)row * QUERY_HEADS + head) * HEAD_DIM + d]);
                square_error += (actual - expected) * (actual - expected);
                square_value += expected * expected;
            }
        }
    }
    free(probability);
    return sqrt(square_error / square_value);
}

static void test_gqa(uint32_t sequence, uint32_t reference_stride) {
    size_t query_count = (size_t)sequence * QUERY_HEADS * HEAD_DIM;
    size_t kv_count = (size_t)sequence * KV_HEADS * HEAD_DIM;
    uint16_t *query = malloc(query_count * sizeof(*query));
    uint16_t *key = malloc(kv_count * sizeof(*key));
    uint16_t *value = malloc(kv_count * sizeof(*value));
    uint16_t *first = malloc(query_count * sizeof(*first));
    uint16_t *again = malloc(query_count * sizeof(*again));
    if (!query || !key || !value || !first || !again) die("out of memory");
    h3_rng rng;
    h3_rng_seed(&rng, 20260512u + sequence);
    fill_bf16(&rng, query, query_count, 4.0f);
    fill_bf16(&rng, key, kv_count, 4.0f);
    fill_bf16(&rng, value, kv_count, 1.0f);

    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, query, query_count);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, key, kv_count);
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, value, kv_count);
    h3_gpu_tensor *to = h3_gpu_tensor_new_bf16(gpu, query_count);
    if (!tq || !tk || !tv || !to) die("tensor allocation failed");
    int diverged = 0;
    size_t worst = 0;
    for (int run = 0; run < RUNS; run++) {
        gpu_ok(h3_gpu_begin(gpu), "begin");
        gpu_ok(h3_gpu_gqa_causal_bf16(gpu, to, tq, tk, tv, sequence,
                                      QUERY_HEADS, KV_HEADS, HEAD_DIM,
                                      (float)GQA_SCALE), "GQA attention");
        gpu_ok(h3_gpu_submit(gpu), "submit");
        uint16_t *destination = run == 0 ? first : again;
        if (!h3_gpu_tensor_read_bf16(to, destination, query_count))
            die("cannot read result");
        if (run == 0) continue;
        size_t differing = 0;
        for (size_t index = 0; index < query_count; index++)
            differing += first[index] != again[index];
        if (differing) diverged++;
        if (differing > worst) worst = differing;
    }
    double error = reference_error(query, key, value, first, sequence,
                                   reference_stride);
    printf("GQA S=%u: %d of %d repeat runs differ from run 0 (worst %zu "
           "elements); rel-L2 vs double reference %.3g\n",
           sequence, diverged, RUNS - 1, worst, error);
    if (diverged) {
        fprintf(stderr, "FAIL: GQA attention changes between identical runs\n");
        failures++;
    }
    if (error > GQA_REL_L2_BOUND) {
        fprintf(stderr, "FAIL: GQA attention is far from the reference\n");
        failures++;
    }
    h3_gpu_tensor_free(tq);
    h3_gpu_tensor_free(tk);
    h3_gpu_tensor_free(tv);
    h3_gpu_tensor_free(to);
    free(query);
    free(key);
    free(value);
    free(first);
    free(again);
}

int main(int argc, char **argv) {
    char error[512];
    gpu = h3_gpu_create(argc > 1 ? argv[1] : "h3_shaders.metal", error,
                        sizeof(error));
    if (!gpu) die(error);
    test_gqa(1024, 8);
    test_gqa(2048, 16);
    h3_gpu_free(gpu);
    if (failures) return 1;
    puts("ok: GPU kernels are deterministic across repeated dispatches");
    return 0;
}
