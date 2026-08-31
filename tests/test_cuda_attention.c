#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

static double seconds_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int benchmark_attention(uint32_t sequence) {
    enum { HEADS = 56, HEAD_DIM = 128, RUNS = 3 };
    size_t elements = (size_t)sequence * HEADS * HEAD_DIM;
    if (!sequence || sequence > 20000 || elements > SIZE_MAX / sizeof(float))
        return 1;
    float *zeros = calloc(elements, sizeof(*zeros));
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, elements);
    CHECK(zeros && gpu && query && key && value && output);
    CHECK(h3_gpu_tensor_write_f32(query, zeros, elements));
    CHECK(h3_gpu_tensor_write_f32(key, zeros, elements));
    CHECK(h3_gpu_tensor_write_f32(value, zeros, elements));
    for (int run = -1; run < RUNS; run++) {
        double started = seconds_now();
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_sdpa_bf16(gpu, output, query, key, value, sequence,
                               HEADS, HEAD_DIM, 1.0f / sqrtf(HEAD_DIM)));
        CHECK(h3_gpu_submit(gpu));
        double elapsed = seconds_now() - started;
        if (run >= 0)
            printf("attention sequence=%u run=%d seconds=%.6f\n",
                   sequence, run + 1, elapsed);
    }
    h3_gpu_tensor_free(output); h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key); h3_gpu_tensor_free(query);
    h3_gpu_free(gpu); free(zeros);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2) return benchmark_attention((uint32_t)strtoul(argv[1], NULL, 10));
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    const float zeros[] = {0,0,0,0};
    const float values[] = {1,2,3,4};
    const float noncausal_expected[] = {2,3,2,3};
    const float causal_expected[] = {1,2,2,3};
    float actual[16];
    h3_gpu_tensor *query = h3_gpu_tensor_from_f32(gpu, zeros, 4);
    h3_gpu_tensor *key = h3_gpu_tensor_from_f32(gpu, zeros, 4);
    h3_gpu_tensor *value = h3_gpu_tensor_from_f32(gpu, values, 4);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(query && key && value && output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_f32(gpu, output, query, key, value, 2, 1, 2, 1.0f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output, actual, 4));
    CHECK(close_values(actual, noncausal_expected, 4, 1e-6f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_causal_f32(gpu, output, query, key, value,
                                 1, 2, 1, 2, 1.0f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(output, actual, 4));
    CHECK(close_values(actual, causal_expected, 4, 1e-6f));

    float wide_values[1024] = {0};
    float wide_actual[1024];
    for (size_t index = 0; index < 1024; index++)
        wide_values[index] = (float)index;
    h3_gpu_tensor *wide_query = h3_gpu_tensor_new_f32(gpu, 1024);
    h3_gpu_tensor *wide_key = h3_gpu_tensor_new_f32(gpu, 1024);
    h3_gpu_tensor *wide_value = h3_gpu_tensor_from_f32(gpu, wide_values, 1024);
    h3_gpu_tensor *wide_output = h3_gpu_tensor_new_f32(gpu, 1024);
    CHECK(wide_query && wide_key && wide_value && wide_output);
    CHECK(h3_gpu_tensor_write_f32(wide_query, (const float[1024]){0}, 1024));
    CHECK(h3_gpu_tensor_write_f32(wide_key, (const float[1024]){0}, 1024));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_causal_f32(gpu, wide_output, wide_query, wide_key,
                                 wide_value, 2, 2, 1, 256, 0.0625f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(wide_output, wide_actual, 1024));
    CHECK(close_values(wide_actual, wide_values, 256, 1e-6f));
    CHECK(close_values(wide_actual + 512, wide_values + 512, 256, 1e-6f));
    for (size_t index = 256; index < 512; index++)
        CHECK(fabsf(wide_actual[index] -
                    0.5f * (wide_values[index - 256] + wide_values[index])) <
              1e-6f);
    for (size_t index = 768; index < 1024; index++)
        CHECK(fabsf(wide_actual[index] -
                    0.5f * (wide_values[index - 256] + wide_values[index])) <
              1e-6f);

    const float two_head_values[] = {1,2,3,4, 10,20,30,40};
    const float row_major_expected[] = {2,3,20,30, 2,3,20,30};
    const float head_major_expected[] = {2,3,2,3, 20,30,20,30};
    h3_gpu_tensor *bquery = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *bkey = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *bvalue = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *boutput = h3_gpu_tensor_new_bf16(gpu, 8);
    CHECK(bquery && bkey && bvalue && boutput);
    CHECK(h3_gpu_tensor_write_f32(bquery, two_head_values, 8));
    CHECK(h3_gpu_tensor_write_f32(bkey, (const float[8]){0}, 8));
    CHECK(h3_gpu_tensor_write_f32(bvalue, two_head_values, 8));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(gpu, boutput, bquery, bkey, bvalue,
                           2, 2, 2, 0.5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(boutput, actual, 8));
    CHECK(close_values(actual, row_major_expected, 8, 0.02f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16_head_major_output(
        gpu, boutput, bquery, bkey, bvalue, 2, 2, 2, 0.5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(boutput, actual, 8));
    CHECK(close_values(actual, head_major_expected, 8, 0.02f));

    enum { TSEQ = 3, THEADS = 2, TDIM = 128, TELEMS = TSEQ * THEADS * TDIM };
    float tiled_query[TELEMS], tiled_key[TELEMS], tiled_value[TELEMS];
    float tiled_expected[TELEMS], tiled_head_expected[TELEMS];
    float tiled_actual[TELEMS];
    for (int index = 0; index < TELEMS; index++) {
        tiled_query[index] = (float)(index % 17 - 8) * 0.01f;
        tiled_key[index] = (float)(index % 13 - 6) * 0.0125f;
        tiled_value[index] = (float)(index % 29 - 14) * 0.025f;
    }
    for (int head = 0; head < THEADS; head++)
        for (int row = 0; row < TSEQ; row++) {
            float scores[TSEQ], maximum = -INFINITY, denominator = 0.0f;
            for (int key_row = 0; key_row < TSEQ; key_row++) {
                float score = 0.0f;
                for (int dimension = 0; dimension < TDIM; dimension++)
                    score += tiled_query[(head * TSEQ + row) * TDIM + dimension] *
                             tiled_key[(head * TSEQ + key_row) * TDIM + dimension];
                scores[key_row] = score / sqrtf((float)TDIM);
                maximum = fmaxf(maximum, scores[key_row]);
            }
            for (int key_row = 0; key_row < TSEQ; key_row++)
                denominator += expf(scores[key_row] - maximum);
            for (int dimension = 0; dimension < TDIM; dimension++) {
                float sum = 0.0f;
                for (int key_row = 0; key_row < TSEQ; key_row++)
                    sum += expf(scores[key_row] - maximum) *
                           tiled_value[(head * TSEQ + key_row) * TDIM + dimension];
                tiled_expected[(row * THEADS + head) * TDIM + dimension] =
                    sum / denominator;
                tiled_head_expected[(head * TSEQ + row) * TDIM + dimension] =
                    sum / denominator;
            }
        }
    h3_gpu_tensor *tiled_q = h3_gpu_tensor_new_bf16(gpu, TELEMS);
    h3_gpu_tensor *tiled_k = h3_gpu_tensor_new_bf16(gpu, TELEMS);
    h3_gpu_tensor *tiled_v = h3_gpu_tensor_new_bf16(gpu, TELEMS);
    h3_gpu_tensor *tiled_o = h3_gpu_tensor_new_bf16(gpu, TELEMS);
    CHECK(tiled_q && tiled_k && tiled_v && tiled_o);
    CHECK(h3_gpu_tensor_write_f32(tiled_q, tiled_query, TELEMS));
    CHECK(h3_gpu_tensor_write_f32(tiled_k, tiled_key, TELEMS));
    CHECK(h3_gpu_tensor_write_f32(tiled_v, tiled_value, TELEMS));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(gpu, tiled_o, tiled_q, tiled_k, tiled_v,
                           TSEQ, THEADS, TDIM, 1.0f / sqrtf((float)TDIM)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(tiled_o, tiled_actual, TELEMS));
    CHECK(close_values(tiled_actual, tiled_expected, TELEMS, 0.002f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16_head_major_output(
        gpu, tiled_o, tiled_q, tiled_k, tiled_v, TSEQ, THEADS, TDIM,
        1.0f / sqrtf((float)TDIM)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(tiled_o, tiled_actual, TELEMS));
    CHECK(close_values(tiled_actual, tiled_head_expected, TELEMS, 0.002f));

    enum { FSEQ = 13, FHEADS = 2, FDIM = 64, FELEMS = FSEQ * FHEADS * FDIM };
    float tiled_f32_query[FELEMS], tiled_f32_key[FELEMS];
    float tiled_f32_value[FELEMS], tiled_f32_expected[FELEMS];
    float tiled_f32_actual[FELEMS], tiled_f32_scalar[FELEMS];
    for (int index = 0; index < FELEMS; index++) {
        tiled_f32_query[index] = (float)(index % 17 - 8) * 0.01f;
        tiled_f32_key[index] = (float)(index % 13 - 6) * 0.0125f;
        tiled_f32_value[index] = (float)(index % 29 - 14) * 0.025f;
    }
    for (int head = 0; head < FHEADS; head++)
        for (int row = 0; row < FSEQ; row++) {
            float scores[FSEQ], maximum = -INFINITY, denominator = 0.0f;
            for (int key_row = 0; key_row < FSEQ; key_row++) {
                float score = 0.0f;
                for (int dimension = 0; dimension < FDIM; dimension++)
                    score += tiled_f32_query[(head * FSEQ + row) * FDIM +
                                             dimension] *
                             tiled_f32_key[(head * FSEQ + key_row) * FDIM +
                                           dimension];
                scores[key_row] = score / sqrtf((float)FDIM);
                maximum = fmaxf(maximum, scores[key_row]);
            }
            for (int key_row = 0; key_row < FSEQ; key_row++)
                denominator += expf(scores[key_row] - maximum);
            for (int dimension = 0; dimension < FDIM; dimension++) {
                float sum = 0.0f;
                for (int key_row = 0; key_row < FSEQ; key_row++)
                    sum += expf(scores[key_row] - maximum) *
                           tiled_f32_value[(head * FSEQ + key_row) * FDIM +
                                           dimension];
                tiled_f32_expected[(row * FHEADS + head) * FDIM + dimension] =
                    sum / denominator;
            }
        }
    h3_gpu_tensor *f32_q = h3_gpu_tensor_new_f32(gpu, FELEMS);
    h3_gpu_tensor *f32_k = h3_gpu_tensor_new_f32(gpu, FELEMS);
    h3_gpu_tensor *f32_v = h3_gpu_tensor_new_f32(gpu, FELEMS);
    h3_gpu_tensor *f32_o = h3_gpu_tensor_new_f32(gpu, FELEMS);
    CHECK(f32_q && f32_k && f32_v && f32_o);
    CHECK(h3_gpu_tensor_write_f32(f32_q, tiled_f32_query, FELEMS));
    CHECK(h3_gpu_tensor_write_f32(f32_k, tiled_f32_key, FELEMS));
    CHECK(h3_gpu_tensor_write_f32(f32_v, tiled_f32_value, FELEMS));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_f32(gpu, f32_o, f32_q, f32_k, f32_v,
                          FSEQ, FHEADS, FDIM, 1.0f / sqrtf((float)FDIM)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(f32_o, tiled_f32_actual, FELEMS));
    CHECK(close_values(tiled_f32_actual, tiled_f32_expected, FELEMS, 2e-5f));
    CHECK(setenv("H3_DISABLE_TILED_ATTENTION", "1", 1) == 0);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_f32(gpu, f32_o, f32_q, f32_k, f32_v,
                          FSEQ, FHEADS, FDIM, 1.0f / sqrtf((float)FDIM)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(f32_o, tiled_f32_scalar, FELEMS));
    CHECK(unsetenv("H3_DISABLE_TILED_ATTENTION") == 0);
    CHECK(close_values(tiled_f32_scalar, tiled_f32_expected, FELEMS, 2e-5f));
    CHECK(close_values(tiled_f32_scalar, tiled_f32_actual, FELEMS, 2e-5f));

    /* Video VAE limit shape: sequence 2805, head_dim 64. */
    enum { LSEQ = 2805, LHEADS = 4, LDIM = 64, LELEMS = LSEQ * LHEADS * LDIM };
    float *long_query = malloc(LELEMS * sizeof(float));
    float *long_key = malloc(LELEMS * sizeof(float));
    float *long_value = malloc(LELEMS * sizeof(float));
    float *long_expected = malloc(LELEMS * sizeof(float));
    float *long_actual = malloc(LELEMS * sizeof(float));
    float *long_scores = malloc(LSEQ * sizeof(float));
    CHECK(long_query && long_key && long_value && long_expected &&
          long_actual && long_scores);
    for (int index = 0; index < LELEMS; index++) {
        long_query[index] = (float)(index % 31 - 15) * 0.005f;
        long_key[index] = (float)(index % 19 - 9) * 0.0075f;
        long_value[index] = (float)(index % 47 - 23) * 0.0125f;
    }
    for (int head = 0; head < LHEADS; head++)
        for (int row = 0; row < LSEQ; row++) {
            float maximum = -INFINITY, denominator = 0.0f;
            for (int key_row = 0; key_row < LSEQ; key_row++) {
                float score = 0.0f;
                for (int dimension = 0; dimension < LDIM; dimension++)
                    score += long_query[(head * LSEQ + row) * LDIM +
                                        dimension] *
                             long_key[(head * LSEQ + key_row) * LDIM +
                                      dimension];
                long_scores[key_row] = score / sqrtf((float)LDIM);
                maximum = fmaxf(maximum, long_scores[key_row]);
            }
            for (int key_row = 0; key_row < LSEQ; key_row++)
                denominator += expf(long_scores[key_row] - maximum);
            for (int dimension = 0; dimension < LDIM; dimension++) {
                float sum = 0.0f;
                for (int key_row = 0; key_row < LSEQ; key_row++)
                    sum += expf(long_scores[key_row] - maximum) *
                           long_value[(head * LSEQ + key_row) * LDIM +
                                      dimension];
                long_expected[(row * LHEADS + head) * LDIM + dimension] =
                    sum / denominator;
            }
        }
    h3_gpu_tensor *long_q = h3_gpu_tensor_new_f32(gpu, LELEMS);
    h3_gpu_tensor *long_k = h3_gpu_tensor_new_f32(gpu, LELEMS);
    h3_gpu_tensor *long_v = h3_gpu_tensor_new_f32(gpu, LELEMS);
    h3_gpu_tensor *long_o = h3_gpu_tensor_new_f32(gpu, LELEMS);
    CHECK(long_q && long_k && long_v && long_o);
    CHECK(h3_gpu_tensor_write_f32(long_q, long_query, LELEMS));
    CHECK(h3_gpu_tensor_write_f32(long_k, long_key, LELEMS));
    CHECK(h3_gpu_tensor_write_f32(long_v, long_value, LELEMS));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_f32(gpu, long_o, long_q, long_k, long_v,
                          LSEQ, LHEADS, LDIM, 1.0f / sqrtf((float)LDIM)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(long_o, long_actual, LELEMS));
    CHECK(close_values(long_actual, long_expected, LELEMS, 2e-5f));
    h3_gpu_tensor_free(long_o); h3_gpu_tensor_free(long_v);
    h3_gpu_tensor_free(long_k); h3_gpu_tensor_free(long_q);
    free(long_scores); free(long_actual);
    free(long_expected); free(long_value); free(long_key); free(long_query);
    h3_gpu_tensor_free(f32_o); h3_gpu_tensor_free(f32_v);
    h3_gpu_tensor_free(f32_k); h3_gpu_tensor_free(f32_q);

    h3_gpu_tensor *gqa_query = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *gqa_key = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *gqa_value = h3_gpu_tensor_new_bf16(gpu, 8);
    CHECK(gqa_query && gqa_key && gqa_value);
    CHECK(h3_gpu_tensor_write_f32(gqa_query, (const float[8]){0}, 8));
    CHECK(h3_gpu_tensor_write_f32(gqa_key, (const float[8]){0}, 8));
    CHECK(h3_gpu_tensor_write_f32(gqa_value,
        (const float[]){1,2,10,20, 3,4,30,40}, 8));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_gqa_causal_bf16(gpu, boutput, gqa_query, gqa_key,
                                  gqa_value, 2, 2, 2, 2, 1.0f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(boutput, actual, 8));
    const float gqa_expected[] = {1,2,10,20, 2,3,20,30};
    CHECK(close_values(actual, gqa_expected, 8, 0.02f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(!h3_gpu_gqa_causal_bf16(gpu, boutput, gqa_query, gqa_key,
                                   gqa_value, 2, 3, 2, 2, 1.0f));
    CHECK(h3_gpu_error(gpu)[0] != '\0');
    CHECK(h3_gpu_submit(gpu));

    h3_gpu_tensor_free(gqa_value);
    h3_gpu_tensor_free(gqa_key);
    h3_gpu_tensor_free(gqa_query);
    h3_gpu_tensor_free(tiled_o); h3_gpu_tensor_free(tiled_v);
    h3_gpu_tensor_free(tiled_k); h3_gpu_tensor_free(tiled_q);
    h3_gpu_tensor_free(wide_output);
    h3_gpu_tensor_free(wide_value);
    h3_gpu_tensor_free(wide_key);
    h3_gpu_tensor_free(wide_query);
    h3_gpu_tensor_free(boutput);
    h3_gpu_tensor_free(bvalue);
    h3_gpu_tensor_free(bkey);
    h3_gpu_tensor_free(bquery);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_free(gpu);
    puts("ok: CUDA SDPA causal, GQA and head-major layouts");
    return 0;
}
