/* Opt-in large-sequence BF16 SDPA regression probe; no model weights needed.
 * Compare first/middle/last rows and heads against a double-precision CPU
 * reference using the exact BF16-rounded inputs supplied to the GPU. */
#include "h3_gpu.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 42;

static uint16_t random_bf16(void) {
    rng_state = 1664525 * rng_state + 1013904223;
    float value = ((float)(rng_state >> 8) / 16777216.0f - 0.5f) * 2.0f;
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static float as_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(int argc, char **argv) {
    unsigned sequence = 12256;
    if (argc > 2) return 2;
    if (argc == 2) {
        char *end;
        errno = 0;
        unsigned long parsed = strtoul(argv[1], &end, 10);
        if (errno || end == argv[1] || *end || parsed < 1 || parsed > 32768) {
            fprintf(stderr, "sequence must be in [1, 32768]\n");
            return 2;
        }
        sequence = (unsigned)parsed;
    }
    enum { HEADS = 56, DIM = 128 };
    size_t count = (size_t)sequence * HEADS * DIM;
    int result = 2;
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return result; }
    uint16_t *q = malloc(count * sizeof(*q));
    uint16_t *k = malloc(count * sizeof(*k));
    uint16_t *v = malloc(count * sizeof(*v));
    uint16_t *output = malloc(count * sizeof(*output));
    double *weights = malloc(sequence * sizeof(*weights));
    h3_gpu_tensor *query = NULL, *key = NULL, *value = NULL, *actual = NULL;
    if (!q || !k || !v || !output || !weights) goto cleanup;
    for (size_t i = 0; i < count; i++) {
        q[i] = random_bf16(); k[i] = random_bf16(); v[i] = random_bf16();
        output[i] = 0x7fc0; /* Detect unwritten output as well as wrong values. */
    }
    query = h3_gpu_tensor_from_bf16(gpu, q, count);
    key = h3_gpu_tensor_from_bf16(gpu, k, count);
    value = h3_gpu_tensor_from_bf16(gpu, v, count);
    actual = h3_gpu_tensor_from_bf16(gpu, output, count);
    if (!query || !key || !value || !actual || !h3_gpu_begin(gpu) ||
        !h3_gpu_sdpa_bf16(gpu, actual, query, key, value, sequence,
                          HEADS, DIM, 1.0f / sqrtf(DIM)) ||
        !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_bf16(actual, output, count)) goto cleanup;
    unsigned rows[] = {0, sequence / 2, sequence - 1};
    unsigned heads[] = {0, HEADS / 2, HEADS - 1};
    double max_error = 0, reference_squared = 0, error_squared = 0;
    unsigned nonfinite = 0;
    for (unsigned ri = 0; ri < 3; ri++) for (unsigned hi = 0; hi < 3; hi++) {
        unsigned row = rows[ri], head = heads[hi];
        double sum = 0, local_error = 0;
        for (unsigned j = 0; j < sequence; j++) {
            double dot = 0;
            for (unsigned c = 0; c < DIM; c++)
                dot += (double)as_float(q[((size_t)row * HEADS + head) * DIM + c]) *
                    as_float(k[((size_t)j * HEADS + head) * DIM + c]);
            /* Inputs lie in [-1,1], so these exponentials cannot overflow. */
            weights[j] = exp(dot / sqrt(DIM));
            sum += weights[j];
        }
        for (unsigned c = 0; c < DIM; c++) {
            double reference = 0;
            for (unsigned j = 0; j < sequence; j++)
                reference += weights[j] *
                    as_float(v[((size_t)j * HEADS + head) * DIM + c]);
            reference /= sum;
            double got = as_float(output[((size_t)row * HEADS + head) * DIM + c]);
            double difference = fabs(reference - got);
            if (!isfinite(got)) { nonfinite++; difference = INFINITY; }
            local_error = fmax(local_error, difference);
            max_error = fmax(max_error, difference);
            error_squared += difference * difference;
            reference_squared += reference * reference;
        }
        printf("sequence=%u row=%u head=%u max_abs=%.7g\n",
               sequence, row, head, local_error);
    }
    double relative_rms = sqrt(error_squared / reference_squared);
    result = max_error < 0.001 && relative_rms < 0.02 && !nonfinite ? 0 : 1;
    printf("RESULT sequence=%u max_abs=%.7g relative_rms=%.7g nonfinite=%u %s\n",
           sequence, max_error, relative_rms, nonfinite, result ? "FAIL" : "PASS");
cleanup:
    if (result == 2) fprintf(stderr, "allocation or GPU operation failed\n");
    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value); h3_gpu_tensor_free(actual);
    h3_gpu_free(gpu);
    free(q); free(k); free(v); free(output); free(weights);
    return result;
}
