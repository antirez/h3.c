#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { H3_BENCH_HEADS = 56, H3_BENCH_HEAD_DIM = 128 };

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
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

static int fill_tensor(h3_gpu_tensor *tensor, size_t elements,
                       unsigned salt) {
    enum { CHUNK = 1 << 20 };
    uint16_t *values = malloc(CHUNK * sizeof(*values));
    if (!values) return 0;
    size_t offset = 0;
    while (offset < elements) {
        size_t count = elements - offset;
        if (count > CHUNK) count = CHUNK;
        for (size_t index = 0; index < count; index++) {
            unsigned pattern = (unsigned)((offset + index + salt) % 257u);
            values[index] = f32_to_bf16(((float)pattern - 128.0f) / 512.0f);
        }
        if (!h3_gpu_tensor_write_bf16_range(
                tensor, offset, values, count)) {
            free(values);
            return 0;
        }
        offset += count;
    }
    free(values);
    return 1;
}

static int parse_positive(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || end == text || !end || *end || parsed < 1 ||
        parsed > INT32_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static int benchmark(h3_gpu *gpu, uint32_t sequence, int warmup, int repeats,
                     int head_major) {
    size_t width = (size_t)H3_BENCH_HEADS * H3_BENCH_HEAD_DIM;
    if (sequence > SIZE_MAX / width) return 0;
    size_t elements = (size_t)sequence * width;
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!query || !key || !value || !output ||
        !fill_tensor(query, elements, 0) ||
        !fill_tensor(key, elements, 31) ||
        !fill_tensor(value, elements, 67)) {
        fprintf(stderr, "SDPA allocation/input initialization failed at N=%u\n",
                sequence);
        h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
        h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
        return 0;
    }
    float scale = 1.0f / sqrtf((float)H3_BENCH_HEAD_DIM);
    for (int run = 0; run < warmup; run++) {
        int ok = h3_gpu_begin(gpu) &&
            (head_major ? h3_gpu_sdpa_bf16_head_major_io(
                 gpu, output, query, key, value, sequence,
                 H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale) :
             h3_gpu_sdpa_bf16(
                 gpu, output, query, key, value, sequence,
                 H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale)) &&
            h3_gpu_submit(gpu);
        if (!ok) {
            fprintf(stderr, "SDPA warmup failed at N=%u: %s\n",
                    sequence, h3_gpu_error(gpu));
            h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
            h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
            return 0;
        }
    }
    h3_gpu_stats before, after;
    if (!h3_gpu_get_stats(gpu, &before)) return 0;
    double started = now_seconds();
    for (int run = 0; run < repeats; run++) {
        int ok = h3_gpu_begin(gpu) &&
            (head_major ? h3_gpu_sdpa_bf16_head_major_io(
                 gpu, output, query, key, value, sequence,
                 H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale) :
             h3_gpu_sdpa_bf16(
                 gpu, output, query, key, value, sequence,
                 H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale)) &&
            h3_gpu_submit(gpu);
        if (!ok) {
            fprintf(stderr, "SDPA run failed at N=%u: %s\n",
                    sequence, h3_gpu_error(gpu));
            h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
            h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
            return 0;
        }
    }
    double wall = now_seconds() - started;
    if (!h3_gpu_get_stats(gpu, &after)) return 0;
    uint16_t sample[16];
    int finite = h3_gpu_tensor_read_bf16(output, sample, 16);
    for (size_t index = 0; finite && index < 16; index++)
        finite = isfinite(bf16_to_f32(sample[index]));
    double buffer_gib = (double)elements * sizeof(uint16_t) * 4.0 /
        (1024.0 * 1024.0 * 1024.0);
    printf("%u\t%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n",
           sequence, head_major ? "head" : "row", repeats, buffer_gib,
           wall * 1000.0 / repeats,
           (after.command_wait_seconds - before.command_wait_seconds) *
               1000.0 / repeats,
           (after.gpu_seconds - before.gpu_seconds) * 1000.0 / repeats,
           finite ? "finite" : "nonfinite");
    fflush(stdout);
    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
    return finite;
}

int main(int argc, char **argv) {
    int warmup = 1;
    int repeats = 3;
    int head_major = 1;
    uint32_t sequences[32] = {2000, 8000, 20000, 38000};
    size_t sequence_count = 4;
    size_t explicit_count = 0;
    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--warmup") && index + 1 < argc) {
            if (!parse_positive(argv[++index], &warmup)) return 2;
        } else if (!strcmp(argv[index], "--repeats") && index + 1 < argc) {
            if (!parse_positive(argv[++index], &repeats)) return 2;
        } else if (!strcmp(argv[index], "--row-major")) {
            head_major = 0;
        } else {
            int parsed;
            if (!parse_positive(argv[index], &parsed) || explicit_count == 32)
                return 2;
            sequences[explicit_count++] = (uint32_t)parsed;
        }
    }
    if (explicit_count) sequence_count = explicit_count;
    char error[512];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create Metal context: %s\n", error);
        return 1;
    }
    puts("sequence\tlayout\trepeats\tbuffers_GiB\twall_ms\twait_ms\troot_gpu_ms\toutput");
    int ok = 1;
    for (size_t index = 0; index < sequence_count; index++)
        if (!benchmark(gpu, sequences[index], warmup, repeats, head_major)) {
            ok = 0;
            break;
        }
    h3_gpu_free(gpu);
    return ok ? 0 : 1;
}
