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
                     int head_major, int flash, uint32_t window_frames,
                     uint32_t frame_rows, uint32_t global_rows) {
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
    uint32_t frames = 0;
    h3_gpu_tensor *pack_q = NULL, *pack_k = NULL, *pack_v = NULL,
                  *pack_o = NULL;
    if (window_frames) {
        if (global_rows >= sequence ||
            (sequence - global_rows) % frame_rows) {
            fprintf(stderr, "N=%u incompatible with --frame-rows/--globals\n",
                    sequence);
            return 0;
        }
        frames = (sequence - global_rows) / frame_rows;
        uint32_t kv_frames = frames < 3 * window_frames ? frames :
            3 * window_frames;
        uint32_t q_pack = window_frames * frame_rows;
        if (global_rows > q_pack) q_pack = global_rows;
        size_t q_elements = (size_t)q_pack * H3_BENCH_HEADS *
            H3_BENCH_HEAD_DIM;
        size_t kv_elements = ((size_t)kv_frames * frame_rows + global_rows) *
            H3_BENCH_HEADS * H3_BENCH_HEAD_DIM;
        pack_q = h3_gpu_tensor_new_bf16(gpu, q_elements);
        pack_k = h3_gpu_tensor_new_bf16(gpu, kv_elements);
        pack_v = h3_gpu_tensor_new_bf16(gpu, kv_elements);
        pack_o = h3_gpu_tensor_new_bf16(gpu, q_elements);
        if (!pack_q || !pack_k || !pack_v || !pack_o) {
            fprintf(stderr, "windowed scratch allocation failed\n");
            return 0;
        }
    }
#define H3_BENCH_RUN() \
    (window_frames ? h3_gpu_sdpa_windowed_bf16_head_major( \
         gpu, output, query, key, value, pack_q, pack_k, pack_v, pack_o, \
         sequence, H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale, \
         global_rows, frames, frame_rows, window_frames) : \
     flash ? h3_gpu_sdpa_flash_bf16_head_major( \
         gpu, output, query, key, value, sequence, \
         H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale) : \
     head_major ? h3_gpu_sdpa_bf16_head_major_io( \
         gpu, output, query, key, value, sequence, \
         H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale) : \
     h3_gpu_sdpa_bf16( \
         gpu, output, query, key, value, sequence, \
         H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale))
    for (int run = 0; run < warmup; run++) {
        int ok = h3_gpu_begin(gpu) && H3_BENCH_RUN() && h3_gpu_submit(gpu);
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
        int ok = h3_gpu_begin(gpu) && H3_BENCH_RUN() && h3_gpu_submit(gpu);
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
    double l2 = -1.0;
    if (finite && (flash || window_frames) && sequence <= 8000) {
        h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, elements);
        int ok = reference && h3_gpu_begin(gpu) &&
            h3_gpu_sdpa_bf16_head_major_io(
                gpu, reference, query, key, value, sequence,
                H3_BENCH_HEADS, H3_BENCH_HEAD_DIM, scale) &&
            h3_gpu_submit(gpu);
        if (ok) {
            enum { CHUNK = 1 << 20 };
            uint16_t *a = malloc(CHUNK * sizeof(*a));
            uint16_t *b = malloc(CHUNK * sizeof(*b));
            double difference = 0.0, magnitude = 0.0;
            size_t offset = 0;
            while (ok && a && b && offset < elements) {
                size_t count = elements - offset;
                if (count > CHUNK) count = CHUNK;
                ok = h3_gpu_tensor_read_bf16_range(output, offset, a, count) &&
                     h3_gpu_tensor_read_bf16_range(reference, offset, b,
                                                   count);
                for (size_t index = 0; ok && index < count; index++) {
                    double flash_value = bf16_to_f32(a[index]);
                    double mps_value = bf16_to_f32(b[index]);
                    difference += (flash_value - mps_value) *
                                  (flash_value - mps_value);
                    magnitude += mps_value * mps_value;
                }
                offset += count;
            }
            if (ok && magnitude > 0.0) l2 = sqrt(difference / magnitude);
            free(a); free(b);
        }
        h3_gpu_tensor_free(reference);
    }
    double buffer_gib = (double)elements * sizeof(uint16_t) * 4.0 /
        (1024.0 * 1024.0 * 1024.0);
    printf("%u\t%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%s", 
           sequence,
           window_frames ? "window" : flash ? "flash" :
               head_major ? "head" : "row", repeats,
           buffer_gib,
           wall * 1000.0 / repeats,
           (after.command_wait_seconds - before.command_wait_seconds) *
               1000.0 / repeats,
           (after.gpu_seconds - before.gpu_seconds) * 1000.0 / repeats,
           finite ? "finite" : "nonfinite");
    if (l2 >= 0.0) printf("\tL2_vs_mps=%.3e", l2);
    putchar('\n');
    fflush(stdout);
    h3_gpu_tensor_free(pack_q); h3_gpu_tensor_free(pack_k);
    h3_gpu_tensor_free(pack_v); h3_gpu_tensor_free(pack_o);
#undef H3_BENCH_RUN
    fflush(stdout);
    h3_gpu_tensor_free(query); h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value); h3_gpu_tensor_free(output);
    return finite;
}

int main(int argc, char **argv) {
    int warmup = 1;
    int repeats = 3;
    int head_major = 1;
    int flash = 0;
    int window = 0;
    int frame_rows = 1000;
    int global_rows = 2000;
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
        } else if (!strcmp(argv[index], "--flash")) {
            flash = 1;
        } else if (!strcmp(argv[index], "--window") && index + 1 < argc) {
            if (!parse_positive(argv[++index], &window)) return 2;
        } else if (!strcmp(argv[index], "--frame-rows") && index + 1 < argc) {
            if (!parse_positive(argv[++index], &frame_rows)) return 2;
        } else if (!strcmp(argv[index], "--globals") && index + 1 < argc) {
            if (!parse_positive(argv[++index], &global_rows)) return 2;
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
        if (!benchmark(gpu, sequences[index], warmup, repeats, head_major,
                       flash, (uint32_t)window, (uint32_t)frame_rows,
                       (uint32_t)global_rows)) {
            ok = 0;
            break;
        }
    h3_gpu_free(gpu);
    return ok ? 0 : 1;
}
