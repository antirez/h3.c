/* Correctness and timing of the dense tiled TensorOps attention against the
 * MPSGraph SDPA path on random BF16 Q/K/V of shape [1, seq, 56, 128].
 *
 *   ./h3_attn_tiled_bench [--time] [--reps N] [--qscale S] [seq ...]
 *
 * Without --time it only checks correctness (safe while the GPU is shared).
 * H3_ATTN_TILED_VARIANT selects a kernel variant by name substring. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEADS = 56, HEAD_DIM = 128, MAX_REPS = 64 };

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_attn_tiled_bench: %s%s%s\n", message,
            gpu ? ": " : "", gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Uniform with unit variance; xorshift64* keeps generation of ~2G values fast. */
static void fill_random(uint16_t *values, size_t count, uint64_t seed,
                        float scale) {
    uint64_t state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1;
    for (size_t index = 0; index < count; index++) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint64_t bits = state * UINT64_C(0x2545f4914f6cdd1d);
        float unit = (float)(bits >> 40) * (1.0f / 16777216.0f);
        values[index] = to_bf16((unit * 2.0f - 1.0f) * 1.7320508f * scale);
    }
}

typedef struct {
    double max_abs, mean_abs, rel_l2, max_ref;
    size_t nonfinite;
} error_stats;

static error_stats compare(const uint16_t *test, const uint16_t *reference,
                           size_t count) {
    error_stats stats = {0};
    double diff_sq = 0.0, ref_sq = 0.0, abs_sum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double a = from_bf16(test[index]);
        double b = from_bf16(reference[index]);
        if (!isfinite(a)) { stats.nonfinite++; continue; }
        double diff = fabs(a - b);
        if (diff > stats.max_abs) stats.max_abs = diff;
        if (fabs(b) > stats.max_ref) stats.max_ref = fabs(b);
        abs_sum += diff;
        diff_sq += diff * diff;
        ref_sq += b * b;
    }
    stats.mean_abs = abs_sum / (double)count;
    stats.rel_l2 = ref_sq > 0.0 ? sqrt(diff_sq / ref_sq) : 0.0;
    return stats;
}

static int cmp_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), cmp_double);
    return count % 2 ? values[count / 2] :
        0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double run(h3_gpu *gpu, int tiled, int head_major_output,
                  h3_gpu_tensor *out, h3_gpu_tensor *q, h3_gpu_tensor *k,
                  h3_gpu_tensor *v, uint32_t seq) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    double start = seconds();
    if (!h3_gpu_begin(gpu)) die("begin", gpu);
    int ok = tiled ?
        h3_gpu_attention_tiled_bf16(gpu, out, q, k, v, seq, HEADS, HEAD_DIM,
                                    scale, head_major_output) :
        head_major_output ?
        h3_gpu_sdpa_bf16_head_major_output(gpu, out, q, k, v, seq, HEADS,
                                           HEAD_DIM, scale) :
        h3_gpu_sdpa_bf16(gpu, out, q, k, v, seq, HEADS, HEAD_DIM, scale);
    if (!ok) die(tiled ? "tiled attention" : "MPSGraph SDPA", gpu);
    if (!h3_gpu_submit(gpu)) die("submit", gpu);
    return seconds() - start;
}

int main(int argc, char **argv) {
    int timing = 0, reps = 5;
    float qscale = 1.0f;
    uint32_t seqs[16];
    int seq_count = 0;
    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--time")) timing = 1;
        else if (!strcmp(argv[index], "--reps") && index + 1 < argc)
            reps = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--qscale") && index + 1 < argc)
            qscale = (float)atof(argv[++index]);
        else if (seq_count < 16)
            seqs[seq_count++] = (uint32_t)strtoul(argv[index], NULL, 10);
    }
    if (reps < 1 || reps > MAX_REPS) die("--reps must be 1..64", NULL);
    if (!seq_count) {
        static const uint32_t defaults[] = {15208, 29416, 42088, 90880};
        for (int index = 0; index < 4; index++) seqs[index] = defaults[index];
        seq_count = 4;
    }
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_attn_tiled_bench: %s\n", error); return 1; }
    if (!h3_gpu_has_tiled_attention(gpu)) die("TensorOps unavailable", NULL);
    const char *variant = getenv("H3_ATTN_TILED_VARIANT");
    printf("variant=%s qscale=%g\n", variant ? variant : "default", qscale);

    for (int case_index = 0; case_index < seq_count; case_index++) {
        uint32_t seq = seqs[case_index];
        size_t count = (size_t)seq * HEADS * HEAD_DIM;
        uint16_t *host = malloc(count * sizeof(uint16_t));
        uint16_t *reference = malloc(count * sizeof(uint16_t));
        if (!host || !reference) die("out of host memory", NULL);
        fill_random(host, count, 1 + seq, qscale);
        h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, host, count);
        fill_random(host, count, 2 + seq, 1.0f);
        h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, host, count);
        fill_random(host, count, 3 + seq, 1.0f);
        h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, host, count);
        h3_gpu_tensor *out_ref = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *out_tiled = h3_gpu_tensor_new_bf16(gpu, count);
        if (!q || !k || !v || !out_ref || !out_tiled)
            die("tensor allocation", gpu);

        for (int head_major = 0; head_major < 2; head_major++) {
            run(gpu, 0, head_major, out_ref, q, k, v, seq);
            run(gpu, 1, head_major, out_tiled, q, k, v, seq);
            if (!h3_gpu_tensor_read_bf16(out_ref, reference, count) ||
                !h3_gpu_tensor_read_bf16(out_tiled, host, count))
                die("readback", gpu);
            error_stats stats = compare(host, reference, count);
            printf("seq=%u %s max_abs=%.3e mean_abs=%.3e rel_l2=%.3e "
                   "max|ref|=%.3f nonfinite=%zu\n", seq,
                   head_major ? "head-major-out" : "row-major-out ",
                   stats.max_abs, stats.mean_abs, stats.rel_l2,
                   stats.max_ref, stats.nonfinite);
        }

        if (timing) {
            double tiled_times[MAX_REPS], graph_times[MAX_REPS];
            for (int warm = 0; warm < 2; warm++) {
                run(gpu, 1, 0, out_tiled, q, k, v, seq);
                run(gpu, 0, 0, out_ref, q, k, v, seq);
            }
            for (int rep = 0; rep < reps; rep++) {
                tiled_times[rep] = run(gpu, 1, 0, out_tiled, q, k, v, seq);
                graph_times[rep] = run(gpu, 0, 0, out_ref, q, k, v, seq);
            }
            double tiled = median(tiled_times, reps);
            double graph = median(graph_times, reps);
            double flops = 4.0 * seq * (double)seq * (double)HEADS *
                           (double)HEAD_DIM;
            printf("seq=%u time tiled=%.4fs (%.1f TFLOPS) mpsgraph=%.4fs "
                   "(%.1f TFLOPS) tiled/mpsgraph=%.3f reps=%d\n", seq, tiled,
                   flops / tiled * 1e-12, graph, flops / graph * 1e-12,
                   tiled / graph, reps);
        }
        fflush(stdout);
        h3_gpu_tensor_free(q);
        h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(out_ref);
        h3_gpu_tensor_free(out_tiled);
        free(host);
        free(reference);
    }
    h3_gpu_free(gpu);
    return 0;
}
