/* Sparse-vs-dense attention error on Q/K/V captured from a real generation
 * (H3_ATTN_CAPTURE_DIR). For each capture prefix (<dir>/bBB_sSS) it runs
 * MPSGraph SDPA (the production dense path), the dense tiled kernel (bf16
 * noise floor) and the sparse kernel, and prints one row:
 *
 *   block step sigma rows density rel_l2 max_abs cosine floor_rel_l2
 *   video_rel_l2 | cpu subset: rel_l2 sparse-fp32-sums, sparse-bf16-sums
 *
 * The CPU subset recomputes dense and the sparse approximation in double for
 * a few heads and sampled video query rows, using the GPU's routes, once with
 * the fp32 summaries and once with the key means/value sums rounded to bf16
 * as the kernel stages them.
 *
 *   ./h3_sparse_attn_check [--shaders P] [--tau T] [--cpu-rows N] PREFIX...
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 128, BLOCK = 64 };

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_sparse_attn_check: %s%s%s\n", message,
            gpu ? ": " : "", gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double json_number(const char *text, const char *key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *at = strstr(text, pattern);
    if (!at) die("capture header lacks a field", NULL);
    return strtod(at + strlen(pattern), NULL);
}

static uint16_t *read_file(const char *path, size_t count) {
    FILE *file = fopen(path, "rb");
    if (!file) die("cannot open capture file", NULL);
    uint16_t *values = malloc(count * sizeof(uint16_t));
    if (!values || fread(values, sizeof(uint16_t), count, file) != count)
        die("short capture file", NULL);
    fclose(file);
    return values;
}

typedef struct {
    double diff_sq, ref_sq, dot, test_sq, max_abs;
} accum;

static void add(accum *a, double test, double reference) {
    double diff = test - reference;
    a->diff_sq += diff * diff;
    a->ref_sq += reference * reference;
    a->dot += test * reference;
    a->test_sq += test * test;
    if (fabs(diff) > a->max_abs) a->max_abs = fabs(diff);
}

static double rel(const accum *a) {
    return a->ref_sq > 0 ? sqrt(a->diff_sq / a->ref_sq) : 0.0;
}

static double cosine(const accum *a) {
    return a->dot / sqrt(a->ref_sq * a->test_sq + 1e-300);
}

static void attention(h3_gpu *gpu, int kind, h3_gpu_tensor *out,
                      h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                      uint32_t rows, uint32_t heads, int head_major,
                      uint32_t sink_rows, float tau) {
    float scale = 1.0f / sqrtf((float)D);
    if (!h3_gpu_begin(gpu)) die("begin", gpu);
    h3_gpu_set_sdpa_inputs_head_major(gpu, head_major);
    int ok = kind == 0 ?
        h3_gpu_sdpa_bf16(gpu, out, q, k, v, rows, heads, D, scale) :
        kind == 1 ?
        h3_gpu_attention_tiled_bf16(gpu, out, q, k, v, rows, heads, D, scale, 0) :
        h3_gpu_attention_sparse_bf16(gpu, out, q, k, v, rows, heads, D, scale,
                                     0, sink_rows, tau);
    if (!ok) die("attention", gpu);
    if (!h3_gpu_submit(gpu)) die("submit", gpu);
}

int main(int argc, char **argv) {
    const char *shaders = "h3_shaders.metal";
    float tau = 1.3f;
    int cpu_rows = 128;
    int first = 1;
    while (first < argc && argv[first][0] == '-') {
        if (!strcmp(argv[first], "--shaders") && first + 1 < argc)
            shaders = argv[++first];
        else if (!strcmp(argv[first], "--tau") && first + 1 < argc)
            tau = strtof(argv[++first], NULL);
        else if (!strcmp(argv[first], "--cpu-rows") && first + 1 < argc)
            cpu_rows = atoi(argv[++first]);
        else die("usage: [--shaders P] [--tau T] [--cpu-rows N] PREFIX...", NULL);
        first++;
    }
    char error[1024];
    h3_gpu *gpu = h3_gpu_create(shaders, error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_sparse_attn_check: %s\n", error); return 1; }
    printf("block\tstep\tsigma\trows\tdensity\trel_l2\tmax_abs\tcosine\t"
           "floor_rel_l2\tvideo_rel_l2\tcpu_dense_vs_gpu_sparse\t"
           "cpu_sparse_fp32_sums\tcpu_sparse_bf16_sums\tfp32_vs_bf16_sums\n");
    for (int arg = first; arg < argc; arg++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s.json", argv[arg]);
        FILE *file = fopen(path, "r");
        if (!file) die("cannot open capture header", NULL);
        char header[2048] = {0};
        if (!fread(header, 1, sizeof(header) - 1, file)) die("empty header", NULL);
        fclose(file);
        uint32_t rows = (uint32_t)json_number(header, "rows");
        uint32_t heads = (uint32_t)json_number(header, "heads");
        int head_major = (int)json_number(header, "head_major");
        uint32_t sink_rows = (uint32_t)json_number(header, "sink_rows");
        int block = (int)json_number(header, "block");
        int step = (int)json_number(header, "step");
        double sigma = json_number(header, "sigma");
        size_t count = (size_t)rows * heads * D;
        uint16_t *host[3];
        const char *suffix[3] = {"q", "k", "v"};
        h3_gpu_tensor *tensors[3];
        for (int t = 0; t < 3; t++) {
            snprintf(path, sizeof(path), "%s.%s", argv[arg], suffix[t]);
            host[t] = read_file(path, count);
            tensors[t] = h3_gpu_tensor_from_bf16(gpu, host[t], count);
            if (!tensors[t]) die("upload", gpu);
        }
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        uint16_t *dense = malloc(count * sizeof(uint16_t));
        uint16_t *tiled = malloc(count * sizeof(uint16_t));
        uint16_t *sparse = malloc(count * sizeof(uint16_t));
        if (!out || !dense || !tiled || !sparse) die("allocation", gpu);
        attention(gpu, 0, out, tensors[0], tensors[1], tensors[2], rows, heads,
                  head_major, sink_rows, tau);
        if (!h3_gpu_tensor_read_bf16(out, dense, count)) die("read", gpu);
        attention(gpu, 1, out, tensors[0], tensors[1], tensors[2], rows, heads,
                  head_major, sink_rows, tau);
        if (!h3_gpu_tensor_read_bf16(out, tiled, count)) die("read", gpu);
        attention(gpu, 2, out, tensors[0], tensors[1], tensors[2], rows, heads,
                  head_major, sink_rows, tau);
        if (!h3_gpu_tensor_read_bf16(out, sparse, count)) die("read", gpu);

        uint32_t blocks = (rows + BLOCK - 1) / BLOCK;
        uint32_t tile_rows = h3_gpu_sparse_attention_tile_rows();
        uint32_t tiles = (rows + tile_rows - 1) / tile_rows;
        size_t plane = (size_t)heads * blocks * D;
        float *sums = malloc((3 * plane + (size_t)heads * 256) * sizeof(float));
        uint16_t *routes = malloc((size_t)heads * tiles * blocks * sizeof(uint16_t));
        uint32_t *counts = malloc((size_t)heads * tiles * sizeof(uint32_t));
        if (!h3_gpu_sparse_attention_read(gpu, rows, heads, sums, routes, counts))
            die("route readback", gpu);
        double routed = 0;
        for (size_t i = 0; i < (size_t)heads * tiles; i++) routed += counts[i];
        double density = routed / ((double)heads * tiles * blocks);

        /* Outputs are row-major [rows][heads][D]. */
        accum all = {0}, video = {0}, floor_stats = {0};
        for (size_t i = 0; i < count; i++) {
            double reference = from_bf16(dense[i]);
            add(&all, from_bf16(sparse[i]), reference);
            add(&floor_stats, from_bf16(tiled[i]), reference);
            if (i / ((size_t)heads * D) >= sink_rows)
                add(&video, from_bf16(sparse[i]), reference);
        }

        /* CPU subset in double. */
        accum gpu_sub = {0}, fp32_sub = {0}, bf16_sub = {0}, sums_sub = {0};
        const uint32_t sample_heads[3] = {0, heads / 2, heads - 1};
        double *scores = malloc(((size_t)rows + blocks) * sizeof(double));
        double *scores_bf16 = malloc((size_t)blocks * sizeof(double));
        for (int hi = 0; hi < 3; hi++) {
            uint32_t h = sample_heads[hi];
            for (int r = 0; r < cpu_rows && rows > sink_rows; r++) {
                uint32_t i = sink_rows + (uint32_t)(((uint64_t)r * 2654435761u) %
                                                    (rows - sink_rows));
#define AT(t, row, d) from_bf16(host[t][head_major ? \
    ((size_t)h * rows + (row)) * D + (d) : ((size_t)(row) * heads + h) * D + (d)])
                double qi[D];
                for (int d = 0; d < D; d++) qi[d] = AT(0, i, d);
                double scale = 1.0 / sqrt((double)D);
                double best = -INFINITY;
                for (uint32_t j = 0; j < rows; j++) {
                    double dot = 0;
                    for (int d = 0; d < D; d++) dot += qi[d] * AT(1, j, d);
                    scores[j] = dot * scale;
                    if (scores[j] > best) best = scores[j];
                }
                const uint16_t *list = routes +
                    ((size_t)h * tiles + i / tile_rows) * blocks;
                uint32_t routed_count = counts[(size_t)h * tiles + i / tile_rows];
                uint8_t *is_routed = calloc(blocks, 1);
                for (uint32_t j = 0; j < routed_count; j++) is_routed[list[j]] = 1;
                const float *kc = sums + plane + (size_t)h * blocks * D;
                const float *vs = sums + 2 * plane + (size_t)h * blocks * D;
                for (uint32_t b = 0; b < blocks; b++) {
                    if (is_routed[b]) continue;
                    double dot = 0, dot_bf16 = 0;
                    for (int d = 0; d < D; d++) {
                        dot += qi[d] * kc[(size_t)b * D + d];
                        dot_bf16 += qi[d] * round_bf16(kc[(size_t)b * D + d]);
                    }
                    scores[rows + b] = dot * scale;
                    scores_bf16[b] = dot_bf16 * scale;
                    if (scores[rows + b] > best) best = scores[rows + b];
                    if (scores_bf16[b] > best) best = scores_bf16[b];
                }
                double dense_num[D] = {0}, sparse32[D] = {0}, sparse16[D] = {0};
                double dense_den = 0, den32 = 0, den16 = 0;
                for (uint32_t j = 0; j < rows; j++) {
                    double p = exp(scores[j] - best);
                    dense_den += p;
                    int exact = is_routed[j / BLOCK];
                    if (exact) { den32 += p; den16 += p; }
                    for (int d = 0; d < D; d++) {
                        double value = AT(2, j, d);
                        dense_num[d] += p * value;
                        if (exact) { sparse32[d] += p * value; sparse16[d] += p * value; }
                    }
                }
                for (uint32_t b = 0; b < blocks; b++) {
                    if (is_routed[b]) continue;
                    double p32 = exp(scores[rows + b] - best);
                    double p16 = exp(scores_bf16[b] - best);
                    den32 += 64.0 * p32;
                    den16 += 64.0 * p16;
                    for (int d = 0; d < D; d++) {
                        sparse32[d] += p32 * vs[(size_t)b * D + d];
                        sparse16[d] += p16 * round_bf16(vs[(size_t)b * D + d]);
                    }
                }
                for (int d = 0; d < D; d++) {
                    double reference = dense_num[d] / dense_den;
                    double s32 = sparse32[d] / den32, s16 = sparse16[d] / den16;
                    add(&gpu_sub, from_bf16(sparse[((size_t)i * heads + h) * D + d]),
                        reference);
                    add(&fp32_sub, s32, reference);
                    add(&bf16_sub, s16, reference);
                    add(&sums_sub, s16, s32);
                }
                free(is_routed);
#undef AT
            }
        }
        printf("%d\t%d\t%.4f\t%u\t%.4f\t%.3e\t%.3e\t%.6f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",
               block, step, sigma, rows, density, rel(&all), all.max_abs,
               cosine(&all), rel(&floor_stats), rel(&video), rel(&gpu_sub),
               rel(&fp32_sub), rel(&bf16_sub), rel(&sums_sub));
        fflush(stdout);
        free(scores); free(scores_bf16); free(sums); free(routes); free(counts);
        free(dense); free(tiled); free(sparse);
        h3_gpu_tensor_free(out);
        for (int t = 0; t < 3; t++) { h3_gpu_tensor_free(tensors[t]); free(host[t]); }
    }
    h3_gpu_free(gpu);
    return 0;
}
