/* H3_SOL_QK_INT8 accuracy on Q/K/V captured from a real generation: the int8
 * QK^T sparse kernel against the bf16 sparse kernel, per head. For scale it
 * also prints the dense tiled kernel against MPSGraph SDPA (the accepted
 * bf16 noise) and both sparse kernels against MPSGraph.
 *
 *   ./h3_sol_qk8_check [--tau T] PREFIX...
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 128 };

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_sol_qk8_check: %s%s%s\n", message, gpu ? ": " : "",
            gpu ? h3_gpu_error(gpu) : "");
    exit(1);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
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

typedef struct { double diff_sq, ref_sq, dot, test_sq, max_abs; } accum;

static accum compare(const uint16_t *test, const uint16_t *ref, uint32_t rows,
                     uint32_t heads, uint32_t h) {
    accum a = {0};
    for (uint32_t r = 0; r < rows; r++)
        for (int d = 0; d < D; d++) {
            size_t i = ((size_t)r * heads + h) * D + d;
            double t = from_bf16(test[i]), f = from_bf16(ref[i]), e = t - f;
            a.diff_sq += e * e;
            a.ref_sq += f * f;
            a.dot += t * f;
            a.test_sq += t * t;
            if (fabs(e) > a.max_abs) a.max_abs = fabs(e);
        }
    return a;
}

static void print(const char *label, accum a) {
    printf(" %s cos=%.6f rel=%.3e max=%.3e", label,
           a.dot / sqrt(a.ref_sq * a.test_sq + 1e-300),
           a.ref_sq > 0 ? sqrt(a.diff_sq / a.ref_sq) : 0.0, a.max_abs);
}

static void attention(h3_gpu *gpu, int kind, h3_gpu_tensor *out,
                      h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                      uint32_t rows, uint32_t heads, int head_major,
                      uint32_t sink_rows, float tau, uint16_t *host) {
    float scale = 1.0f / sqrtf((float)D);
    if (kind == 3) setenv("H3_SOL_QK_INT8", "1", 1);
    else unsetenv("H3_SOL_QK_INT8");
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
    unsetenv("H3_SOL_QK_INT8");
    if (!h3_gpu_tensor_read_bf16(out, host, (size_t)rows * heads * D))
        die("read", gpu);
}

int main(int argc, char **argv) {
    float tau = 1.3f;
    int first = 1;
    if (first + 1 < argc && !strcmp(argv[first], "--tau")) {
        tau = strtof(argv[first + 1], NULL);
        first += 2;
    }
    if (first >= argc) die("usage: [--tau T] PREFIX...", NULL);
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_sol_qk8_check: %s\n", error); return 1; }
    double worst_cos = 1.0;
    for (int arg = first; arg < argc; arg++) {
        char path[4096], header[2048] = {0};
        snprintf(path, sizeof(path), "%s.json", argv[arg]);
        FILE *file = fopen(path, "r");
        if (!file || !fread(header, 1, sizeof(header) - 1, file))
            die("cannot read capture header", NULL);
        fclose(file);
        uint32_t rows = (uint32_t)json_number(header, "rows");
        uint32_t heads = (uint32_t)json_number(header, "heads");
        int head_major = (int)json_number(header, "head_major");
        uint32_t sink_rows = (uint32_t)json_number(header, "sink_rows");
        size_t count = (size_t)rows * heads * D;
        h3_gpu_tensor *tensors[3];
        const char *suffix[3] = {"q", "k", "v"};
        uint16_t *host = malloc(count * sizeof(uint16_t));
        for (int t = 0; t < 3; t++) {
            snprintf(path, sizeof(path), "%s.%s", argv[arg], suffix[t]);
            file = fopen(path, "rb");
            if (!file || fread(host, sizeof(uint16_t), count, file) != count)
                die("cannot read capture", NULL);
            fclose(file);
            tensors[t] = h3_gpu_tensor_from_bf16(gpu, host, count);
            if (!tensors[t]) die("upload", gpu);
        }
        free(host);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        uint16_t *result[4];
        for (int kind = 0; kind < 4; kind++) {
            result[kind] = malloc(count * sizeof(uint16_t));
            if (!out || !result[kind]) die("allocation", gpu);
            attention(gpu, kind, out, tensors[0], tensors[1], tensors[2], rows,
                      heads, head_major, sink_rows, tau, result[kind]);
        }
        for (uint32_t h = 0; h < heads; h++) {
            printf("%s rows=%u head=%u", strrchr(argv[arg], '/') ?
                   strrchr(argv[arg], '/') + 1 : argv[arg], rows, h);
            accum gate = compare(result[3], result[2], rows, heads, h);
            double cos = gate.dot / sqrt(gate.ref_sq * gate.test_sq + 1e-300);
            if (cos < worst_cos) worst_cos = cos;
            print("int8_vs_bf16sparse", gate);
            print("| tiled_vs_mps", compare(result[1], result[0], rows, heads, h));
            print("| bf16sparse_vs_mps", compare(result[2], result[0], rows, heads, h));
            print("| int8sparse_vs_mps", compare(result[3], result[0], rows, heads, h));
            printf("\n");
        }
        fflush(stdout);
        for (int kind = 0; kind < 4; kind++) free(result[kind]);
        h3_gpu_tensor_free(out);
        for (int t = 0; t < 3; t++) h3_gpu_tensor_free(tensors[t]);
    }
    printf("worst int8_vs_bf16sparse head cosine=%.6f gate(>=0.999)=%s\n",
           worst_cos, worst_cos >= 0.999 ? "PASS" : "FAIL");
    h3_gpu_free(gpu);
    return worst_cos >= 0.999 ? 0 : 2;
}
