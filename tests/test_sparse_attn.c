/* Unit correctness of the Sol-Attn sparse attention kernel on synthetic
 * inputs, with routing forced or checked against a CPU reference.
 *
 *   ./h3_sparse_attn_test [h3_shaders.metal]
 *
 * Cases per (seq, layout):
 *   all     tau=-1e30: every block routed; must equal the dense tiled kernel
 *           bit for bit and MPSGraph SDPA to bf16 rounding.
 *   none    tau=+1e30: only sinks, neighbours and the partial tail block are
 *           exact; routes must match the CPU routing and the output must
 *           match the CPU approximation (fp32 summaries; also reported with
 *           the summaries rounded to bf16 as the kernel stages them).
 *   mixed   tau=1.3 on block-structured inputs: CPU approximation evaluated
 *           with the routes the GPU chose. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEADS = 4, D = 128, BLOCK = 64 };

static int failures;

static void die(const char *message, const h3_gpu *gpu) {
    fprintf(stderr, "h3_sparse_attn_test: %s%s%s\n", message,
            gpu ? ": " : "", gpu ? h3_gpu_error(gpu) : "");
    exit(1);
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

static float round_bf16(float value) { return from_bf16(to_bf16(value)); }

static uint64_t rng_state;
static float uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t bits = rng_state * UINT64_C(0x2545f4914f6cdd1d);
    return (float)(bits >> 40) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

/* Canonical [head][seq][D] values, bf16-representable. structure > 0 adds a
 * per-block offset so block summaries differ and routing has something to
 * choose. */
static void fill(float *values, uint32_t seq, uint64_t seed, float scale,
                 float structure) {
    rng_state = seed * UINT64_C(0x9e3779b97f4a7c15) + 1;
    uint32_t blocks = (seq + BLOCK - 1) / BLOCK;
    float *offsets = calloc((size_t)HEADS * blocks * D, sizeof(float));
    for (size_t i = 0; i < (size_t)HEADS * blocks * D; i++)
        offsets[i] = structure * uniform() * 1.7320508f;
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t s = 0; s < seq; s++)
            for (uint32_t d = 0; d < D; d++) {
                float base = uniform() * 1.7320508f * scale +
                             offsets[((size_t)h * blocks + s / BLOCK) * D + d];
                values[((size_t)h * seq + s) * D + d] = round_bf16(base);
            }
    free(offsets);
}

static size_t index_of(int head_major, uint32_t seq, uint32_t h, uint32_t s,
                       uint32_t d) {
    return head_major ? ((size_t)h * seq + s) * D + d
                      : ((size_t)s * HEADS + h) * D + d;
}

static h3_gpu_tensor *upload(h3_gpu *gpu, const float *values, uint32_t seq,
                             int head_major) {
    size_t count = (size_t)seq * HEADS * D;
    uint16_t *packed = malloc(count * sizeof(uint16_t));
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t s = 0; s < seq; s++)
            for (uint32_t d = 0; d < D; d++)
                packed[index_of(head_major, seq, h, s, d)] =
                    to_bf16(values[((size_t)h * seq + s) * D + d]);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, packed, count);
    free(packed);
    if (!tensor) die("upload", gpu);
    return tensor;
}

/* Output as canonical [head][seq][D] floats. */
static void download(const h3_gpu_tensor *tensor, float *values, uint32_t seq,
                     int head_major) {
    size_t count = (size_t)seq * HEADS * D;
    uint16_t *packed = malloc(count * sizeof(uint16_t));
    if (!h3_gpu_tensor_read_bf16(tensor, packed, count)) die("readback", NULL);
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t s = 0; s < seq; s++)
            for (uint32_t d = 0; d < D; d++)
                values[((size_t)h * seq + s) * D + d] =
                    from_bf16(packed[index_of(head_major, seq, h, s, d)]);
    free(packed);
}

typedef enum { RUN_SPARSE, RUN_TILED, RUN_GRAPH } run_kind;

static void run(h3_gpu *gpu, run_kind kind, h3_gpu_tensor *out,
                h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
                uint32_t seq, int in_head_major, int out_head_major,
                uint32_t sink_rows, float tau) {
    float scale = 1.0f / sqrtf((float)D);
    if (!h3_gpu_begin(gpu)) die("begin", gpu);
    h3_gpu_set_sdpa_inputs_head_major(gpu, in_head_major);
    int ok;
    if (kind == RUN_SPARSE)
        ok = h3_gpu_attention_sparse_bf16(gpu, out, q, k, v, seq, HEADS, D,
                                          scale, out_head_major, sink_rows, tau);
    else if (kind == RUN_TILED)
        ok = h3_gpu_attention_tiled_bf16(gpu, out, q, k, v, seq, HEADS, D,
                                         scale, out_head_major);
    else
        ok = out_head_major ?
            h3_gpu_sdpa_bf16_head_major_output(gpu, out, q, k, v, seq, HEADS,
                                               D, scale) :
            h3_gpu_sdpa_bf16(gpu, out, q, k, v, seq, HEADS, D, scale);
    if (!ok) die("attention", gpu);
    if (!h3_gpu_submit(gpu)) die("submit", gpu);
}

typedef struct { double rel_l2, max_abs, max_ref; size_t mismatches; } stats;

static stats compare(const float *test, const float *reference, size_t count) {
    stats result = {0};
    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = test[i], b = reference[i];
        double diff = fabs(a - b);
        if (!(diff == diff)) diff = INFINITY;
        if (diff > result.max_abs) result.max_abs = diff;
        if (fabs(b) > result.max_ref) result.max_ref = fabs(b);
        if (test[i] != reference[i]) result.mismatches++;
        diff_sq += diff * diff;
        ref_sq += b * b;
    }
    result.rel_l2 = ref_sq > 0.0 ? sqrt(diff_sq / ref_sq) : 0.0;
    return result;
}

/* CPU summaries, same summation order as the kernel. */
static void summaries(const float *x, uint32_t seq, uint32_t blocks,
                      float *mean, float *sum) {
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t start = b * BLOCK;
            uint32_t length = seq - start < BLOCK ? seq - start : BLOCK;
            for (uint32_t d = 0; d < D; d++) {
                float total = 0.0f;
                for (uint32_t r = 0; r < length; r++)
                    total += x[((size_t)h * seq + start + r) * D + d];
                if (mean) mean[((size_t)h * blocks + b) * D + d] =
                    total / (float)length;
                if (sum) sum[((size_t)h * blocks + b) * D + d] = total;
            }
        }
}

/* CPU routing flags [head][tile][block] for forced taus (+-inf thresholds
 * reduce to the structural rules). */
static void cpu_route_structural(uint32_t seq, uint32_t sink_rows, int all,
                                 uint8_t *flags) {
    uint32_t blocks = (seq + BLOCK - 1) / BLOCK;
    uint32_t tile_rows = h3_gpu_sparse_attention_tile_rows();
    uint32_t tiles = (seq + tile_rows - 1) / tile_rows;
    uint32_t per_tile = tile_rows / BLOCK;
    uint32_t sink_blocks = (sink_rows + BLOCK - 1) / BLOCK;
    uint32_t forced = seq % BLOCK ? blocks - 1 : blocks;
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t t = 0; t < tiles; t++)
            for (uint32_t b = 0; b < blocks; b++) {
                int flag = all || t * per_tile < sink_blocks ||
                           b < sink_blocks || b == forced;
                for (uint32_t i = 0; i < per_tile; i++) {
                    uint32_t qb = t * per_tile + i;
                    if (qb < blocks && (qb > b ? qb - b : b - qb) <= 1)
                        flag = 1;
                }
                flags[((size_t)h * tiles + t) * blocks + b] = (uint8_t)flag;
            }
}

static void flags_from_routes(const uint16_t *routes, const uint32_t *counts,
                              uint32_t heads_tiles, uint32_t blocks,
                              uint8_t *flags, int *valid) {
    *valid = 1;
    memset(flags, 0, (size_t)heads_tiles * blocks);
    for (uint32_t i = 0; i < heads_tiles; i++) {
        const uint16_t *list = routes + (size_t)i * blocks;
        uint8_t *row = flags + (size_t)i * blocks;
        uint8_t *seen = calloc(blocks, 1);
        for (uint32_t j = 0; j < blocks; j++) {
            if (list[j] >= blocks || seen[list[j]]) { *valid = 0; break; }
            seen[list[j]] = 1;
            if (j < counts[i]) {
                row[list[j]] = 1;
                if (j && list[j] <= list[j - 1]) *valid = 0;  /* ascending */
            }
        }
        free(seen);
    }
}

/* CPU approximation: per query row, exact keys of routed blocks plus one
 * virtual key per unrouted block (key mean, value sum, weight = length). */
static void cpu_sparse(const float *q, const float *k, const float *v,
                       uint32_t seq, const uint8_t *flags, int round_summaries,
                       float *out) {
    uint32_t blocks = (seq + BLOCK - 1) / BLOCK;
    uint32_t tile_rows = h3_gpu_sparse_attention_tile_rows();
    uint32_t tiles = (seq + tile_rows - 1) / tile_rows;
    float *kc = malloc((size_t)HEADS * blocks * D * sizeof(float));
    float *vs = malloc((size_t)HEADS * blocks * D * sizeof(float));
    summaries(k, seq, blocks, kc, NULL);
    summaries(v, seq, blocks, NULL, vs);
    if (round_summaries)
        for (size_t i = 0; i < (size_t)HEADS * blocks * D; i++) {
            kc[i] = round_bf16(kc[i]);
            vs[i] = round_bf16(vs[i]);
        }
    double scale = 1.0 / sqrt((double)D);
    double *scores = malloc(((size_t)seq + blocks) * sizeof(double));
    double accum[D];
    for (uint32_t h = 0; h < HEADS; h++)
        for (uint32_t i = 0; i < seq; i++) {
            const uint8_t *row = flags + ((size_t)h * tiles + i / tile_rows) * blocks;
            const float *qi = q + ((size_t)h * seq + i) * D;
            double best = -INFINITY;
            for (uint32_t b = 0; b < blocks; b++) {
                uint32_t start = b * BLOCK;
                uint32_t length = seq - start < BLOCK ? seq - start : BLOCK;
                if (row[b]) {
                    for (uint32_t r = 0; r < length; r++) {
                        const float *kj = k + ((size_t)h * seq + start + r) * D;
                        double dot = 0.0;
                        for (uint32_t d = 0; d < D; d++) dot += (double)qi[d] * kj[d];
                        scores[start + r] = dot * scale;
                        if (scores[start + r] > best) best = scores[start + r];
                    }
                } else {
                    const float *kb = kc + ((size_t)h * blocks + b) * D;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < D; d++) dot += (double)qi[d] * kb[d];
                    scores[seq + b] = dot * scale;
                    if (scores[seq + b] > best) best = scores[seq + b];
                }
            }
            double denominator = 0.0;
            memset(accum, 0, sizeof(accum));
            for (uint32_t b = 0; b < blocks; b++) {
                uint32_t start = b * BLOCK;
                uint32_t length = seq - start < BLOCK ? seq - start : BLOCK;
                if (row[b]) {
                    for (uint32_t r = 0; r < length; r++) {
                        double p = exp(scores[start + r] - best);
                        const float *vj = v + ((size_t)h * seq + start + r) * D;
                        denominator += p;
                        for (uint32_t d = 0; d < D; d++) accum[d] += p * vj[d];
                    }
                } else {
                    double p = exp(scores[seq + b] - best);
                    const float *vb = vs + ((size_t)h * blocks + b) * D;
                    denominator += p * length;
                    for (uint32_t d = 0; d < D; d++) accum[d] += p * vb[d];
                }
            }
            for (uint32_t d = 0; d < D; d++)
                out[((size_t)h * seq + i) * D + d] = (float)(accum[d] / denominator);
        }
    free(scores);
    free(kc);
    free(vs);
}

static void check(const char *label, int ok) {
    printf("  %-58s %s\n", label, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

static void report(const char *label, stats s, double rel_limit, int exact) {
    char line[160];
    snprintf(line, sizeof(line),
             "%s rel_l2=%.2e max_abs=%.2e max|ref|=%.2f diff=%zu", label,
             s.rel_l2, s.max_abs, s.max_ref, s.mismatches);
    check(line, exact ? s.mismatches == 0 : s.rel_l2 <= rel_limit);
}

int main(int argc, char **argv) {
    char error[1024];
    h3_gpu *gpu = h3_gpu_create(argc > 1 ? argv[1] : "h3_shaders.metal",
                                error, sizeof(error));
    if (!gpu) { fprintf(stderr, "h3_sparse_attn_test: %s\n", error); return 1; }
    if (!h3_gpu_has_tiled_attention(gpu)) die("TensorOps unavailable", NULL);
    static const uint32_t seqs[] = {2000, 3000, 4096};
    const uint32_t sink_rows = 300;
    for (size_t si = 0; si < sizeof(seqs) / sizeof(seqs[0]); si++) {
        uint32_t seq = seqs[si];
        uint32_t blocks = (seq + BLOCK - 1) / BLOCK;
        uint32_t tile_rows = h3_gpu_sparse_attention_tile_rows();
        uint32_t tiles = (seq + tile_rows - 1) / tile_rows;
        size_t count = (size_t)seq * HEADS * D;
        size_t route_count = (size_t)HEADS * tiles * blocks;
        float *q = malloc(count * sizeof(float));
        float *k = malloc(count * sizeof(float));
        float *v = malloc(count * sizeof(float));
        float *got = malloc(count * sizeof(float));
        float *want = malloc(count * sizeof(float));
        uint16_t *routes = malloc(route_count * sizeof(uint16_t));
        uint32_t *counts = malloc((size_t)HEADS * tiles * sizeof(uint32_t));
        uint8_t *gpu_flags = malloc(route_count);
        uint8_t *cpu_flags = malloc(route_count);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, count);
        for (int structured = 0; structured < 2; structured++) {
            fill(q, seq, 11 + seq, 2.0f, structured ? 1.5f : 0.0f);
            fill(k, seq, 12 + seq, 1.0f, structured ? 1.5f : 0.0f);
            fill(v, seq, 13 + seq, 1.0f, structured ? 0.5f : 0.0f);
            for (int layout = 0; layout < 2; layout++) {
                int in_head_major = layout, out_head_major = layout;
                printf("seq=%u tail=%u %s inputs=%s\n", seq, seq % BLOCK,
                       structured ? "structured" : "iid",
                       in_head_major ? "head-major" : "row-major");
                h3_gpu_tensor *tq = upload(gpu, q, seq, in_head_major);
                h3_gpu_tensor *tk = upload(gpu, k, seq, in_head_major);
                h3_gpu_tensor *tv = upload(gpu, v, seq, in_head_major);
                if (!structured) {
                    /* all routed */
                    run(gpu, RUN_SPARSE, out, tq, tk, tv, seq, in_head_major,
                        out_head_major, sink_rows, -1.0e30f);
                    download(out, got, seq, out_head_major);
                    run(gpu, RUN_TILED, out, tq, tk, tv, seq, in_head_major,
                        out_head_major, sink_rows, 0.0f);
                    download(out, want, seq, out_head_major);
                    report("all routed vs dense tiled (bitwise)",
                           compare(got, want, count), 0.0, 1);
                    run(gpu, RUN_GRAPH, out, tq, tk, tv, seq, in_head_major,
                        out_head_major, sink_rows, 0.0f);
                    download(out, want, seq, out_head_major);
                    stats graph = compare(got, want, count);
                    report("all routed vs MPSGraph", graph, 4e-3, 0);
                    /* max abs within 2 bf16 ulps of the largest output */
                    check("  ... max_abs <= 2 ulp(max|ref|)",
                          graph.max_abs <= 2.0 * ldexp(1.0,
                              (int)floor(log2(graph.max_ref)) - 7));
                    /* none routed */
                    run(gpu, RUN_SPARSE, out, tq, tk, tv, seq, in_head_major,
                        out_head_major, sink_rows, 1.0e30f);
                    download(out, got, seq, out_head_major);
                    int valid;
                    if (!h3_gpu_sparse_attention_read(gpu, seq, HEADS, NULL,
                                                      routes, counts))
                        die("route readback", gpu);
                    flags_from_routes(routes, counts, HEADS * tiles, blocks,
                                      gpu_flags, &valid);
                    cpu_route_structural(seq, sink_rows, 0, cpu_flags);
                    check("none routed: GPU route list is a valid permutation",
                          valid);
                    check("none routed: GPU routes == CPU structural routes",
                          !memcmp(gpu_flags, cpu_flags, route_count));
                    double density = 0.0;
                    for (size_t i = 0; i < route_count; i++)
                        density += cpu_flags[i];
                    printf("  density %.3f\n", density / (double)route_count);
                    cpu_sparse(q, k, v, seq, cpu_flags, 0, want);
                    report("none routed vs CPU approx (fp32 summaries)",
                           compare(got, want, count), 6e-3, 0);
                    cpu_sparse(q, k, v, seq, cpu_flags, 1, want);
                    report("none routed vs CPU approx (bf16-staged summaries)",
                           compare(got, want, count), 4e-3, 0);
                } else {
                    run(gpu, RUN_SPARSE, out, tq, tk, tv, seq, in_head_major,
                        out_head_major, sink_rows, 1.3f);
                    download(out, got, seq, out_head_major);
                    int valid;
                    if (!h3_gpu_sparse_attention_read(gpu, seq, HEADS, NULL,
                                                      routes, counts))
                        die("route readback", gpu);
                    flags_from_routes(routes, counts, HEADS * tiles, blocks,
                                      gpu_flags, &valid);
                    cpu_route_structural(seq, sink_rows, 0, cpu_flags);
                    int superset = valid;
                    double density = 0.0;
                    for (size_t i = 0; i < route_count; i++) {
                        if (cpu_flags[i] && !gpu_flags[i]) superset = 0;
                        density += gpu_flags[i];
                    }
                    printf("  tau=1.3 density %.3f\n",
                           density / (double)route_count);
                    check("mixed: routes valid and include structural blocks",
                          superset);
                    /* Threshold rule from the GPU's own summaries, in double;
                     * pairs within 1e-3 of the threshold are not judged. */
                    size_t plane = (size_t)HEADS * blocks * D;
                    float *sums = malloc((3 * plane + HEADS * 256) * sizeof(float));
                    float *kmean = malloc(plane * sizeof(float));
                    if (!h3_gpu_sparse_attention_read(gpu, seq, HEADS, sums,
                                                      NULL, NULL))
                        die("summary readback", gpu);
                    summaries(k, seq, blocks, kmean, NULL);
                    int summaries_ok = !memcmp(kmean, sums + plane,
                                               plane * sizeof(float));
                    check("mixed: GPU key means == CPU key means (bitwise)",
                          summaries_ok);
                    double c = 1.4426950408889634 / sqrt((double)D);
                    size_t disagree = 0, judged = 0;
                    uint32_t per_tile = tile_rows / BLOCK;
                    for (uint32_t h = 0; h < HEADS; h++)
                        for (uint32_t t = 0; t < tiles; t++)
                            for (uint32_t b = 0; b < blocks; b++) {
                                size_t f = ((size_t)h * tiles + t) * blocks + b;
                                if (cpu_flags[f]) continue;
                                int flag = 0, close = 0;
                                for (uint32_t i = 0; i < per_tile; i++) {
                                    uint32_t qb = t * per_tile + i;
                                    if (qb >= blocks) continue;
                                    const float *qc = sums + ((size_t)h * blocks + qb) * D;
                                    const float *kc = sums + plane + ((size_t)h * blocks + b) * D;
                                    const float *st = sums + 3 * plane + (size_t)h * 256;
                                    double mean = 0, var = 0, score = 0;
                                    for (uint32_t d = 0; d < D; d++) {
                                        mean += (double)qc[d] * st[d];
                                        var += (double)qc[d] * qc[d] * st[128 + d];
                                        score += (double)qc[d] * kc[d];
                                    }
                                    double thr = mean * c + 1.3 * sqrt(var * c * c + 1e-6);
                                    if (fabs(score * c - thr) < 1e-3) close = 1;
                                    if (score * c > thr) flag = 1;
                                }
                                if (close) continue;
                                judged++;
                                if (flag != gpu_flags[f]) disagree++;
                            }
                    char line[128];
                    snprintf(line, sizeof(line),
                             "mixed: threshold routing matches CPU (%zu/%zu differ)",
                             disagree, judged);
                    check(line, disagree == 0);
                    free(sums);
                    free(kmean);
                    cpu_sparse(q, k, v, seq, gpu_flags, 1, want);
                    report("mixed vs CPU approx with GPU routes (bf16 summaries)",
                           compare(got, want, count), 4e-3, 0);
                }
                h3_gpu_tensor_free(tq);
                h3_gpu_tensor_free(tk);
                h3_gpu_tensor_free(tv);
            }
        }
        h3_gpu_tensor_free(out);
        free(q); free(k); free(v); free(got); free(want);
        free(routes); free(counts); free(gpu_flags); free(cpu_flags);
    }
    h3_gpu_free(gpu);
    printf(failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
