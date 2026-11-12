#include "h3_gpu.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    TOKENS = 5, HIDDEN = 16, HEADS = 2, HEAD_DIM = 8, INNER = HEADS * HEAD_DIM,
    FFN = 32, ROPE_HALF = 2
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_image_gpu.c: %s\n", message);
    exit(1);
}

static h3_gpu *gpu;

static h3_gpu_tensor *upload(const float *values, size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, count);
    if (!tensor) die("tensor allocation failed");
    return tensor;
}

static h3_gpu_tensor *scratch(size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, count);
    if (!tensor) die("tensor allocation failed");
    return tensor;
}

static void gpu_ok(int ok, const char *operation) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_image_gpu.c: %s: %s\n", operation,
                h3_gpu_error(gpu));
        exit(1);
    }
}

static void fill(h3_rng *rng, float *values, size_t count, float scale,
                 float offset) {
    for (size_t index = 0; index < count; index++)
        values[index] = h3_rng_normal(rng) * scale + offset;
}

static double relative_l2(const h3_gpu_tensor *tensor, const double *want,
                          size_t count) {
    float *got = malloc(count * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_f32(tensor, got, count))
        die("cannot read result");
    double square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - want[index];
        square_error += delta * delta;
        square_value += want[index] * want[index];
    }
    free(got);
    return sqrt(square_error / square_value);
}

static void rms_norm(double *out, const double *in, const float *weight,
                     int rows, int width) {
    for (int row = 0; row < rows; row++) {
        double sum = 0.0;
        for (int k = 0; k < width; k++)
            sum += in[row * width + k] * in[row * width + k];
        double inverse = 1.0 / sqrt(sum / width + 1e-5);
        for (int k = 0; k < width; k++)
            out[row * width + k] = in[row * width + k] * inverse *
                                   (weight ? weight[k] : 1.0);
    }
}

static void linear(double *out, const double *in, const float *weight,
                   const float *bias, int rows, int input_dim,
                   int output_dim) {
    for (int row = 0; row < rows; row++)
        for (int o = 0; o < output_dim; o++) {
            double sum = bias[o];
            for (int k = 0; k < input_dim; k++)
                sum += (double)weight[o * input_dim + k] *
                       in[row * input_dim + k];
            out[row * output_dim + o] = sum;
        }
}

static double silu(double x) { return x / (1.0 + exp(-x)); }

static void test_block(void) {
    h3_rng rng;
    h3_rng_seed(&rng, 20260612);
    float x[TOKENS * HIDDEN], norm1[HIDDEN], norm2[HIDDEN];
    float qkv_w[INNER * 3 * HIDDEN], qkv_b[INNER * 3];
    float out_w[HIDDEN * INNER], out_b[HIDDEN];
    float ff1_w[FFN * 2 * HIDDEN], ff1_b[FFN * 2];
    float ff2_w[HIDDEN * FFN], ff2_b[HIDDEN];
    float scale1[HIDDEN], scale2[HIDDEN], cosine[TOKENS * ROPE_HALF],
          sine[TOKENS * ROPE_HALF];
    fill(&rng, x, TOKENS * HIDDEN, 1.0f, 0.0f);
    fill(&rng, norm1, HIDDEN, 0.2f, 1.0f);
    fill(&rng, norm2, HIDDEN, 0.2f, 1.0f);
    fill(&rng, qkv_w, INNER * 3 * HIDDEN, 0.4f, 0.0f);
    fill(&rng, qkv_b, INNER * 3, 0.1f, 0.0f);
    fill(&rng, out_w, HIDDEN * INNER, 0.3f, 0.0f);
    fill(&rng, out_b, HIDDEN, 0.1f, 0.0f);
    fill(&rng, ff1_w, FFN * 2 * HIDDEN, 0.3f, 0.0f);
    fill(&rng, ff1_b, FFN * 2, 0.1f, 0.0f);
    fill(&rng, ff2_w, HIDDEN * FFN, 0.2f, 0.0f);
    fill(&rng, ff2_b, HIDDEN, 0.1f, 0.0f);
    fill(&rng, scale1, HIDDEN, 0.3f, 0.5f);
    fill(&rng, scale2, HIDDEN, 0.3f, 0.5f);
    for (int index = 0; index < TOKENS * ROPE_HALF; index++) {
        float angle = h3_rng_normal(&rng) * 2.0f;
        cosine[index] = cosf(angle);
        sine[index] = sinf(angle);
    }

    double hidden[TOKENS * HIDDEN], normed[TOKENS * HIDDEN];
    for (int index = 0; index < TOKENS * HIDDEN; index++)
        hidden[index] = x[index];
    double qkv[TOKENS * INNER * 3], query[TOKENS * INNER], key[TOKENS * INNER],
           value[TOKENS * INNER], heads[TOKENS * INNER],
           branch[TOKENS * HIDDEN];
    rms_norm(normed, hidden, norm1, TOKENS, HIDDEN);
    linear(qkv, normed, qkv_w, qkv_b, TOKENS, HIDDEN, INNER * 3);
    for (int token = 0; token < TOKENS; token++)
        for (int head = 0; head < HEADS; head++) {
            const double *base =
                qkv + ((size_t)token * HEADS + head) * HEAD_DIM * 3;
            double q[HEAD_DIM], k[HEAD_DIM];
            rms_norm(q, base, NULL, 1, HEAD_DIM);
            rms_norm(k, base + HEAD_DIM, NULL, 1, HEAD_DIM);
            double q_out[HEAD_DIM], k_out[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                q_out[d] = q[d];
                k_out[d] = k[d];
                if (d < ROPE_HALF) {
                    double c = cosine[token * ROPE_HALF + d];
                    double s = sine[token * ROPE_HALF + d];
                    q_out[d] = q[d] * c - q[d + ROPE_HALF] * s;
                    k_out[d] = k[d] * c - k[d + ROPE_HALF] * s;
                } else if (d < 2 * ROPE_HALF) {
                    double c = cosine[token * ROPE_HALF + d - ROPE_HALF];
                    double s = sine[token * ROPE_HALF + d - ROPE_HALF];
                    q_out[d] = q[d] * c + q[d - ROPE_HALF] * s;
                    k_out[d] = k[d] * c + k[d - ROPE_HALF] * s;
                }
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                size_t at = ((size_t)token * HEADS + head) * HEAD_DIM + d;
                query[at] = q_out[d];
                key[at] = k_out[d];
                value[at] = base[2 * HEAD_DIM + d];
            }
        }
    for (int head = 0; head < HEADS; head++)
        for (int token = 0; token < TOKENS; token++) {
            double score[TOKENS], peak = -1e300, total = 0.0;
            for (int other = 0; other < TOKENS; other++) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += query[((size_t)token * HEADS + head) * HEAD_DIM + d] *
                           key[((size_t)other * HEADS + head) * HEAD_DIM + d];
                score[other] = dot / sqrt((double)HEAD_DIM);
                if (score[other] > peak) peak = score[other];
            }
            for (int other = 0; other < TOKENS; other++) {
                score[other] = exp(score[other] - peak);
                total += score[other];
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                double sum = 0.0;
                for (int other = 0; other < TOKENS; other++)
                    sum += score[other] / total *
                           value[((size_t)other * HEADS + head) * HEAD_DIM + d];
                heads[((size_t)token * HEADS + head) * HEAD_DIM + d] = sum;
            }
        }
    linear(branch, heads, out_w, out_b, TOKENS, INNER, HIDDEN);
    for (int index = 0; index < TOKENS * HIDDEN; index++)
        hidden[index] += branch[index] * scale1[index % HIDDEN];
    double ff1[TOKENS * FFN * 2], activated[TOKENS * FFN];
    rms_norm(normed, hidden, norm2, TOKENS, HIDDEN);
    linear(ff1, normed, ff1_w, ff1_b, TOKENS, HIDDEN, FFN * 2);
    for (int token = 0; token < TOKENS; token++)
        for (int column = 0; column < FFN; column++)
            activated[token * FFN + column] =
                silu(ff1[token * FFN * 2 + column]) *
                ff1[token * FFN * 2 + FFN + column];
    linear(branch, activated, ff2_w, ff2_b, TOKENS, FFN, HIDDEN);
    for (int index = 0; index < TOKENS * HIDDEN; index++)
        hidden[index] += branch[index] * scale2[index % HIDDEN];

    h3_gpu_tensor *g_hidden = upload(x, TOKENS * HIDDEN);
    h3_gpu_tensor *g_norm = scratch(TOKENS * HIDDEN);
    h3_gpu_tensor *g_qkv = scratch(TOKENS * INNER * 3);
    h3_gpu_tensor *g_query = scratch(TOKENS * INNER);
    h3_gpu_tensor *g_key = scratch(TOKENS * INNER);
    h3_gpu_tensor *g_value = scratch(TOKENS * INNER);
    h3_gpu_tensor *g_heads = scratch(TOKENS * INNER);
    h3_gpu_tensor *g_branch = scratch(TOKENS * HIDDEN);
    h3_gpu_tensor *g_ff1 = scratch(TOKENS * FFN * 2);
    h3_gpu_tensor *g_act = scratch(TOKENS * FFN);
    h3_gpu_tensor *g_norm1 = upload(norm1, HIDDEN);
    h3_gpu_tensor *g_norm2 = upload(norm2, HIDDEN);
    h3_gpu_tensor *g_qkv_w = upload(qkv_w, INNER * 3 * HIDDEN);
    h3_gpu_tensor *g_qkv_b = upload(qkv_b, INNER * 3);
    h3_gpu_tensor *g_out_w = upload(out_w, HIDDEN * INNER);
    h3_gpu_tensor *g_out_b = upload(out_b, HIDDEN);
    h3_gpu_tensor *g_ff1_w = upload(ff1_w, FFN * 2 * HIDDEN);
    h3_gpu_tensor *g_ff1_b = upload(ff1_b, FFN * 2);
    h3_gpu_tensor *g_ff2_w = upload(ff2_w, HIDDEN * FFN);
    h3_gpu_tensor *g_ff2_b = upload(ff2_b, HIDDEN);
    h3_gpu_tensor *g_scale1 = upload(scale1, HIDDEN);
    h3_gpu_tensor *g_scale2 = upload(scale2, HIDDEN);
    h3_gpu_tensor *g_cos = upload(cosine, TOKENS * ROPE_HALF);
    h3_gpu_tensor *g_sin = upload(sine, TOKENS * ROPE_HALF);

    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_rms_norm_f32(gpu, g_norm, g_hidden, g_norm1, TOKENS, HIDDEN,
                               1e-5f), "attention norm");
    gpu_ok(h3_gpu_linear_f32(gpu, g_qkv, g_norm, g_qkv_w, g_qkv_b, TOKENS,
                             HIDDEN, INNER * 3), "QKV");
    gpu_ok(h3_gpu_video_qkv_rope_f32(gpu, g_query, g_key, g_value, g_qkv,
                                     g_cos, g_sin, TOKENS, HEADS, HEAD_DIM,
                                     ROPE_HALF, 1e-5f), "QK norm and RoPE");
    gpu_ok(h3_gpu_sdpa_f32(gpu, g_heads, g_query, g_key, g_value, TOKENS,
                           HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "attention");
    gpu_ok(h3_gpu_linear_f32(gpu, g_branch, g_heads, g_out_w, g_out_b, TOKENS,
                             INNER, HIDDEN), "attention output");
    gpu_ok(h3_gpu_scale_add_f32(gpu, g_hidden, g_hidden, g_branch, g_scale1,
                                TOKENS, HIDDEN), "attention residual");
    gpu_ok(h3_gpu_rms_norm_f32(gpu, g_norm, g_hidden, g_norm2, TOKENS, HIDDEN,
                               1e-5f), "MLP norm");
    gpu_ok(h3_gpu_linear_f32(gpu, g_ff1, g_norm, g_ff1_w, g_ff1_b, TOKENS,
                             HIDDEN, FFN * 2), "MLP input");
    gpu_ok(h3_gpu_swiglu_f32(gpu, g_act, g_ff1, TOKENS, FFN), "SwiGLU");
    gpu_ok(h3_gpu_linear_f32(gpu, g_branch, g_act, g_ff2_w, g_ff2_b, TOKENS,
                             FFN, HIDDEN), "MLP output");
    gpu_ok(h3_gpu_scale_add_f32(gpu, g_hidden, g_hidden, g_branch, g_scale2,
                                TOKENS, HIDDEN), "MLP residual");
    gpu_ok(h3_gpu_submit(gpu), "submit");

    double l2 = relative_l2(g_hidden, hidden, TOKENS * HIDDEN);
    printf("2D ViT block rel-L2 %.3g\n", l2);
    if (l2 >= 1e-5) die("2D ViT decoder block differs from the reference");
    h3_gpu_tensor *all[] = {
        g_hidden, g_norm, g_qkv, g_query, g_key, g_value, g_heads, g_branch,
        g_ff1, g_act, g_norm1, g_norm2, g_qkv_w, g_qkv_b, g_out_w, g_out_b,
        g_ff1_w, g_ff1_b, g_ff2_w, g_ff2_b, g_scale1, g_scale2, g_cos, g_sin
    };
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        h3_gpu_tensor_free(all[index]);
}

static void test_swiglu_half_order(void) {
    enum { ROWS = 3, WIDTH = 4 };
    const float fused[ROWS * WIDTH * 2] = {
        1.0f, -2.0f, 0.5f, 3.0f, 0.25f, 4.0f, -1.5f, 2.0f,
        -0.75f, 1.5f, 2.5f, -3.0f, 5.0f, -1.0f, 0.5f, 1.25f,
        0.1f, 0.2f, -0.3f, 0.4f, -2.0f, 3.0f, 1.0f, -0.5f
    };
    double first_gated[ROWS * WIDTH], second_gated[ROWS * WIDTH];
    for (int row = 0; row < ROWS; row++)
        for (int column = 0; column < WIDTH; column++) {
            double first = fused[row * WIDTH * 2 + column];
            double second = fused[row * WIDTH * 2 + WIDTH + column];
            first_gated[row * WIDTH + column] = silu(first) * second;
            second_gated[row * WIDTH + column] = first * silu(second);
        }
    h3_gpu_tensor *input = upload(fused, ROWS * WIDTH * 2);
    h3_gpu_tensor *output = scratch(ROWS * WIDTH);
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_swiglu_f32(gpu, output, input, ROWS, WIDTH), "SwiGLU");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    double matching = relative_l2(output, first_gated, ROWS * WIDTH);
    double swapped = relative_l2(output, second_gated, ROWS * WIDTH);
    printf("SwiGLU silu(first)*second rel-L2 %.3g, swapped order %.3g\n",
           matching, swapped);
    if (matching >= 1e-6) die("SwiGLU is not silu(first half) * second half");
    if (swapped < 0.1)
        die("SwiGLU half-order test cannot tell the orders apart");
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
}

int main(void) {
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    test_block();
    test_swiglu_half_order();
    h3_gpu_free(gpu);
    puts("ok: image decoder Metal ops match double-precision references");
    return 0;
}
