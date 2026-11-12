#include "h3_dit.h"
#include "h3_gpu.h"
#include "h3_host.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
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

static void test_int8_mlp_row_guard(void) {
    h3_gpu_tensor *tensor = scratch(128);
    int ok = h3_gpu_mlp_int8_bf16(gpu, tensor, tensor, tensor, tensor, tensor,
                                  tensor, tensor, tensor, tensor, NULL, NULL,
                                  4, 128, 128, 128, 0, 0, 0, 0);
    if (ok) die("int8 MLP accepted 4 rows");
    if (!strstr(h3_gpu_error(gpu), "128 rows"))
        die("int8 MLP row rejection has no explanatory error");
    h3_gpu_tensor_free(tensor);
}

enum { PATCH_IN = 96, PATCH_OUT = 5376, SENTINEL = 0xABCD };
#define PROJECTION_BOUND 5e-3

static double bf16_double(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return (double)result;
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

/* The default fused_patch_pack path of an image with references projects all
 * R + P video rows through h3_gpu_patch_linear_bf16_map and the map of
 * h3_dit_video_source_rows. Plain linear, the per-segment offset kernel and a
 * double-precision reference must agree with it, text rows must stay
 * untouched, and the generated rows written at R * 96 after the condition
 * rows (as h3_dit_forward_image does) must land at rows text_len + R and on. */
static void test_mapped_projection(const char *name, int text_len,
                                   int latent_h, int latent_w,
                                   const h3_layout_ref *references,
                                   size_t reference_count) {
    char error[256];
    h3_layout layout;
    h3_layout_summary summary;
    require(h3_layout_build_image(text_len, latent_h, latent_w, references,
                                  reference_count, &layout, error,
                                  sizeof(error)), error);
    require(h3_layout_validate(&layout, (size_t)text_len, &summary, error,
                               sizeof(error)), error);
    size_t condition = summary.video_condition;
    size_t total = condition + summary.video_target;
    size_t sequence = layout.seq_len;
    size_t generated_start = (size_t)text_len + condition;
    require(sequence == (size_t)text_len + total &&
            summary.video_target_start == generated_start &&
            summary.video_target ==
                (size_t)(latent_h / 2) * (size_t)(latent_w / 2),
            "layout counts are not [text | references | generated]");

    uint32_t *source = malloc(total * sizeof(*source));
    require(source != NULL, "oom");
    require(h3_dit_video_source_rows(&layout, source, total) == total,
            "video source row count differs from R + P");
    for (size_t index = 0; index < total; index++)
        require(source[index] == (uint32_t)(text_len + (int)index),
                "video source rows are not the rows after the text");

    h3_rng rng;
    h3_rng_seed(&rng, 20261112u + (uint64_t)condition);
    size_t generated = total - condition;
    float *weight = malloc((size_t)PATCH_OUT * PATCH_IN * sizeof(*weight));
    float *bias = malloc(PATCH_OUT * sizeof(*bias));
    float *condition_rows = malloc(condition * PATCH_IN * sizeof(float));
    float *generated_rows = malloc(generated * PATCH_IN * sizeof(float));
    float *input = calloc(total * PATCH_IN, sizeof(float));
    require(weight && bias && condition_rows && generated_rows && input,
            "oom");
    fill(&rng, weight, (size_t)PATCH_OUT * PATCH_IN, 0.2f, 0.0f);
    fill(&rng, bias, PATCH_OUT, 0.1f, 0.0f);
    fill(&rng, condition_rows, condition * PATCH_IN, 1.0f, 0.0f);
    fill(&rng, generated_rows, generated * PATCH_IN, 1.0f, 0.5f);
    memcpy(input, condition_rows, condition * PATCH_IN * sizeof(float));
    memcpy(input + condition * PATCH_IN, generated_rows,
           generated * PATCH_IN * sizeof(float));

    float *zeros = calloc(total * PATCH_IN, sizeof(*zeros));
    require(zeros != NULL, "oom");
    h3_gpu_tensor *video_input = upload(zeros, total * PATCH_IN);
    free(zeros);
    h3_gpu_tensor *gpu_weight = upload(weight, (size_t)PATCH_OUT * PATCH_IN);
    h3_gpu_tensor *gpu_bias = upload(bias, PATCH_OUT);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, source, total);
    require(map != NULL, "map allocation failed");
    uint16_t *sentinel = malloc(sequence * PATCH_OUT * sizeof(*sentinel));
    require(sentinel != NULL, "oom");
    for (size_t index = 0; index < sequence * PATCH_OUT; index++)
        sentinel[index] = SENTINEL;
    h3_gpu_tensor *mapped = h3_gpu_tensor_from_bf16(gpu, sentinel,
                                                    sequence * PATCH_OUT);
    h3_gpu_tensor *segmented = h3_gpu_tensor_from_bf16(gpu, sentinel,
                                                       sequence * PATCH_OUT);
    h3_gpu_tensor *plain = h3_gpu_tensor_new_bf16(gpu, total * PATCH_OUT);
    require(mapped && segmented && plain, "tensor allocation failed");

    uint16_t *before = malloc(sequence * PATCH_OUT * sizeof(*before));
    uint16_t *after = malloc(sequence * PATCH_OUT * sizeof(*after));
    uint16_t *by_segments = malloc(sequence * PATCH_OUT * sizeof(*by_segments));
    uint16_t *plain_rows = malloc(total * PATCH_OUT * sizeof(*plain_rows));
    require(before && after && by_segments && plain_rows, "oom");

    /* Load time: the condition rows only; the generated rows are still zero. */
    gpu_ok(h3_gpu_tensor_write_f32_range(video_input, 0, condition_rows,
                                         condition * PATCH_IN),
           "write condition rows");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_patch_linear_bf16_map(
        gpu, mapped, video_input, gpu_weight, gpu_bias, map,
        (uint32_t)sequence, (uint32_t)total, PATCH_IN, PATCH_OUT),
           "mapped projection of the condition rows");
    gpu_ok(h3_gpu_submit(gpu), "submit");
    gpu_ok(h3_gpu_tensor_read_bf16(mapped, before, sequence * PATCH_OUT),
           "read before");

    /* A forward: the generated rows are written after the condition rows. */
    gpu_ok(h3_gpu_tensor_write_f32_range(video_input, condition * PATCH_IN,
                                         generated_rows,
                                         generated * PATCH_IN),
           "write generated rows");
    gpu_ok(h3_gpu_begin(gpu), "begin");
    gpu_ok(h3_gpu_patch_linear_bf16_map(
        gpu, mapped, video_input, gpu_weight, gpu_bias, map,
        (uint32_t)sequence, (uint32_t)total, PATCH_IN, PATCH_OUT),
           "mapped projection");
    gpu_ok(h3_gpu_patch_linear_bf16(
        gpu, plain, video_input, gpu_weight, gpu_bias, (uint32_t)total,
        PATCH_IN, PATCH_OUT), "plain projection");
    size_t video_offset = 0;
    for (size_t index = 0; index < layout.segment_count; index++) {
        const h3_segment *segment = &layout.segments[index];
        if (segment->kind == H3_SEG_TEXT) continue;
        size_t rows = segment->stop - segment->start;
        gpu_ok(h3_gpu_patch_linear_bf16_offset(
            gpu, segmented, segment->start * PATCH_OUT, video_input,
            video_offset * PATCH_IN, gpu_weight, gpu_bias, (uint32_t)rows,
            PATCH_IN, PATCH_OUT), "per-segment projection");
        video_offset += rows;
    }
    gpu_ok(h3_gpu_submit(gpu), "submit");
    require(video_offset == total, "segments do not cover the video rows");
    gpu_ok(h3_gpu_tensor_read_bf16(mapped, after, sequence * PATCH_OUT),
           "read mapped");
    gpu_ok(h3_gpu_tensor_read_bf16(segmented, by_segments,
                                   sequence * PATCH_OUT), "read segmented");
    gpu_ok(h3_gpu_tensor_read_bf16(plain, plain_rows, total * PATCH_OUT),
           "read plain");

    for (size_t index = 0; index < (size_t)text_len * PATCH_OUT; index++)
        require(after[index] == SENTINEL && before[index] == SENTINEL,
                "the mapped projection wrote a text row");
    require(memcmp(before + (size_t)text_len * PATCH_OUT,
                   after + (size_t)text_len * PATCH_OUT,
                   condition * PATCH_OUT * sizeof(*after)) == 0,
            "writing the generated rows changed a condition row");
    int moved = 0;
    for (size_t row = generated_start; row < sequence && !moved; row++)
        moved = memcmp(before + row * PATCH_OUT, after + row * PATCH_OUT,
                       PATCH_OUT * sizeof(*after)) != 0;
    require(moved, "the generated rows did not reach the generated positions");
    require(memcmp(after, by_segments,
                   sequence * PATCH_OUT * sizeof(*after)) == 0,
            "mapped projection differs from the per-segment projection");
    for (size_t index = 0; index < total; index++)
        require(memcmp(plain_rows + index * PATCH_OUT,
                       after + (size_t)source[index] * PATCH_OUT,
                       PATCH_OUT * sizeof(*after)) == 0,
                "mapped row differs from the plain projection of its source");

    double worst_row = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < total; index++) {
        double row_error = 0.0, row_value = 0.0;
        for (size_t column = 0; column < PATCH_OUT; column++) {
            double sum = bias[column];
            for (size_t k = 0; k < PATCH_IN; k++)
                sum += (double)weight[column * PATCH_IN + k] *
                       (double)input[index * PATCH_IN + k];
            double delta = bf16_double(
                after[(size_t)source[index] * PATCH_OUT + column]) - sum;
            row_error += delta * delta;
            row_value += sum * sum;
        }
        square_error += row_error;
        square_value += row_value;
        double row_relative = sqrt(row_error / row_value);
        if (row_relative > worst_row) worst_row = row_relative;
    }
    double relative = sqrt(square_error / square_value);
    printf("%s: R=%zu P=%zu rows, mapped vs double rel-L2 %.3g, worst row "
           "%.3g, bit-identical to the plain and per-segment projections\n",
           name, condition, generated, relative, worst_row);
    require(relative < PROJECTION_BOUND && worst_row < 2 * PROJECTION_BOUND,
            "mapped projection differs from the double-precision reference");

    free(source); free(weight); free(bias); free(condition_rows);
    free(generated_rows); free(input); free(sentinel); free(before);
    free(after); free(by_segments); free(plain_rows);
    h3_gpu_tensor_free(video_input); h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias); h3_gpu_tensor_free(map);
    h3_gpu_tensor_free(mapped); h3_gpu_tensor_free(segmented);
    h3_gpu_tensor_free(plain);
    h3_layout_free(&layout);
}

static void test_image_edit_projection(void) {
    const h3_layout_ref one[] = {{H3_LAYOUT_REF_IMAGE, 1, 6, 4, 0}};
    test_mapped_projection("one reference", 3, 8, 8, one, 1);
    const h3_layout_ref two[] = {{H3_LAYOUT_REF_IMAGE, 1, 6, 4, 0},
                                 {H3_LAYOUT_REF_IMAGE, 1, 4, 8, 0}};
    test_mapped_projection("two references", 5, 8, 6, two, 2);
}

int main(void) {
    char error[512];
    gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    test_block();
    test_swiglu_half_order();
    test_int8_mlp_row_guard();
    test_image_edit_projection();
    h3_gpu_free(gpu);
    puts("ok: image Metal ops match double-precision references");
    return 0;
}
