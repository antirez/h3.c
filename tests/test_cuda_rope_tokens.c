#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int close_values(const float *got, const float *want, size_t count,
                        float tolerance) {
    for (size_t i = 0; i < count; i++)
        if (fabsf(got[i] - want[i]) > tolerance) return 0;
    return 1;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    const float qkv_values[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    const float ones[] = {1,1,1,1};
    const float cos_values[] = {1,1};
    const float sin_values[] = {0,0};
    h3_gpu_tensor *qkv = h3_gpu_tensor_from_f32(gpu, qkv_values, 12);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, ones, 4);
    h3_gpu_tensor *cosine = h3_gpu_tensor_from_f32(gpu, cos_values, 2);
    h3_gpu_tensor *sine = h3_gpu_tensor_from_f32(gpu, sin_values, 2);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, 4);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, 4);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(qkv && weight && cosine && sine && q && k && v);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_qkv_rope_f32(gpu, q, k, v, qkv, weight, weight,
                              cosine, sine, 1, 1, 4, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float got[12];
    CHECK(h3_gpu_tensor_read_f32(q, got, 4));
    float q_inverse = 1.0f / sqrtf(7.5f + 1e-5f);
    const float q_want[] = {1*q_inverse,2*q_inverse,3*q_inverse,4*q_inverse};
    CHECK(close_values(got, q_want, 4, 2e-6f));
    CHECK(h3_gpu_tensor_read_f32(v, got, 4));
    CHECK(close_values(got, qkv_values + 8, 4, 0.0f));

    const float multi_qkv_values[] = {
        1,2,3,4, 5,6,7,8, 9,10,11,12,
        13,14,15,16, 17,18,19,20, 21,22,23,24};
    h3_gpu_tensor *multi_qkv = h3_gpu_tensor_from_f32(
        gpu, multi_qkv_values, 24);
    h3_gpu_tensor *multi_q = h3_gpu_tensor_new_f32(gpu, 8);
    h3_gpu_tensor *multi_k = h3_gpu_tensor_new_f32(gpu, 8);
    h3_gpu_tensor *multi_v = h3_gpu_tensor_new_f32(gpu, 8);
    CHECK(multi_qkv && multi_q && multi_k && multi_v);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_qkv_rope_f32(gpu, multi_q, multi_k, multi_v, multi_qkv,
        weight, weight, weight, weight, 2, 2, 2, 0, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(multi_v, got, 8));
    CHECK(close_values(got,
        (const float[]){9,10,21,22,11,12,23,24}, 8, 0.0f));

    h3_gpu_tensor *bqkv = h3_gpu_tensor_new_bf16(gpu, 12);
    h3_gpu_tensor *bweight = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bcos = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *bsin = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *bq = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bk = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bv = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(bqkv && bweight && bcos && bsin && bq && bk && bv);
    CHECK(h3_gpu_tensor_write_f32(bqkv, qkv_values, 12));
    CHECK(h3_gpu_tensor_write_f32(bweight, ones, 4));
    CHECK(h3_gpu_tensor_write_f32(bcos, cos_values, 2));
    CHECK(h3_gpu_tensor_write_f32(bsin, sin_values, 2));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_qkv_rope_bf16(gpu, bq, bk, bv, bqkv, bweight, bweight,
                               bcos, bsin, 1, 1, 4, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bq, got, 4));
    CHECK(close_values(got, q_want, 4, 0.01f));
    const float projection_input_values[] = {1,1,1,1};
    float projection_weight_values[48] = {0};
    for (size_t row = 0; row < 12; row++)
        projection_weight_values[row * 4] = qkv_values[row];
    h3_gpu_tensor *projection_input = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *projection_weight = h3_gpu_tensor_new_bf16(gpu, 48);
    CHECK(projection_input && projection_weight);
    CHECK(h3_gpu_tensor_write_f32(projection_input, projection_input_values, 4));
    CHECK(h3_gpu_tensor_write_f32(projection_weight, projection_weight_values, 48));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_grouped_qkv_linear_rope_bf16(
        gpu, bq, bk, bv, bqkv, projection_input, projection_weight, bweight,
        bweight, bcos, bsin, 1, 4, 1, 4, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bq, got, 4));
    CHECK(close_values(got, q_want, 4, 0.01f));
    h3_gpu_tensor *projection_weight_i8 = h3_gpu_tensor_new_i8(gpu, 48);
    h3_gpu_tensor *projection_weight_scales = h3_gpu_tensor_new_f32(gpu, 12);
    h3_gpu_tensor *projection_quantized = h3_gpu_tensor_new_i8(gpu, 4);
    h3_gpu_tensor *projection_scales = h3_gpu_tensor_new_f32(gpu, 1);
    CHECK(projection_weight_i8 && projection_weight_scales &&
          projection_quantized && projection_scales);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_quantize_weight_int8(gpu, projection_weight_i8,
        projection_weight_scales, projection_weight, 12, 4));
    CHECK(h3_gpu_grouped_qkv_linear_rope_int8(
        gpu, bq, bk, bv, projection_quantized, projection_scales,
        projection_input, projection_weight_i8, projection_weight_scales,
        bweight, bweight, bcos, bsin, 1, 4, 1, 4, 2, 1e-5f,
        0, 0, 0, 0));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bq, got, 4));
    CHECK(close_values(got, q_want, 4, 0.02f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_grouped_qkv_rope_bf16(gpu, bq, bk, bv, bqkv, bweight,
        bweight, bcos, bsin, 1, 1, 4, 2, 1e-5f));
    CHECK(h3_gpu_grouped_qkv_rope_bf16(gpu, bq, bk, bv, bqkv, bweight,
        bweight, bweight, bweight, 1, 1, 4, 0, 1e-5f));
    CHECK(h3_gpu_vision_qkv_rope_bf16(gpu, bq, bk, bv, bqkv, bcos, bsin,
                                      1, 1, 4, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bq, got, 4));
    CHECK(close_values(got, qkv_values, 4, 0.0f));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_video_qkv_rope_f32(gpu, q, k, v, qkv, cosine, sine,
                                    1, 1, 4, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(q, got, 4));
    CHECK(close_values(got, q_want, 4, 2e-6f));

    const float text_values[] = {1,2,3,4};
    const float zero_cos[] = {0,0};
    const float one_sin[] = {1,1};
    h3_gpu_tensor *text_q = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *text_k = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *text_cos = h3_gpu_tensor_from_f32(gpu, zero_cos, 2);
    h3_gpu_tensor *text_sin = h3_gpu_tensor_from_f32(gpu, one_sin, 2);
    CHECK(text_q && text_k && text_cos && text_sin);
    CHECK(h3_gpu_tensor_write_f32(text_q, text_values, 4));
    CHECK(h3_gpu_tensor_write_f32(text_k, text_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_rope_text_bf16(gpu, text_q, text_k, text_cos, text_sin,
                                1, 1, 1, 4));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(text_q, got, 4));
    const float text_want[] = {-3,-4,1,2};
    CHECK(close_values(got, text_want, 4, 0.0f));
    CHECK(h3_gpu_tensor_write_f32(text_q, text_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_head_rms_norm_bf16(gpu, text_q, bweight, 1, 1, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(text_q, got, 4));
    CHECK(close_values(got, q_want, 4, 0.01f));
    h3_gpu_tensor *text_q_out = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *text_k_out = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bzero_cos = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *bone_sin = h3_gpu_tensor_new_bf16(gpu, 2);
    CHECK(text_q_out && text_k_out && bzero_cos && bone_sin);
    CHECK(h3_gpu_tensor_write_f32(text_q, text_values, 4));
    CHECK(h3_gpu_tensor_write_f32(text_k, text_values, 4));
    CHECK(h3_gpu_tensor_write_f32(bzero_cos, zero_cos, 2));
    CHECK(h3_gpu_tensor_write_f32(bone_sin, one_sin, 2));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_text_qk_rope_bf16(gpu, text_q_out, text_k_out, text_q,
        text_k, bweight, bweight, bzero_cos, bone_sin, 1, 1, 1, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(text_q_out, got, 4));
    const float text_norm_want[] = {-3*q_inverse,-4*q_inverse,
                                     1*q_inverse, 2*q_inverse};
    CHECK(close_values(got, text_norm_want, 4, 0.01f));

    const float pool_input[] = {
        1,2, 3,4, 5,6, 7,8
    };
    const uint32_t pairs_values[] = {0,0, 1,2, 3,3};
    const uint32_t baseline_index_values[] = {UINT32_MAX, 0, UINT32_MAX};
    h3_gpu_tensor *pool_source = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *pool_output = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *original = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *baseline = h3_gpu_tensor_new_bf16(gpu, 2);
    h3_gpu_tensor *pairs = h3_gpu_tensor_from_u32(gpu, pairs_values, 6);
    h3_gpu_tensor *baseline_indices = h3_gpu_tensor_from_u32(
        gpu, baseline_index_values, 3);
    CHECK(pool_source && pool_output && original && baseline && pairs && baseline_indices);
    CHECK(h3_gpu_tensor_write_f32(pool_source, pool_input, 8));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_token_pool_bf16(gpu, pool_output, pool_source, 0, original,
        0, baseline, 0, baseline_indices, pairs, 4, 3, 1, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(pool_output, got, 6));
    const float pooled_want[] = {1,2, 4,5, 7,8};
    CHECK(close_values(got, pooled_want, 6, 0.0f));

    const float reduced_values[] = {10,20, 6,8, 70,80};
    const uint32_t parents_values[] = {0,1,1,2};
    h3_gpu_tensor *reduced = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *expanded = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *parents = h3_gpu_tensor_from_u32(gpu, parents_values, 4);
    CHECK(reduced && expanded && parents);
    CHECK(h3_gpu_tensor_write_f32(reduced, reduced_values, 6));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_token_expand_delta_bf16(gpu, expanded, original, 0, reduced,
        baseline, 0, baseline_indices, parents, 4, 3, 1, 2, 1, 1.0f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(expanded, got, 8));
    const float expanded_want[] = {10,20, 5,7, 7,9, 70,80};
    CHECK(close_values(got, expanded_want, 8, 0.0f));

    h3_gpu_tensor_free(parents); h3_gpu_tensor_free(expanded);
    h3_gpu_tensor_free(reduced); h3_gpu_tensor_free(baseline_indices);
    h3_gpu_tensor_free(pairs); h3_gpu_tensor_free(baseline);
    h3_gpu_tensor_free(original); h3_gpu_tensor_free(pool_output);
    h3_gpu_tensor_free(pool_source); h3_gpu_tensor_free(bone_sin);
    h3_gpu_tensor_free(bzero_cos); h3_gpu_tensor_free(text_k_out);
    h3_gpu_tensor_free(text_q_out); h3_gpu_tensor_free(text_sin);
    h3_gpu_tensor_free(text_cos); h3_gpu_tensor_free(text_k);
    h3_gpu_tensor_free(projection_scales);
    h3_gpu_tensor_free(projection_quantized);
    h3_gpu_tensor_free(projection_weight_scales);
    h3_gpu_tensor_free(projection_weight_i8);
    h3_gpu_tensor_free(projection_weight);
    h3_gpu_tensor_free(projection_input);
    h3_gpu_tensor_free(text_q); h3_gpu_tensor_free(multi_v);
    h3_gpu_tensor_free(multi_k); h3_gpu_tensor_free(multi_q);
    h3_gpu_tensor_free(multi_qkv); h3_gpu_tensor_free(v); h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(q); h3_gpu_tensor_free(bv); h3_gpu_tensor_free(bk);
    h3_gpu_tensor_free(bq); h3_gpu_tensor_free(bsin); h3_gpu_tensor_free(bcos);
    h3_gpu_tensor_free(bweight); h3_gpu_tensor_free(bqkv);
    h3_gpu_tensor_free(sine); h3_gpu_tensor_free(cosine);
    h3_gpu_tensor_free(weight); h3_gpu_tensor_free(qkv); h3_gpu_free(gpu);
    puts("ok: CUDA RoPE and token transforms");
    return 0;
}
