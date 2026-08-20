#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int close_array(const float *got, const float *want, size_t count,
                       float tolerance) {
    for (size_t index = 0; index < count; index++)
        if (fabsf(got[index] - want[index]) > tolerance) return 0;
    return 1;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu);
    const float input[] = {-2.0f, -0.5f, 0.5f, 2.0f};
    const float other[] = {1.0f, 2.0f, 3.0f, 4.0f};
    h3_gpu_tensor *x = h3_gpu_tensor_from_f32(gpu, input, 4);
    h3_gpu_tensor *y = h3_gpu_tensor_from_f32(gpu, other, 4);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, 4);
    h3_gpu_tensor *bx = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *by = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bout = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(x && y && out && bx && by && bout);
    CHECK(h3_gpu_tensor_write_f32(bx, input, 4));
    CHECK(h3_gpu_tensor_write_f32(by, other, 4));

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_silu_f32(gpu, out, x, 4));
    CHECK(h3_gpu_submit(gpu));
    float got[4];
    CHECK(h3_gpu_tensor_read_f32(out, got, 4));
    float want_silu[4];
    for (size_t i = 0; i < 4; i++)
        want_silu[i] = input[i] / (1.0f + expf(-input[i]));
    CHECK(close_array(got, want_silu, 4, 1e-6f));

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_add_scaled_f32(gpu, out, x, y, 0.5f, -2.0f, 4));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(out, got, 4));
    const float want_scaled[] = {-3.0f, -4.25f, -5.75f, -7.0f};
    CHECK(close_array(got, want_scaled, 4, 1e-7f));

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_add_bf16(gpu, bout, bx, by, 4));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(bout, got, 4));
    const float want_add[] = {-1.0f, 1.5f, 3.5f, 6.0f};
    CHECK(close_array(got, want_add, 4, 0.01f));

    const float fused_values[] = {-2, -0.5f, 0.5f, 2, 1, 2, 3, 4};
    const float unit_values[] = {1,1,1,1};
    const float magnitude_values[] = {2};
    h3_gpu_tensor *fused = h3_gpu_tensor_from_f32(gpu, fused_values, 8);
    h3_gpu_tensor *bfused = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *units = h3_gpu_tensor_from_f32(gpu, unit_values, 4);
    h3_gpu_tensor *bunits = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *magnitude = h3_gpu_tensor_from_f32(gpu, magnitude_values, 1);
    CHECK(fused && bfused && units && bunits && magnitude);
    CHECK(h3_gpu_tensor_write_f32(bfused, fused_values, 8));
    CHECK(h3_gpu_tensor_write_f32(bunits, unit_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_clip_f32(gpu, out, x, 4, -1.0f, 1.0f));
    CHECK(h3_gpu_scale_add_f32(gpu, out, x, y, units, 1, 4));
    CHECK(h3_gpu_swiglu_f32(gpu, out, fused, 1, 4));
    CHECK(h3_gpu_swiglu_bf16(gpu, bout, bfused, 1, 4));
    CHECK(h3_gpu_geglu_f32(gpu, out, x, y, 4));
    CHECK(h3_gpu_snake1d_f32(gpu, out, x, units, 1, 1, 4));
    CHECK(h3_gpu_weight_norm_f32(gpu, out, x, magnitude, 1, 4));
    CHECK(h3_gpu_sub_bf16(gpu, bout, by, bx, 4));
    CHECK(h3_gpu_silu_bf16(gpu, bout, bx, 4));
    CHECK(h3_gpu_gelu_bf16(gpu, bout, bx, 4, 0));
    CHECK(h3_gpu_gelu_bf16(gpu, bout, bx, 4, 1));
    CHECK(h3_gpu_silu_mul_bf16(gpu, bout, bx, by, 4));
    CHECK(h3_gpu_cast_f32_to_bf16(gpu, bout, x, 4));
    CHECK(h3_gpu_cast_bf16_to_f32(gpu, out, bout, 4));
    CHECK(h3_gpu_euler_bf16(gpu, out, 0, bx, by, 4, 0.1f, 0.5f));
    CHECK(h3_gpu_rms_norm_bf16(gpu, bout, bx, bunits, 1, 4, 1e-5f));
    CHECK(h3_gpu_layer_norm_bf16(gpu, bout, bx, bunits, bunits,
                                 1, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));

    const float norm_input[] = {1, 2, 3, 4, -1, -2, -3, -4};
    const float weights[] = {1, 1, 1, 1};
    const float biases[] = {0, 0, 0, 0};
    h3_gpu_tensor *nx = h3_gpu_tensor_from_f32(gpu, norm_input, 8);
    h3_gpu_tensor *nw = h3_gpu_tensor_from_f32(gpu, weights, 4);
    h3_gpu_tensor *nb = h3_gpu_tensor_from_f32(gpu, biases, 4);
    h3_gpu_tensor *no = h3_gpu_tensor_new_f32(gpu, 8);
    CHECK(nx && nw && nb && no);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_rms_norm_f32(gpu, no, nx, nw, 2, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float norm_got[8];
    CHECK(h3_gpu_tensor_read_f32(no, norm_got, 8));
    float inverse = 1.0f / sqrtf(7.5f + 1e-5f);
    for (size_t i = 0; i < 8; i++) CHECK(fabsf(norm_got[i] - norm_input[i] * inverse) < 2e-6f);

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_layer_norm_f32(gpu, no, nx, nw, nb, 2, 4, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(no, norm_got, 8));
    CHECK(fabsf(norm_got[0] + 1.341635f) < 2e-5f);
    CHECK(fabsf(norm_got[3] - 1.341635f) < 2e-5f);

    const uint32_t row_map_values[] = {1, 0};
    const float modulation_values[] = {
        0,0,0,0,  0.1f,0.2f,0.3f,0.4f,  0.5f,0.5f,0.5f,0.5f,
        0,0,0,0, -0.1f,-0.2f,-0.3f,-0.4f,  2,2,2,2
    };
    h3_gpu_tensor *row_map = h3_gpu_tensor_from_u32(gpu, row_map_values, 2);
    h3_gpu_tensor *mod = h3_gpu_tensor_from_f32(gpu, modulation_values, 24);
    h3_gpu_tensor *bnx = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *bnw = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bmod = h3_gpu_tensor_new_bf16(gpu, 24);
    h3_gpu_tensor *bgated = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *bnout = h3_gpu_tensor_new_bf16(gpu, 8);
    CHECK(row_map && mod && bnx && bnw && bmod && bgated && bnout);
    CHECK(h3_gpu_tensor_write_f32(bnx, norm_input, 8));
    CHECK(h3_gpu_tensor_write_f32(bnw, weights, 4));
    CHECK(h3_gpu_tensor_write_f32(bmod, modulation_values, 24));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_adaln_f32(gpu, no, nx, nw, mod, row_map,
                            2, 4, 3, 1, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(no, norm_got, 8));
    CHECK(fabsf(norm_got[0] - (norm_input[0] * inverse * 3.0f - 0.1f)) < 2e-5f);
    CHECK(fabsf(norm_got[4] - (norm_input[4] * inverse * 1.5f + 0.1f)) < 2e-5f);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_gate_f32(gpu, no, nx, nx, mod, row_map, 2, 4, 3, 2));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(no, norm_got, 8));
    CHECK(norm_got[0] == 3.0f && norm_got[4] == -1.5f);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_adaln_bf16_offset(gpu, bnout, bnx, 0, bnw, bmod, row_map,
                                    2, 4, 3, 1, 2, 1e-5f));
    CHECK(h3_gpu_gate_adaln_bf16(gpu, bgated, bnout, bnx, bnx, bnw,
        bmod, bmod, row_map, 2, 4, 3, 2, 1, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    const float identity4[] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };
    h3_gpu_tensor *bidentity = h3_gpu_tensor_new_bf16(gpu, 16);
    h3_gpu_tensor *binverse = h3_gpu_tensor_new_f32(gpu, 2);
    h3_gpu_tensor *blinear = h3_gpu_tensor_new_bf16(gpu, 8);
    h3_gpu_tensor *bquantized = h3_gpu_tensor_new_i8(gpu, 12);
    h3_gpu_tensor *bscales = h3_gpu_tensor_new_f32(gpu, 3);
    CHECK(bidentity && binverse && blinear && bquantized && bscales);
    CHECK(h3_gpu_tensor_write_f32(bidentity, identity4, 16));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_adaln_linear_bf16(gpu, blinear, binverse, bnx, 0, bnw,
        bmod, row_map, bidentity, NULL, 2, 4, 4, 3, 1, 2, 1e-5f));
    CHECK(h3_gpu_gate_adaln_quantize_int8(gpu, bgated, bquantized, bscales,
        bnx, bnx, bnw, bmod, bmod, row_map, 2, 3, 4, 3, 2, 1, 2, 1e-5f));
    CHECK(h3_gpu_submit(gpu));
    float inverse_values[2];
    CHECK(h3_gpu_tensor_read_f32(binverse, inverse_values, 2));
    CHECK(fabsf(inverse_values[0] - inverse) < 2e-5f);

    const float embedding_values[] = {1,2, 3,4, 5,6};
    const uint32_t ids[] = {2, 0, 9};
    h3_gpu_tensor *embedding = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *embedding_out = h3_gpu_tensor_new_bf16(gpu, 6);
    h3_gpu_tensor *token_ids = h3_gpu_tensor_from_u32(gpu, ids, 3);
    CHECK(embedding && embedding_out && token_ids);
    CHECK(h3_gpu_tensor_write_f32(embedding, embedding_values, 6));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_embedding_bf16(gpu, embedding_out, embedding, token_ids,
                                3, 3, 2));
    CHECK(h3_gpu_submit(gpu));
    float embedding_got[6];
    CHECK(h3_gpu_tensor_read_f32(embedding_out, embedding_got, 6));
    const float embedding_want[] = {5,6, 1,2, 0,0};
    CHECK(close_array(embedding_got, embedding_want, 6, 0.0f));

    h3_gpu_tensor_free(bscales); h3_gpu_tensor_free(bquantized);
    h3_gpu_tensor_free(blinear); h3_gpu_tensor_free(binverse);
    h3_gpu_tensor_free(bidentity);
    h3_gpu_tensor_free(token_ids); h3_gpu_tensor_free(embedding_out);
    h3_gpu_tensor_free(embedding); h3_gpu_tensor_free(bnout);
    h3_gpu_tensor_free(bgated); h3_gpu_tensor_free(bmod);
    h3_gpu_tensor_free(bnw); h3_gpu_tensor_free(bnx); h3_gpu_tensor_free(mod);
    h3_gpu_tensor_free(row_map); h3_gpu_tensor_free(magnitude);
    h3_gpu_tensor_free(bunits); h3_gpu_tensor_free(units);
    h3_gpu_tensor_free(bfused); h3_gpu_tensor_free(fused);
    h3_gpu_tensor_free(no);
    h3_gpu_tensor_free(nb); h3_gpu_tensor_free(nw);
    h3_gpu_tensor_free(nx); h3_gpu_tensor_free(bout); h3_gpu_tensor_free(by);
    h3_gpu_tensor_free(bx); h3_gpu_tensor_free(out); h3_gpu_tensor_free(y);
    h3_gpu_tensor_free(x); h3_gpu_free(gpu);
    puts("ok: CUDA elementwise and normalization primitives");
    return 0;
}
