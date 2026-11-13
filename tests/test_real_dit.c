#include "h3_dit.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_dit.c: %s\n", message);
    exit(1);
}

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 0 || completed == total || completed % 10 == 0)
        fprintf(stderr, "production DiT %-24s %d/%d\n", phase, completed, total);
}

static uint16_t *load_bf16(const h3_st_header *header, const char *name,
                            size_t expected) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != expected) die("bad fixture tensor");
    uint16_t *values = malloc(expected * sizeof(*values));
    if (!values) die("out of memory loading fixture");
    char error[512];
    if (!h3_st_read_data(header, tensor, values, expected * sizeof(*values),
                         error, sizeof(error))) die(error);
    return values;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void compare(const char *label, const float *got, const uint16_t *want,
                    size_t count, double bound) {
    double maximum = 0.0, scale = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        double expected = bf16_to_f32(want[index]);
        double delta = (double)got[index] - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        square_error += delta * delta;
        square_value += expected * expected;
    }
    double relative = maximum / (scale > 1e-12 ? scale : 1e-12);
    double l2 = sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24));
    printf("%-18s rel-max %.6g abs %.6g rel-L2 %.6g\n",
           label, relative, maximum, l2);
    if (relative >= bound || l2 >= bound) die("production DiT parity bound exceeded");
}

int main(int argc, char **argv) {
    /* This is the close-reference integration test. The production fast path
     * deliberately changes MLP operation boundaries and is covered by the
     * fused-operation check in test_bf16 plus semantic video validation. */
    setenv("H3_DISABLE_FUSED_MLP", "1", 1);
    setenv("H3_DIT_F32_FINAL", "1", 1);
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *block_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_dit_block0_bf16.safetensors";
    const char *step_path = argc > 3 ? argv[3] :
        "misc/fixtures/h3_real_dit_step0_bf16.safetensors";
    const char *denoise_path = argc > 4 ? argv[4] :
        "misc/fixtures/h3_real_dit_denoise20_bf16.safetensors";
    char error[512];
    h3_st_header block, step, denoise;
    if (!h3_st_read_header(block_path, &block, error, sizeof(error))) die(error);
    if (!h3_st_read_header(step_path, &step, error, sizeof(error))) die(error);
    if (!h3_st_read_header(denoise_path, &denoise, error, sizeof(error))) die(error);
    enum { TEXT_ROWS = 6, VIDEO_ELEMENTS = 24 * 2 * 2 * 2,
           AUDIO_ELEMENTS = 32 * 2 * 8 };
    uint16_t *text_bf = load_bf16(&block, "x.text_qwen", TEXT_ROWS * 5120);
    uint16_t *video_bf = load_bf16(&block, "x.video_latent", VIDEO_ELEMENTS);
    uint16_t *audio_bf = load_bf16(&block, "x.audio_latent", AUDIO_ELEMENTS);
    uint16_t *video_want = load_bf16(&step, "x.video_output", VIDEO_ELEMENTS);
    uint16_t *audio_want = load_bf16(&step, "x.audio_output", AUDIO_ELEMENTS);
    uint16_t *video_final = load_bf16(&denoise, "x.video_after_20",
                                      VIDEO_ELEMENTS);
    uint16_t *audio_final = load_bf16(&denoise, "x.audio_after_20",
                                      AUDIO_ELEMENTS);
    float video[VIDEO_ELEMENTS], audio[AUDIO_ELEMENTS];
    float video_got[VIDEO_ELEMENTS], audio_got[AUDIO_ELEMENTS];
    for (size_t index = 0; index < VIDEO_ELEMENTS; index++)
        video[index] = bf16_to_f32(video_bf[index]);
    for (size_t index = 0; index < AUDIO_ELEMENTS; index++)
        audio[index] = bf16_to_f32(audio_bf[index]);
    h3_text_embedding text = {TEXT_ROWS, 5120, text_bf, {0}};
    h3_layout_spec spec = {TEXT_ROWS, 2, 2, 2, 8, 5,
                           NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) die(error);
    h3_sigma_schedule sigmas;
    if (!h3_schedule_build(20, &sigmas)) die("cannot build sampler schedule");
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/transformer", model_root);
    const h3_dit_tuning tuning = {
        .use_slower_bf16_qkv = 1, .use_slower_bf16_attention_output = 1,
        .use_slower_row_major_attention_output = 1,
        .use_slower_unfused_int8_inputs = 1,
        .use_slower_unfused_qkv_rope = 1, .use_slower_scalar_qkv_rms = 1,
        .use_slower_uncached_int8_scales = 1
    };
    h3_dit *dit = h3_dit_load_t2va(weights, "h3_shaders.metal", &text,
                                    &layout, &sigmas, 50, 1, 0, &tuning, 1.0f,
                                    progress, NULL,
                                    error, sizeof(error));
    if (!dit) die(error);
    if (h3_dit_video_elements(dit) != VIDEO_ELEMENTS ||
        h3_dit_audio_elements(dit) != AUDIO_ELEMENTS)
        die("production DiT latent geometry mismatch");
    if (!h3_dit_forward(dit, 0, video, audio, video_got, audio_got,
                        error, sizeof(error))) die(error);
    compare("video velocity", video_got, video_want, VIDEO_ELEMENTS, 0.25);
    compare("audio velocity", audio_got, audio_want, AUDIO_ELEMENTS, 0.30);
    if (!h3_dit_denoise(dit, video, audio, progress, NULL,
                        error, sizeof(error))) die(error);
    compare("video latent 20", video, video_final, VIDEO_ELEMENTS, 1.0);
    compare("audio latent 20", audio, audio_final, AUDIO_ELEMENTS, 1.0);
    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats)) die("cannot read GPU stats");
    printf("persistent model: %.3f GiB cumulative, %.3f GPU seconds, "
           "%llu submissions\n",
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           stats.gpu_seconds, (unsigned long long)stats.submissions);
    if (stats.submissions != 75)
        die("unexpected production setup/forward submission count");
    h3_dit_free(dit);
    h3_layout_free(&layout);
    h3_st_free_header(&block);
    h3_st_free_header(&step);
    h3_st_free_header(&denoise);
    free(text_bf); free(video_bf); free(audio_bf);
    free(video_want); free(audio_want);
    free(video_final); free(audio_final);
    puts("ok: persistent production DiT step matches the MLX latent oracle");
    return 0;
}
