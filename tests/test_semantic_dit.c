#include "h3_dit.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    TEXT_ROWS = 6,
    TEXT_WIDTH = 5120,
    LATENT_T = 7,
    LATENT_H = 16,
    LATENT_W = 16,
    AUDIO_T = 37,
    VIDEO_ELEMENTS = 24 * LATENT_T * LATENT_H * LATENT_W,
    AUDIO_ELEMENTS = 32 * 2 * AUDIO_T
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_semantic_dit.c: %s\n", message);
    exit(1);
}

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 0 || completed == total || completed % 10 == 0)
        fprintf(stderr, "semantic DiT %-24s %d/%d\n", phase, completed, total);
}

static void *load_tensor(const h3_st_header *header, const char *name,
                         h3_dtype dtype, size_t count, size_t element_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != (uint64_t)count)
        die("semantic fixture has the wrong schema");
    void *values = malloc(count * element_size);
    if (!values) die("out of memory loading semantic fixture");
    char error[512];
    if (!h3_st_read_data(header, tensor, values, count * element_size,
                         error, sizeof(error))) die(error);
    return values;
}

static void compare(const float *got, const float *want, size_t count) {
    double maximum = 0.0, scale = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - (double)want[index];
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs((double)want[index]) > scale) scale = fabs((double)want[index]);
        square_error += delta * delta;
        square_value += (double)want[index] * (double)want[index];
    }
    double relative_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double relative_l2 = sqrt(square_error /
        (square_value > 1e-24 ? square_value : 1e-24));
    printf("256x256x22 final latent: rel-max %.6g abs %.6g rel-L2 %.6g\n",
           relative_max, maximum, relative_l2);
    if (relative_max >= 0.15 || relative_l2 >= 0.10)
        die("native 22-frame latent diverges from MLX");
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *prompt_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_prompt_bf16.safetensors";
    const char *semantic_path = argc > 3 ? argv[3] :
        "misc/fixtures/h3_semantic_256x22_seed42.safetensors";
    char error[512];
    h3_st_header prompt, semantic;
    if (!h3_st_read_header(prompt_path, &prompt, error, sizeof(error))) die(error);
    if (!h3_st_read_header(semantic_path, &semantic, error, sizeof(error)))
        die(error);
    uint16_t *text_values = load_tensor(&prompt, "x.output", H3_DTYPE_BF16,
                                         TEXT_ROWS * TEXT_WIDTH,
                                         sizeof(*text_values));
    float *video = load_tensor(&semantic, "x.video_initial", H3_DTYPE_F32,
                               VIDEO_ELEMENTS, sizeof(*video));
    float *audio = load_tensor(&semantic, "x.audio_initial", H3_DTYPE_F32,
                               AUDIO_ELEMENTS, sizeof(*audio));
    float *video_want = load_tensor(&semantic, "x.video_final", H3_DTYPE_F32,
                                    VIDEO_ELEMENTS, sizeof(*video_want));
    h3_text_embedding text = {TEXT_ROWS, TEXT_WIDTH, text_values, {0}};
    h3_layout_spec spec = {TEXT_ROWS, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, 22,
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
        die("semantic DiT geometry mismatch");
    if (!h3_dit_denoise(dit, video, audio, progress, NULL,
                        error, sizeof(error))) die(error);
    compare(video, video_want, VIDEO_ELEMENTS);
    h3_gpu_stats stats;
    if (!h3_dit_get_gpu_stats(dit, &stats)) die("cannot read GPU stats");
    printf("semantic DiT: %.3f GiB cumulative, %.3f GPU seconds, "
           "%llu submissions\n",
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           stats.gpu_seconds, (unsigned long long)stats.submissions);
    h3_dit_free(dit);
    h3_layout_free(&layout);
    h3_st_free_header(&prompt);
    h3_st_free_header(&semantic);
    free(text_values);
    free(video);
    free(audio);
    free(video_want);
    puts("ok: native Metal matches MLX at the first trained temporal chunk");
    return 0;
}
