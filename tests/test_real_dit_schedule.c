#include "h3_dit_schedule.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BLOCK_OUTPUT = 3 * 6 * 5376 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_dit_schedule.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed % 5 == 0 || completed == total)
        fprintf(stderr, "native AdaLN precompute: %d/%d blocks\n",
                completed, total);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *fixture_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_dit_block0_bf16.safetensors";
    char weight_path[1024];
    snprintf(weight_path, sizeof(weight_path), "%s/FL2VA/transformer",
             model_root);
    char error[512];
    h3_weight_store *weights = h3_weight_store_open(weight_path, error,
                                                     sizeof(error));
    if (!weights) die(error);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    h3_sigma_schedule sigmas;
    if (!h3_schedule_build(20, &sigmas)) die("cannot build 20-step schedule");
    h3_dit_schedule *schedule = h3_dit_schedule_precompute(
        weights, h3_dit_names_for(H3_DIT_FAMILY_VIDEO), gpu, &sigmas, 1, 0,
        H3_DIT_ADALN_CAST_THEN_SILU, progress, NULL, error, sizeof(error));
    if (!schedule) die(error);
    if (h3_dit_schedule_steps(schedule) != 20 ||
        h3_dit_schedule_time_rows(schedule) != 40 ||
        h3_dit_schedule_video_row(schedule, 0) != 0 ||
        h3_dit_schedule_audio_row(schedule, 0) != 0 ||
        h3_dit_schedule_visual_condition_row(schedule, 0) != 39 ||
        h3_dit_schedule_video_row(schedule, 1) != 1 ||
        h3_dit_schedule_audio_row(schedule, 1) != 2)
        die("unexpected global timestep row layout");
    int keyframes[] = {0};
    h3_layout_spec spec = {6, 2, 2, 2, 8, 5,
                           keyframes, 1, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) die(error);
    uint8_t text_tags[] = {1, 0, 0, 1, 1, 1};
    uint32_t row_map[25];
    if (!h3_dit_schedule_row_map(schedule, 0, &layout, text_tags, 6,
                                  row_map, 25))
        die("cannot build step-0 row map");
    const uint32_t expected_text[] = {1, 0, 0, 1, 1, 1};
    for (size_t index = 0; index < 6; index++)
        if (row_map[index] != expected_text[index])
            die("bad multimodal text modulation row");
    if (row_map[6] != 117) die("bad visual-condition modulation row");
    for (size_t index = 7; index < 23; index++)
        if (row_map[index] != 2) die("bad audio modulation row");
    for (size_t index = 23; index < 25; index++)
        if (row_map[index] != 0) die("bad video modulation row");

    h3_st_header fixture;
    if (!h3_st_read_header(fixture_path, &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *gold = h3_st_find(&fixture, "x.modulation");
    if (!gold || gold->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(gold) != BLOCK_OUTPUT)
        die("fixture modulation tensor is absent or malformed");
    uint16_t *want = malloc((size_t)BLOCK_OUTPUT * sizeof(*want));
    uint16_t *got = malloc((size_t)BLOCK_OUTPUT * sizeof(*got));
    if (!want || !got) die("out of memory comparing modulation");
    if (!h3_st_read_data(&fixture, gold, want,
                         (size_t)BLOCK_OUTPUT * sizeof(*want),
                         error, sizeof(error))) die(error);
    if (!h3_gpu_tensor_read_bf16(h3_dit_schedule_block(schedule, 0), got,
                                 BLOCK_OUTPUT))
        die("cannot read precomputed block-0 modulation");
    double max_error = 0.0, max_value = 0.0, square_error = 0.0;
    double square_value = 0.0;
    for (size_t index = 0; index < BLOCK_OUTPUT; index++) {
        double actual = bf16_to_f32(got[index]);
        double expected = bf16_to_f32(want[index]);
        double delta = actual - expected;
        if (fabs(delta) > max_error) max_error = fabs(delta);
        if (fabs(expected) > max_value) max_value = fabs(expected);
        square_error += delta * delta;
        square_value += expected * expected;
    }
    double relative = max_error / (max_value > 1e-12 ? max_value : 1e-12);
    double relative_l2 = sqrt(square_error /
                              (square_value > 1e-24 ? square_value : 1e-24));
    printf("precomputed block-0 modulation: rel-max %.6g abs %.6g "
           "rel-L2 %.6g\n", relative, max_error, relative_l2);
    if (relative >= 0.025 || relative_l2 >= 0.025)
        die("precomputed modulation exceeds MLX parity bound");
    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(gpu, &stats)) die("cannot read GPU statistics");
    printf("AdaLN setup: %u time rows, %.3f GiB cumulative allocations, "
           "%.3f GPU seconds, %llu submissions\n",
           h3_dit_schedule_time_rows(schedule),
           (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           stats.gpu_seconds, (unsigned long long)stats.submissions);
    if (stats.submissions != 52)
        die("AdaLN setup did not preserve one-projection-at-a-time residency");

    free(want);
    free(got);
    h3_st_free_header(&fixture);
    h3_layout_free(&layout);
    h3_dit_schedule_free(schedule);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    puts("ok: persistent 20-step AdaLN schedule matches MLX step 0");
    return 0;
}
