#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

#include "h3_dit_names.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_DIT_BLOCKS 50u
#define H3_DIT_HIDDEN 5376u
#define H3_DIT_TIME_DIM 2688u
#define H3_DIT_MODALITIES 3u
#define H3_DIT_ADALN_SLOTS 6u
#define H3_DIT_TIME_INPUT 256u

typedef struct h3_dit_schedule h3_dit_schedule;

/* Sinusoidal timestep features [cos | sin] of one time value, H3_DIT_TIME_INPUT
 * floats. Public internally so a cheap test can pin it. */
void h3_dit_time_features(float time, float *row);

typedef void (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Order of the AdaLN activation on the timestep embedding. The video
 * checkpoint was validated with the BF16 cast first; the image reference
 * applies SiLU to the F32 embedding and casts afterwards. */
typedef enum {
    H3_DIT_ADALN_CAST_THEN_SILU = 0,
    H3_DIT_ADALN_SILU_THEN_CAST
} h3_dit_adaln_order;

/* Timestep features [rows, H3_DIT_TIME_INPUT] through the time embedder MLP:
 * F32 [rows, H3_DIT_TIME_DIM]. */
h3_gpu_tensor *h3_dit_time_embed(const h3_weight_store *weights,
                                 const h3_dit_names *names, h3_gpu *gpu,
                                 uint32_t rows, const float *features,
                                 char *error, size_t error_size);

/* BF16 [rows, H3_DIT_TIME_DIM] input of every AdaLN projection, from the F32
 * embedding of h3_dit_time_embed. */
h3_gpu_tensor *h3_dit_adaln_input(h3_gpu *gpu, const h3_gpu_tensor *time,
                                  uint32_t rows, h3_dit_adaln_order order,
                                  char *error, size_t error_size);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, const h3_dit_names *names, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, h3_dit_adaln_order order,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
void h3_dit_schedule_free(h3_dit_schedule *schedule);

int h3_dit_schedule_steps(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step);
const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block);
double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block);
void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count);
const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule);

/* Build the row map consumed by the fused AdaLN/gate kernels. text_tags may be
 * NULL (all tag 1), or one tag per text row. Qwen vision presentation spans use
 * tag 0. Segment kinds select target/condition timesteps and modality tags. */
int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count);

#endif
