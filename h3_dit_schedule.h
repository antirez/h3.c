#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_lora.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_DIT_BLOCKS 50u
#define H3_DIT_HIDDEN 5376u
#define H3_DIT_TIME_DIM 2688u
#define H3_DIT_MODALITIES 3u
#define H3_DIT_ADALN_SLOTS 6u

typedef struct h3_dit_schedule h3_dit_schedule;

typedef void (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, const h3_lora_set *loras, size_t *peak_scratch,
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
