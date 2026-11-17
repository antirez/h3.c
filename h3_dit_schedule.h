#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

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
#define H3_DIT_ADALN_OUTPUT (H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN)
#define H3_DIT_FINAL_ADALN_OUTPUT (2u * H3_DIT_HIDDEN)
/* Timesteps of the clean visual and audio condition rows. */
#define H3_DIT_VISUAL_CONDITION_TIME 0.999f
#define H3_DIT_AUDIO_CONDITION_TIME 1.0f

typedef struct h3_dit_schedule h3_dit_schedule;

typedef void (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
void h3_dit_schedule_free(h3_dit_schedule *schedule);

/* A packed (INT8) checkpoint carries precomputed AdaLN rows, one per timestep,
 * instead of the AdaLN projection weights. */
int h3_dit_schedule_is_packed(const h3_weight_store *weights);
/* The distinct timestep of every AdaLN row a schedule uses, in row order.
 * The caller frees *times. */
int h3_dit_schedule_timesteps(const h3_sigma_schedule *sigmas,
                              int visual_condition, int audio_condition,
                              float **times, uint32_t *count);
/* Packed AdaLN rows are stored per kernel regime: a schedule with R timestep
 * rows reads regime h3_dit_schedule_table_regime(R), whose rows were computed
 * in batches of h3_dit_schedule_regime_batch(regime). */
#define H3_DIT_TABLE_REGIMES 2u
unsigned h3_dit_schedule_table_regime(uint32_t rows);
uint32_t h3_dit_schedule_regime_batch(unsigned regime);
/* Packing: project count timesteps through every AdaLN layer in batches of
 * batch rows. sink receives the [count, H3_DIT_ADALN_OUTPUT] BF16 table of
 * each block, then (index H3_DIT_BLOCKS) the [count,
 * H3_DIT_FINAL_ADALN_OUTPUT] final table, and owns it. */
typedef int (*h3_dit_schedule_table_sink)(unsigned index,
                                          h3_gpu_tensor *table, void *opaque,
                                          char *error, size_t error_size);
int h3_dit_schedule_build_tables(const h3_weight_store *weights, h3_gpu *gpu,
                                 const float *times, uint32_t count,
                                 uint32_t batch,
                                 h3_dit_schedule_table_sink sink,
                                 void *opaque, char *error,
                                 size_t error_size);

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
