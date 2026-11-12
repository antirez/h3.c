#ifndef H3_DIT_H
#define H3_DIT_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_dit h3_dit;

typedef void (*h3_dit_progress)(const char *phase, int completed, int total,
                                void *opaque);

typedef int (*h3_dit_preview)(int completed_steps, int total_steps,
                              const float *video_latent,
                              size_t video_elements, void *opaque);

/* Load a text-only FL2VA transformer. Text refinement and AdaLN precomputation
 * happen before the persistent core is loaded. SSD streaming retains only the
 * small block norms and two alternating BF16 matrix slots. */
h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load FL2VA/Ref2VA packing. Condition inputs are already patchified F32 row
 * sources: visual rows have width 96 and audio rows width 32. Their element
 * counts must exactly match layout.img_cond_rows/audio_cond_rows. */
h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Load the image transformer (diffusers MiniMax-H3-Image layout). The packed
 * layout is [text | reference rows | latent rows] for a latent_height x
 * latent_width canvas; the schedule is the fixed six-step grid.
 * `references` are reference_count image references (H3_LAYOUT_REF_IMAGE,
 * one latent frame each, even sides of at most 1024) and condition_rows their
 * noised, patchified F32 rows, [R, 96] with R the sum of the references'
 * latent_h * latent_w / 4 (condition_elements = R * 96), written once at load
 * and held for all steps. Pass NULL, 0, NULL, 0 for text-to-image.
 * `active_blocks` thins the 50 blocks as for video. Options have the video
 * loaders' meaning. */
h3_dit *h3_dit_load_image(const char *weight_directory,
                          const char *shader_source_path,
                          const h3_text_embedding *text,
                          int latent_height, int latent_width,
                          const h3_layout_ref *references,
                          size_t reference_count,
                          const float *condition_rows,
                          size_t condition_elements,
                          unsigned active_blocks,
                          int ssd_streaming,
                          int use_slower_bf16_mlp,
                          int use_slower_bf16_qkv,
                          int use_slower_bf16_attention_output,
                          int use_slower_row_major_attention_output,
                          int use_slower_unfused_int8_inputs,
                          int use_slower_unfused_qkv_rope,
                          int use_slower_scalar_qkv_rms,
                          int use_slower_uncached_int8_scales,
                          int use_slower_dynamic_fc1_k,
                          int use_slower_grouped_quantizer,
                          int use_int8_row_fc2,
                          h3_dit_progress progress, void *progress_opaque,
                          char *error, size_t error_size);
void h3_dit_free(h3_dit *dit);

/* One image velocity evaluation with output head `step`. latent_rows is the
 * patchified F32 [rows, 96] canvas, velocity_rows receives F32 [rows, 96]
 * straight from the F32 head with no BF16 rounding. */
int h3_dit_forward_image(h3_dit *dit, int step, const float *latent_rows,
                         float *velocity_rows, char *error, size_t error_size);

/* Rows of the generated image (P) of an image transformer, else 0. */
size_t h3_dit_image_rows(const h3_dit *dit);

/* Sequence row of every video row source, in source order: the rows of the
 * COND, REF_IMAGE and VIDEO segments as the layout lists them, which is the
 * order of the F32 video input (condition rows, then generated rows). Writes
 * at most `capacity` entries and returns the total count. The mapped patch
 * projection of the DiT consumes exactly this map. */
size_t h3_dit_video_source_rows(const h3_layout *layout, uint32_t *rows,
                                size_t capacity);

/* Number of distinct timestep rows of the AdaLN tables: the six steps plus
 * one for the reference rows of an edit. */
uint32_t h3_dit_time_rows(const h3_dit *dit);

/* The AdaLN row map the fused kernels use at `step` (time row * 3 + modality
 * tag for each of the `count` rows of the packed sequence), recomputed from
 * the loaded schedule and layout. text_tags is one tag per text row or NULL. */
int h3_dit_read_row_map(const h3_dit *dit, int step, const uint8_t *text_tags,
                        size_t text_tag_count, uint32_t *rows, size_t count);

/* Copy the F32 final hidden state [rows, 5376] that fed the output head of
 * the last image forward, for diagnostics. */
int h3_dit_read_final_hidden(const h3_dit *dit, float *values,
                             size_t elements);

/* Copy the refined text rows [tokens, 5376] as BF16, for diagnostics. */
int h3_dit_read_refined_text(const h3_dit *dit, uint16_t *values,
                             size_t elements);

/* Reset mutable sampler state and replace seed-dependent condition rows before
 * reusing an otherwise identical prepared transformer. */
int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size);

size_t h3_dit_video_elements(const h3_dit *dit);
size_t h3_dit_audio_elements(const h3_dit *dit);

/* One raw data-ward velocity evaluation. Input/output video layout is
 * [24,T,H,W], audio is [32,2,T], all F32 on the host boundary. */
int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size);

/* Run every configured RES multistep update in place. Solver state remains
 * F32 between transformer evaluations, matching the released MLX path. */
int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size);

/* Current serving sampler: independent video/audio shifted Euler grids. */
int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size);

/* Preview variant. The callback runs after every Euler transition with the
 * current channel-major F32 video latent. It is deliberately opt-in because it
 * introduces a synchronization point after each step. */
int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size);

/* Build the velocity-evaluation mask used by the serving sampler. Returns the
 * evaluation count, or -1 for invalid arguments. Public internally so the
 * quality-tuned aggressive schedule remains pinned by cheap host tests. */
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count);

int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats);

/* The weight path this transformer runs: "stream" (SSD streaming, BF16),
 * "int8" (int8 MLP, QKV and attention output), "bf16" (all resident BF16) or
 * "mixed". */
const char *h3_dit_weight_mode(const h3_dit *dit);

/* Exact row-order conversions, public internally so they can be pinned by
 * cheap tests independently of the 62 GiB checkpoint. */
int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements);
int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements);
int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements);
int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements);

#endif
