#ifndef H3_IMAGE_PIPELINE_H
#define H3_IMAGE_PIPELINE_H

#include "h3_dit.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#define H3_IMAGE_MAX_PROMPT_TOKENS 4096
#define H3_IMAGE_LATENT_CHANNELS 24

/* HF `add_special_tokens=False` ids, cut to H3_IMAGE_MAX_PROMPT_TOKENS. An
 * empty prompt is an error: image prompts are never padded. */
int h3_image_tokenize(const h3_tokenizer *tokenizer, const char *prompt,
                      uint32_t **ids, size_t *count,
                      char *error, size_t error_size);

/* Tokenize with <model_directory>/tokenizer and run the language layers of
 * <model_directory>/text_encoder. ids (optional) receives the token ids, to be
 * released with h3_tokenizer_ids_free(). */
int h3_image_encode_prompt(const char *model_directory, const char *prompt,
                           h3_text_progress progress, void *progress_opaque,
                           uint32_t **ids, size_t *count,
                           h3_text_embedding *text,
                           char *error, size_t error_size);

/* Initial latent noise [24][latent_height][latent_width] from the seed. */
void h3_image_noise(uint64_t seed, int latent_height, int latent_width,
                    float *noise);

/* Velocity of the image at `step` for patchified F32 rows [P][96]. */
typedef int (*h3_image_velocity)(void *opaque, int step, const float *rows,
                                 float *velocity, char *error,
                                 size_t error_size);
/* Called with completed in [0, H3_IMAGE_STEPS]; a nonzero return cancels. */
typedef int (*h3_image_step_progress)(int completed, int total, void *opaque);

/* The six-step Euler sampler on F32 patchified rows (T = 1). noise and
 * latent_out are [24][latent_height][latent_width] and may alias. */
int h3_image_sample(int latent_height, int latent_width, const float *noise,
                    h3_image_velocity velocity, void *velocity_opaque,
                    h3_image_step_progress progress, void *progress_opaque,
                    float *latent_out, char *error, size_t error_size);

/* h3_image_velocity of a loaded image transformer. */
int h3_image_dit_velocity(void *dit, int step, const float *rows,
                          float *velocity, char *error, size_t error_size);

/* Planar [3][h][w] bytes to interleaved [h][w][3]. */
void h3_image_interleave_rgb(const uint8_t *planar, int width, int height,
                             uint8_t *interleaved);

#endif
