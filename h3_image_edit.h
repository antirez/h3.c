#ifndef H3_IMAGE_EDIT_H
#define H3_IMAGE_EDIT_H

#include "h3_image.h"

#include <stddef.h>
#include <stdint.h>

/* The two resized copies of an upright reference, planar bytes [3][h][w], each
 * malloc'd (the caller frees them): the VAE copy (geometry vae_w x vae_h) and
 * the Qwen vision copy (vision_w x vision_h). Both resize with the PIL
 * Lanczos port; the vision copy's second stage (smart_resize) is bicubic in
 * the reference and Lanczos here. A size that does not change is copied. */
int h3_image_edit_copies(const uint8_t *planar, int width, int height,
                         const h3_image_reference_geometry *geometry,
                         uint8_t **vae, uint8_t **vision);

/* Reference latents [24][latent_h][latent_w] (normalized) and their
 * augmentation noise, same shape: t * x + (1.0f - t) * noise with t =
 * H3_REFERENCE_NOISE_AUG, patchified to rows [latent_h * latent_w / 4][96].
 * Both sides must be even. */
int h3_image_edit_condition_rows(const float *latents, const float *aug_noise,
                                 int latent_h, int latent_w, float *rows);

#endif
