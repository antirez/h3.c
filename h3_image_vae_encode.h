#ifndef H3_IMAGE_VAE_ENCODE_H
#define H3_IMAGE_VAE_ENCODE_H

#include "h3_gpu.h"
#include "h3_image.h"
#include "h3_image_vae.h"

#include <stddef.h>
#include <stdint.h>

/* Reference-image preprocessing and encoding for image editing, as the
 * diffusers MiniMax-H3-Image pipeline does it. Images are planar RGB bytes
 * [3][height][width]. */

/* The level reference latents are noised to: t in t * x + (1 - t) * noise. */
#define H3_REFERENCE_NOISE_AUG 0.999f

/* PIL Image.resize(resample=LANCZOS) byte for byte: 22-bit fixed-point
 * coefficients, a horizontal pass then a vertical pass, each rounded to bytes.
 * Both sizes must be positive. Returns 0 only when out of memory. */
int h3_reference_resize_lanczos(const uint8_t *planar, int height, int width,
                                int out_height, int out_width, uint8_t *out);

/* Top-left crops of planar [channels][height][width] data. */
void h3_reference_crop_u8(const uint8_t *planar, int channels, int height,
                          int width, int crop_height, int crop_width,
                          uint8_t *out);
void h3_reference_crop_f32(const float *planar, int channels, int height,
                           int width, int crop_height, int crop_width,
                           float *out);

/* Round to the nearest float16 (ties to even) and back. */
float h3_round_to_f16(float value);

/* DiagonalGaussianDistribution.sample() with the noise given: moments hold
 * count means then count log-variances; the log-variance is clamped to
 * [-30, 20] and sample = mean + exp(0.5 * logvar) * noise. */
void h3_posterior_sample(const float *moments, const float *noise,
                         size_t count, float *sample);

/* In place: (value - mean[channel]) / std[channel] over [channels][plane]. */
void h3_reference_normalize(float *latents, int channels, size_t plane,
                            const float *mean, const float *std);

/* MiniMaxH3Scheduler.scale_noise with a float32 timestep tensor:
 * t * latents + (1.0f - t) * noise, each operation rounded to float32. out may
 * alias an input. */
void h3_reference_scale_noise(const float *latents, const float *noise,
                              float t, size_t count, float *out);

typedef struct {
    /* Latents after the patch crop, normalized: [24][latent_height][latent_width],
     * both sides even. */
    int latent_height;
    int latent_width;
    float *latents;
    /* The encoder's moments of the 16-cropped image before sampling:
     * [48][moments_height][moments_width], means then log-variances. */
    int moments_height;
    int moments_width;
    float *moments;
    h3_gpu_stats gpu_stats;
} h3_reference_latents;

/* The reference VAE encoder step on an already resized image: crop top-left to
 * multiples of 16, ImageNet-normalize, encode in the released 256 px tiles,
 * sample the posterior (noise [24][height / 16][width / 16] given, or NULL to
 * draw it from seed with h3_rng), round to float16, crop to whole 2x2 patches
 * and apply (z - mean) / std with the config.json latent statistics in
 * weight_directory. Height and width must be at least 32. */
int h3_image_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const uint8_t *rgb_planar, int height, int width,
                        const float *noise, uint64_t seed,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_reference_latents *output,
                        char *error, size_t error_size);
void h3_reference_latents_free(h3_reference_latents *latents);

#endif
