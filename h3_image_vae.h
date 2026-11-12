#ifndef H3_IMAGE_VAE_H
#define H3_IMAGE_VAE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int height;
    int width;
    /* Planar [3][height][width] F32. h3_image_vae_decode() returns the raw
     * decoder output (ImageNet-normalised); h3_image_planes_to_rgb() turns it
     * into RGB in [0,1]. */
    float *data;
    h3_gpu_stats gpu_stats;
} h3_image_planes;

typedef void (*h3_image_vae_progress)(int completed_tiles, int total_tiles,
                                      void *opaque);

/* Decode normalised latents [24][latent_height][latent_width] with the 2D ViT
 * decoder in 256 px tiles (64 px minimum overlap, linear blend). The weight
 * directory is the diffusers VAE directory: its safetensors file and
 * config.json (latent mean and standard deviation). */
int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent,
                        int latent_height, int latent_width,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_image_planes *output,
                        char *error, size_t error_size);
void h3_image_planes_free(h3_image_planes *planes);

/* Parse the JSON array `"key": [v0, ..., v(count-1)]` of exactly count finite
 * numbers. */
int h3_image_vae_parse_float_array(const char *json, const char *key,
                                   float *values, size_t count, char *error,
                                   size_t error_size);

/* One decoder tile: patch rows [latent_height * latent_width][768], component
 * (channel * 16 + y) * 16 + x, to interleaved [y][x][3] pixels. */
void h3_image_vae_unpack_patches(const float *rows, int latent_height,
                                 int latent_width, float *rgb);
/* In place: ImageNet de-normalisation, then clamp to [0,1]. */
void h3_image_planes_to_rgb(h3_image_planes *planes);
/* Planar bytes by truncation (x * 255, clamped), as the reference does. */
void h3_image_planes_to_u8(const h3_image_planes *planes, uint8_t *bytes);

#endif
