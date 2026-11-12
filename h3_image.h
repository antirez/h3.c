#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include "h3_host.h"

#include <stddef.h>

#define H3_IMAGE_STEPS 6
#define H3_IMAGE_MIN_AREA_LOW 1024
#define H3_IMAGE_MIN_AREA_HIGH 16777216
#define H3_IMAGE_DIT_ROPE_FREQS 16
#define H3_IMAGE_VAE_ROPE_FREQS 8

/* The distilled six-step grid, with sigma rounded through float32 timesteps
 * exactly as the reference scheduler does. video[] and audio[] are equal. */
int h3_image_schedule_build(h3_sigma_schedule *schedule);

/* Canvas (multiple of 32) an image of the requested size is generated on.
 * Returns 0 for a non-positive size or a min_area outside
 * [H3_IMAGE_MIN_AREA_LOW, H3_IMAGE_MIN_AREA_HIGH]. */
int h3_image_canvas(int height, int width, int min_area,
                    int *canvas_height, int *canvas_width);

/* DiT rotary frequencies, used when the checkpoint has no rope.inv_freq. */
void h3_image_dit_inv_freq(float *out);
/* Image VAE decoder rotary frequencies. */
void h3_image_vae_inv_freq(float *out);

/* Row of the interleaved [head][q,k,v][head_dim] QKV matrix: which projection
 * (0 = q, 1 = k, 2 = v) and which row of that projection's matrix it comes
 * from. Returns 0 when dest_row is outside 3 * heads * head_dim. */
int h3_image_qkv_source(size_t dest_row, unsigned heads, unsigned head_dim,
                        unsigned *projection, size_t *source_row);

#endif
