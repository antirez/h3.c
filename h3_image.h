#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include "h3_host.h"

#include <stddef.h>

#define H3_IMAGE_STEPS 6
#define H3_IMAGE_MAX_SIDE 16384
#define H3_IMAGE_VAE_SCALE 16
#define H3_IMAGE_MAX_LATENT_SIDE 1024
#define H3_IMAGE_MAX_CANVAS_AREA 16777216
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

/* Parse "WxH" (or "WXH"): two plain decimal integers in [1, INT32_MAX], no
 * sign, no whitespace, nothing after. */
int h3_image_parse_size(const char *text, int *width, int *height);

/* NULL when a canvas can be generated, else a one-line reason: a latent side
 * above H3_IMAGE_MAX_LATENT_SIDE or an area above H3_IMAGE_MAX_CANVAS_AREA. */
const char *h3_image_canvas_error(int canvas_height, int canvas_width);

/* Whether the first half of the image checkpoint's FC1 (ff.net.0.proj) rows is
 * the SwiGLU gate, the order the fused kernels expect. 0 means the loader swaps
 * the halves. */
#define H3_IMAGE_FC1_GATE_FIRST 0

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
