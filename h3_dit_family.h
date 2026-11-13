#ifndef H3_DIT_FAMILY_H
#define H3_DIT_FAMILY_H

#include "h3_dit_names.h"
#include "h3_host.h"

#include <stddef.h>

/* Model constants of the image family (see h3_dit_names.h for its tensor
 * names): the DiT layer owns them, the image pipeline reads them. */
#define H3_DIT_IMAGE_STEPS 6
#define H3_DIT_IMAGE_ROPE_FREQS 16

/* The distilled six-step grid, with sigma rounded through float32 timesteps
 * exactly as the reference scheduler does. video[] and audio[] are equal. */
int h3_dit_image_schedule_build(h3_sigma_schedule *schedule);

/* Whether the first half of the image checkpoint's FC1 (ff.net.0.proj) rows is
 * the SwiGLU gate, the order the fused kernels expect. 0 means the loader swaps
 * the halves. */
#define H3_DIT_IMAGE_FC1_GATE_FIRST 0

/* DiT rotary frequencies, used when the checkpoint has no rope.inv_freq. */
void h3_dit_image_inv_freq(float *out);

/* Row of the interleaved [head][q,k,v][head_dim] QKV matrix: which projection
 * (0 = q, 1 = k, 2 = v) and which row of that projection's matrix it comes
 * from. Returns 0 when dest_row is outside 3 * heads * head_dim. */
int h3_dit_qkv_source(size_t dest_row, unsigned heads, unsigned head_dim,
                      unsigned *projection, size_t *source_row);

#endif
