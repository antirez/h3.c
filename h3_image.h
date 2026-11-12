#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include "h3_host.h"

#include <stddef.h>
#include <stdint.h>

#define H3_IMAGE_STEPS 6
#define H3_IMAGE_MAX_SIDE 16384
#define H3_IMAGE_VAE_SCALE 16
#define H3_IMAGE_MAX_LATENT_SIDE 1024
#define H3_IMAGE_MAX_CANVAS_AREA 33554432
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
 * above H3_IMAGE_MAX_LATENT_SIDE or an area above H3_IMAGE_MAX_CANVAS_AREA
 * (8192x4096 pixels, 32768 image tokens, the largest size measured: resident
 * BF16 needs 44 GiB there and about seven minutes; memory is not the limit,
 * the quadratic attention time is). */
const char *h3_image_canvas_error(int canvas_height, int canvas_width);

#define H3_REFERENCE_CANVAS_MULTIPLE 32
#define H3_REFERENCE_MIN_SHORT_EDGE 1024
#define H3_REFERENCE_MAX_SHORT_EDGE 2048
#define H3_VISION_MIN_SHORT_EDGE 1024
#define H3_VISION_MAX_SHORT_EDGE 1024

/* Reference sizing of an edit (MiniMaxH3ImageEditResizeStep): a short edge in
 * [min_short, max_short] keeps the size, else the image is scaled to put the
 * short edge on the nearer bound and each side rounds half-to-even to a
 * multiple of 32 (at least 32). Returns 0 for a non-positive argument or a
 * result beyond INT32_MAX. */
int h3_image_edit_bound_short_edge(int w, int h, int min_short, int max_short,
                                   int *out_w, int *out_h);
/* Qwen smart_resize of the vision copy: factor 32, min_pixels 65536,
 * max_pixels 16777216. Returns 0 for an aspect ratio above 200. */
int h3_image_edit_smart_resize(int w, int h, int *out_w, int *out_h);

typedef struct {
    int vae_w, vae_h;       /* VAE copy: bound_short_edge(1024, 2048) */
    int crop_w, crop_h;     /* top-left crop to multiples of 16 */
    int latent_w, latent_h; /* crop / 16, cut to even sides */
    int vision_w, vision_h; /* bound_short_edge(1024, 1024), then smart_resize */
    int vision_tokens;      /* vision_w * vision_h / 1024 */
    int dit_rows;           /* latent_w * latent_h / 4 */
} h3_image_reference_geometry;
/* Returns 0 when a side is below 32 or the vision aspect ratio is above 200. */
int h3_image_edit_reference_geometry(int w, int h,
                                     h3_image_reference_geometry *geometry);

#define H3_IMAGE_EDIT_MAX_ROWS 32768
#define H3_IMAGE_EDIT_MAX_SEQUENCE 36864

/* Scan a prompt for <Picture N> tags (N a plain decimal, no spaces, at least
 * 1). Sets *max_tag to the largest N (0 without tags) and bit N-1 of
 * *referenced_mask for every N in [1, min(reference_count, 32)]. Either
 * output may be NULL. */
void h3_image_edit_scan_tags(const char *prompt, size_t reference_count,
                             size_t *max_tag, uint32_t *referenced_mask);

/* Seed of the noise stream of reference `index` (0-based): role 1 draws its
 * posterior sample, role 2 its augmentation noise. Injective in (role, index)
 * for a given seed. */
uint64_t h3_image_edit_stream(uint64_t seed, int role, size_t index);

/* Everything an edit can reject before a weight loads. widths/heights are the
 * pixel sizes of the references after EXIF orientation; canvas_latent_* is
 * the generated image's latent grid; presentation_tokens is the full text
 * presentation length N: the "Picture i: " label tokens, plus vision_tokens +
 * 2 for each reference, plus the prompt tokens; 0 skips the sequence check.
 * For each reference i (1-based in the message):
 * both sides at least 32 and a vision aspect ratio of at most 200, a latent
 * side of at most H3_IMAGE_MAX_LATENT_SIDE, and a running R + P of at most
 * H3_IMAGE_EDIT_MAX_ROWS; then N + R + P of at most
 * H3_IMAGE_EDIT_MAX_SEQUENCE. geometry (reference_count entries) is filled
 * for the references that pass and may be NULL. Returns 0 with a one-line
 * message on the first failure. */
int h3_image_edit_check_limits(const int *widths, const int *heights,
                               size_t reference_count, int canvas_latent_height,
                               int canvas_latent_width, size_t presentation_tokens,
                               h3_image_reference_geometry *geometry,
                               char *error, size_t error_size);

/* The layout reference of an edit reference: latent_h and latent_w map to the
 * h and w fields, one latent frame. */
h3_layout_ref h3_image_edit_layout_ref(const h3_image_reference_geometry *geometry);

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
