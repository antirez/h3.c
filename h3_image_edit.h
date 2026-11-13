#ifndef H3_IMAGE_EDIT_H
#define H3_IMAGE_EDIT_H

#include "h3_host.h"
#include "h3_image.h"

#include <stddef.h>
#include <stdint.h>

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

/* The two resized copies of an upright reference, planar bytes [3][h][w], each
 * malloc'd (the caller frees them): the VAE copy (geometry vae_w x vae_h) and
 * the Qwen vision copy (vision_w x vision_h); a NULL vision makes only the VAE
 * copy. Both resize with the PIL Lanczos port; the vision copy's second stage
 * (smart_resize) is bicubic in the reference and Lanczos here. A size that
 * does not change is copied. */
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
