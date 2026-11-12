#include "h3_image_edit.h"

#include "h3_dit.h"
#include "h3_image_pipeline.h"
#include "h3_image_vae_encode.h"

#include <stdlib.h>
#include <string.h>

static uint8_t *resized_copy(const uint8_t *planar, int width, int height,
                             int out_width, int out_height) {
    size_t bytes = (size_t)3 * (size_t)out_width * (size_t)out_height;
    uint8_t *out = malloc(bytes);
    if (!out) return NULL;
    if (out_width == width && out_height == height)
        memcpy(out, planar, bytes);
    else if (!h3_reference_resize_lanczos(planar, height, width, out_height,
                                          out_width, out)) {
        free(out);
        return NULL;
    }
    return out;
}

int h3_image_edit_copies(const uint8_t *planar, int width, int height,
                         const h3_image_reference_geometry *geometry,
                         uint8_t **vae, uint8_t **vision) {
    if (vae) *vae = NULL;
    if (vision) *vision = NULL;
    if (!planar || !geometry || !vae || !vision || width < 1 || height < 1)
        return 0;
    int first_w, first_h;
    if (!h3_image_edit_bound_short_edge(width, height,
                                        H3_VISION_MIN_SHORT_EDGE,
                                        H3_VISION_MAX_SHORT_EDGE, &first_w,
                                        &first_h)) return 0;
    uint8_t *vae_copy = resized_copy(planar, width, height, geometry->vae_w,
                                     geometry->vae_h);
    uint8_t *first = resized_copy(planar, width, height, first_w, first_h);
    uint8_t *vision_copy = first ?
        resized_copy(first, first_w, first_h, geometry->vision_w,
                     geometry->vision_h) : NULL;
    free(first);
    if (!vae_copy || !vision_copy) {
        free(vae_copy);
        free(vision_copy);
        return 0;
    }
    *vae = vae_copy;
    *vision = vision_copy;
    return 1;
}

int h3_image_edit_condition_rows(const float *latents, const float *aug_noise,
                                 int latent_h, int latent_w, float *rows) {
    if (!latents || !aug_noise || !rows || latent_h < 2 || latent_w < 2 ||
        latent_h % 2 || latent_w % 2) return 0;
    size_t count = (size_t)H3_IMAGE_LATENT_CHANNELS * (size_t)latent_h * (size_t)latent_w;
    float *noised = malloc(count * sizeof(*noised));
    if (!noised) return 0;
    h3_reference_scale_noise(latents, aug_noise, H3_REFERENCE_NOISE_AUG, count,
                             noised);
    int ok = h3_dit_patchify_video(noised, H3_IMAGE_LATENT_CHANNELS, 1, latent_h, latent_w, rows,
                                   count);
    free(noised);
    return ok;
}
