#include "h3_image_edit.h"

#include "h3_dit.h"
#include "h3_image_pipeline.h"
#include "h3_image_vae_encode.h"

#include <math.h>
#include <stdio.h>
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
    if (!planar || !geometry || !vae || width < 1 || height < 1) return 0;
    uint8_t *vae_copy = resized_copy(planar, width, height, geometry->vae_w,
                                     geometry->vae_h);
    uint8_t *vision_copy = NULL;
    if (vae_copy && vision) {
        int first_w, first_h;
        uint8_t *first = NULL;
        if (h3_image_edit_bound_short_edge(width, height,
                                           H3_VISION_MIN_SHORT_EDGE,
                                           H3_VISION_MAX_SHORT_EDGE, &first_w,
                                           &first_h))
            first = resized_copy(planar, width, height, first_w, first_h);
        if (first)
            vision_copy = resized_copy(first, first_w, first_h,
                                       geometry->vision_w,
                                       geometry->vision_h);
        free(first);
    }
    if (!vae_copy || (vision && !vision_copy)) {
        free(vae_copy);
        free(vision_copy);
        return 0;
    }
    *vae = vae_copy;
    if (vision) *vision = vision_copy;
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

int h3_image_edit_bound_short_edge(int w, int h, int min_short, int max_short,
                                   int *out_w, int *out_h) {
    if (w < 1 || h < 1 || min_short < 1 || max_short < min_short || !out_w ||
        !out_h) return 0;
    int short_edge = w < h ? w : h;
    if (short_edge >= min_short && short_edge <= max_short) {
        *out_w = w;
        *out_h = h;
        return 1;
    }
    double scale = (double)(short_edge < min_short ? min_short : max_short) /
                   short_edge;
    double sides[2] = {w, h};
    int sized[2];
    for (int side = 0; side < 2; side++) {
        double units = nearbyint(sides[side] * scale /
                                 H3_REFERENCE_CANVAS_MULTIPLE);
        if (units < 1.0) units = 1.0;
        if (units * H3_REFERENCE_CANVAS_MULTIPLE > (double)INT32_MAX) return 0;
        sized[side] = (int)units * H3_REFERENCE_CANVAS_MULTIPLE;
    }
    *out_w = sized[0];
    *out_h = sized[1];
    return 1;
}

int h3_image_edit_smart_resize(int w, int h, int *out_w, int *out_h) {
    const double FACTOR = 32.0;
    const double min_pixels = 65536.0, max_pixels = 16777216.0;
    if (w < 1 || h < 1 || !out_w || !out_h) return 0;
    if ((double)(w > h ? w : h) / (double)(w < h ? w : h) > 200.0) return 0;
    double width = w, height = h;
    double h_bar = nearbyint(height / FACTOR) * FACTOR;
    double w_bar = nearbyint(width / FACTOR) * FACTOR;
    if (h_bar * w_bar > max_pixels) {
        double beta = sqrt(height * width / max_pixels);
        h_bar = floor(height / beta / FACTOR) * FACTOR;
        w_bar = floor(width / beta / FACTOR) * FACTOR;
        if (h_bar < FACTOR) h_bar = FACTOR;
        if (w_bar < FACTOR) w_bar = FACTOR;
    } else if (h_bar * w_bar < min_pixels) {
        double beta = sqrt(min_pixels / (height * width));
        h_bar = ceil(height * beta / FACTOR) * FACTOR;
        w_bar = ceil(width * beta / FACTOR) * FACTOR;
    }
    if (h_bar > INT32_MAX || w_bar > INT32_MAX) return 0;
    *out_w = (int)w_bar;
    *out_h = (int)h_bar;
    return 1;
}

int h3_image_edit_reference_geometry(int w, int h,
                                     h3_image_reference_geometry *geometry) {
    if (!geometry || w < 32 || h < 32) return 0;
    h3_image_reference_geometry g;
    int stage_w, stage_h;
    if (!h3_image_edit_bound_short_edge(w, h, H3_REFERENCE_MIN_SHORT_EDGE,
            H3_REFERENCE_MAX_SHORT_EDGE, &g.vae_w, &g.vae_h) ||
        !h3_image_edit_bound_short_edge(w, h, H3_VISION_MIN_SHORT_EDGE,
            H3_VISION_MAX_SHORT_EDGE, &stage_w, &stage_h) ||
        !h3_image_edit_smart_resize(stage_w, stage_h, &g.vision_w,
                                    &g.vision_h)) return 0;
    g.crop_w = g.vae_w / H3_IMAGE_VAE_SCALE * H3_IMAGE_VAE_SCALE;
    g.crop_h = g.vae_h / H3_IMAGE_VAE_SCALE * H3_IMAGE_VAE_SCALE;
    g.latent_w = g.crop_w / H3_IMAGE_VAE_SCALE / 2 * 2;
    g.latent_h = g.crop_h / H3_IMAGE_VAE_SCALE / 2 * 2;
    g.vision_tokens = (int)((int64_t)g.vision_w * g.vision_h / 1024);
    g.dit_rows = g.latent_w * g.latent_h / 4;
    *geometry = g;
    return 1;
}

void h3_image_edit_scan_tags(const char *prompt, size_t reference_count,
                             size_t *max_tag, uint32_t *referenced_mask) {
    static const char opening[] = "<Picture ";
    size_t largest = 0;
    uint32_t mask = 0;
    for (const char *at = prompt; at && (at = strstr(at, opening)); at++) {
        const char *digits = at + sizeof(opening) - 1;
        const char *end = digits;
        size_t value = 0;
        while (*end >= '0' && *end <= '9' && value < 1000000) {
            value = value * 10 + (size_t)(*end - '0');
            end++;
        }
        if (end == digits || *end != '>' || !value) continue;
        if (value > largest) largest = value;
        if (value <= reference_count && value <= 32)
            mask |= UINT32_C(1) << (value - 1);
    }
    if (max_tag) *max_tag = largest;
    if (referenced_mask) *referenced_mask = mask;
}

uint64_t h3_image_edit_stream(uint64_t seed, int role, size_t index) {
    return seed ^ (UINT64_C(0x9E3779B97F4A7C15) *
                   ((uint64_t)index * 2 + (uint64_t)role + 1));
}

#define EDIT_FAIL(...) do { \
        if (error && error_size) snprintf(error, error_size, __VA_ARGS__); \
        return 0; \
    } while (0)

int h3_image_edit_check_limits(const int *widths, const int *heights,
                               size_t reference_count, int canvas_latent_height,
                               int canvas_latent_width, size_t presentation_tokens,
                               h3_image_reference_geometry *geometry,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if ((reference_count && (!widths || !heights)) ||
        canvas_latent_height < 2 || canvas_latent_width < 2)
        EDIT_FAIL("invalid edit limit arguments");
    int64_t rows = (int64_t)canvas_latent_height * canvas_latent_width / 4;
    for (size_t index = 0; index < reference_count; index++) {
        h3_image_reference_geometry local;
        h3_image_reference_geometry *g = geometry ? &geometry[index] : &local;
        int w = widths[index], h = heights[index];
        if (w < 32 || h < 32)
            EDIT_FAIL("reference %zu is %dx%d pixels: both sides must be at "
                      "least 32", index + 1, w, h);
        if (!h3_image_edit_reference_geometry(w, h, g))
            EDIT_FAIL("reference %zu is %dx%d pixels: the aspect ratio is "
                      "above 200 or the size is too large", index + 1, w, h);
        if (g->latent_w > H3_IMAGE_MAX_LATENT_SIDE ||
            g->latent_h > H3_IMAGE_MAX_LATENT_SIDE)
            EDIT_FAIL("reference %zu is %dx%d pixels: its latent grid %dx%d "
                      "is above %d on a side", index + 1, w, h, g->latent_w,
                      g->latent_h, H3_IMAGE_MAX_LATENT_SIDE);
        rows += g->dit_rows;
        if (rows > H3_IMAGE_EDIT_MAX_ROWS)
            EDIT_FAIL("reference %zu brings the reference and image rows to "
                      "%lld, above %d", index + 1, (long long)rows,
                      H3_IMAGE_EDIT_MAX_ROWS);
    }
    /* 36864 is measured up to about 36.6k rows. */
    if (presentation_tokens &&
        (int64_t)presentation_tokens + rows > H3_IMAGE_EDIT_MAX_SEQUENCE)
        EDIT_FAIL("the packed sequence of %lld rows is above %d",
                  (long long)((int64_t)presentation_tokens + rows),
                  H3_IMAGE_EDIT_MAX_SEQUENCE);
    return 1;
}

#undef EDIT_FAIL

h3_layout_ref h3_image_edit_layout_ref(const h3_image_reference_geometry *geometry) {
    return (h3_layout_ref){H3_LAYOUT_REF_IMAGE, 1, geometry->latent_h,
                           geometry->latent_w, 0};
}
