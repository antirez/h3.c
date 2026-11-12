#include "h3_image.h"

#include <math.h>
#include <stdint.h>

int h3_image_schedule_build(h3_sigma_schedule *schedule) {
    static const double sample_sigmas[H3_IMAGE_STEPS + 1] = {
        1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
        0.5327739119529724, 0.23200887441635132, 0.0
    };
    if (!schedule) return 0;
    schedule->steps = H3_IMAGE_STEPS;
    for (int index = 0; index <= H3_IMAGE_STEPS; index++) {
        float timestep = (float)(1.0 - sample_sigmas[index]);
        float sigma = 1.0f - timestep;
        schedule->video[index] = sigma;
        schedule->audio[index] = sigma;
    }
    return 1;
}

int h3_image_canvas(int height, int width, int min_area,
                    int *canvas_height, int *canvas_width) {
    if (height < 1 || width < 1 || !canvas_height || !canvas_width ||
        min_area < H3_IMAGE_MIN_AREA_LOW || min_area > H3_IMAGE_MIN_AREA_HIGH) {
        return 0;
    }
    double scale = fmax(1.0, sqrt((double)min_area /
                                  ((double)width * (double)height)));
    double multiple = H3_CANVAS_MULTIPLE;
    double scaled_height = nearbyint((double)height * scale / multiple) * multiple;
    double scaled_width = nearbyint((double)width * scale / multiple) * multiple;
    if (scaled_height > 1e9 || scaled_width > 1e9) return 0;
    *canvas_height = (int)fmax(multiple, scaled_height);
    *canvas_width = (int)fmax(multiple, scaled_width);
    return 1;
}

static const char *parse_dimension(const char *text, long *value) {
    if (*text < '0' || *text > '9') return NULL;
    long parsed = 0;
    while (*text >= '0' && *text <= '9') {
        parsed = parsed * 10 + (*text - '0');
        if (parsed > INT32_MAX) return NULL;
        text++;
    }
    *value = parsed;
    return text;
}

int h3_image_parse_size(const char *text, int *width, int *height) {
    long parsed_width, parsed_height;
    if (!text || !width || !height) return 0;
    const char *middle = parse_dimension(text, &parsed_width);
    if (!middle || (*middle != 'x' && *middle != 'X')) return 0;
    const char *end = parse_dimension(middle + 1, &parsed_height);
    if (!end || *end || parsed_width < 1 || parsed_height < 1) return 0;
    *width = (int)parsed_width;
    *height = (int)parsed_height;
    return 1;
}

const char *h3_image_canvas_error(int canvas_height, int canvas_width) {
    if (canvas_height / H3_IMAGE_VAE_SCALE > H3_IMAGE_MAX_LATENT_SIDE ||
        canvas_width / H3_IMAGE_VAE_SCALE > H3_IMAGE_MAX_LATENT_SIDE)
        return "the generation canvas is more than 16384 pixels on a side";
    if ((long long)canvas_height * canvas_width > H3_IMAGE_MAX_CANVAS_AREA)
        return "the generation canvas is larger than 33554432 pixels (32768 "
               "image tokens); runs were measured up to 8192x4096, and the "
               "cap is the same in every weight mode by policy (--int8 is "
               "faster per step but does not raise it), so ask for a "
               "smaller image";
    return NULL;
}

static void rope_inv_freq(float *out, int count, int index_step, float base,
                          int span) {
    for (int index = 0; index < count; index++) {
        out[index] = 1.0f / powf(base, (float)(index * index_step) / (float)span);
    }
}

void h3_image_dit_inv_freq(float *out) {
    rope_inv_freq(out, H3_IMAGE_DIT_ROPE_FREQS, 2, 10000.0f, 32);
}

void h3_image_vae_inv_freq(float *out) {
    rope_inv_freq(out, H3_IMAGE_VAE_ROPE_FREQS, 1, 100.0f, 8);
}

int h3_image_qkv_source(size_t dest_row, unsigned heads, unsigned head_dim,
                        unsigned *projection, size_t *source_row) {
    if (!heads || !head_dim || !projection || !source_row ||
        dest_row >= (size_t)3 * heads * head_dim) {
        return 0;
    }
    size_t slab = dest_row / head_dim;
    *projection = (unsigned)(slab % 3);
    *source_row = (slab / 3) * head_dim + dest_row % head_dim;
    return 1;
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
