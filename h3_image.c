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
        return "the generation canvas is larger than 33554432 pixels (8192x4096, "
               "32768 image tokens), the largest size that was measured; "
               "the limit is time, not memory (attention time grows with "
               "the square of the token count), and --int8 and "
               "--ssd-streaming do not raise it, so ask for a smaller image";
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
