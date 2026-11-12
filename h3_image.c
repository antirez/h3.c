#include "h3_image.h"

#include <math.h>

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

static void rope_inv_freq(float *out, int count, double base, double span) {
    for (int index = 0; index < count; index++) {
        out[index] = (float)(1.0 / pow(base, (double)index / span));
    }
}

void h3_image_dit_inv_freq(float *out) {
    rope_inv_freq(out, H3_IMAGE_DIT_ROPE_FREQS, 10000.0,
                  H3_IMAGE_DIT_ROPE_FREQS);
}

void h3_image_vae_inv_freq(float *out) {
    rope_inv_freq(out, H3_IMAGE_VAE_ROPE_FREQS, 100.0, H3_IMAGE_VAE_ROPE_FREQS);
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
