#include "h3_dit_family.h"

#include <math.h>

int h3_dit_image_schedule_build(h3_sigma_schedule *schedule) {
    static const double sample_sigmas[H3_DIT_IMAGE_STEPS + 1] = {
        1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
        0.5327739119529724, 0.23200887441635132, 0.0
    };
    if (!schedule) return 0;
    schedule->steps = H3_DIT_IMAGE_STEPS;
    for (int index = 0; index <= H3_DIT_IMAGE_STEPS; index++) {
        float timestep = (float)(1.0 - sample_sigmas[index]);
        float sigma = 1.0f - timestep;
        schedule->video[index] = sigma;
        schedule->audio[index] = sigma;
    }
    return 1;
}

void h3_dit_image_inv_freq(float *out) {
    for (int index = 0; index < H3_DIT_IMAGE_ROPE_FREQS; index++) {
        out[index] = 1.0f / powf(10000.0f, (float)(index * 2) / 32.0f);
    }
}

int h3_dit_qkv_source(size_t dest_row, unsigned heads, unsigned head_dim,
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

