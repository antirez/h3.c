#include "h3_vae_tiles.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SPATIAL_RATIO = H3_VAE_SPATIAL_RATIO,
    TILE_OVERLAP_MIN = H3_VAE_TILE_OVERLAP_MIN
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

void h3_vae_tile_axis_free(h3_vae_tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

int h3_vae_tile_count(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

int h3_vae_tile_axis_build(int extent, int tile_pixels,
                           h3_vae_tile_axis *axis, char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % SPATIAL_RATIO) {
        fail(error, error_size, "video VAE tile extent must be a multiple of %d",
             SPATIAL_RATIO);
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing VAE tile plan");
            return 0;
        }
        return 1;
    }
    int count = h3_vae_tile_count(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        h3_vae_tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing VAE tile plan");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

int h3_vae_stitch_tiles(float **tiles, const h3_vae_tile_axis *y_axis,
                        const h3_vae_tile_axis *x_axis, int frame_count,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h *
                   (size_t)full_w * 3;
    float *rgb = malloc(count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory stitching video VAE tiles");
        return 0;
    }
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
            int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
            int keep_h = y_axis->length -
                (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
            int keep_w = x_axis->length -
                (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
            for (int frame = 0; frame < frame_count; frame++)
                for (int y = 0; y < keep_h; y++)
                    for (int x = 0; x < keep_w; x++)
                        for (int channel = 0; channel < 3; channel++) {
                            size_t local = (((size_t)frame *
                                (size_t)y_axis->length + (size_t)y) *
                                (size_t)x_axis->length + (size_t)x) * 3 +
                                (size_t)channel;
                            float value = current[local];
                            if (above && y < overlap_y) {
                                size_t top = (((size_t)frame *
                                    (size_t)y_axis->length +
                                    (size_t)(y_axis->length - overlap_y + y)) *
                                    (size_t)x_axis->length + (size_t)x) * 3 +
                                    (size_t)channel;
                                float alpha = (float)y / (float)overlap_y;
                                value = above[top] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            if (left && x < overlap_x) {
                                size_t prior = (((size_t)frame *
                                    (size_t)y_axis->length + (size_t)y) *
                                    (size_t)x_axis->length +
                                    (size_t)(x_axis->length - overlap_x + x)) *
                                    3 + (size_t)channel;
                                float alpha = (float)x / (float)overlap_x;
                                value = left[prior] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            size_t destination = (((size_t)frame *
                                (size_t)full_h +
                                (size_t)(y_axis->starts[tile_y] + y)) *
                                (size_t)full_w +
                                (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                                (size_t)channel;
                            rgb[destination] = value;
                        }
        }
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    return 1;
}
