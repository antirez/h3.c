#ifndef H3_IMAGE_H
#define H3_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#define H3_IMAGE_MAX_SIDE 16384
#define H3_IMAGE_VAE_SCALE 16
#define H3_IMAGE_MAX_LATENT_SIDE 1024
#define H3_IMAGE_MAX_CANVAS_AREA 33554432
#define H3_IMAGE_MIN_AREA_LOW 1024
#define H3_IMAGE_MIN_AREA_HIGH 16777216

/* Canvas (multiple of 32) an image of the requested size is generated on.
 * Returns 0 for a non-positive size or a min_area outside
 * [H3_IMAGE_MIN_AREA_LOW, H3_IMAGE_MIN_AREA_HIGH]. */
int h3_image_canvas(int height, int width, int min_area,
                    int *canvas_height, int *canvas_width);

/* NULL when a canvas can be generated, else a one-line reason: a latent side
 * above H3_IMAGE_MAX_LATENT_SIDE or an area above H3_IMAGE_MAX_CANVAS_AREA
 * (8192x4096 pixels, 32768 image tokens, the largest size measured: resident
 * BF16 needs 44 GiB there and about seven minutes; memory is not the limit,
 * the quadratic attention time is). */
const char *h3_image_canvas_error(int canvas_height, int canvas_width);

#endif
