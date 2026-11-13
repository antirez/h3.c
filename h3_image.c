#include "h3_image.h"

#include "h3_host.h"

#include <math.h>

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

