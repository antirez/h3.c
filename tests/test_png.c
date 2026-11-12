#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_png.c: %s\n", message);
    exit(1);
}

static uint32_t big_endian32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16 |
           (uint32_t)bytes[2] << 8 | bytes[3];
}

int main(void) {
    enum { WIDTH = 37, HEIGHT = 23 };
    uint8_t rgb[WIDTH * HEIGHT * 3];
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            size_t pixel = ((size_t)y * WIDTH + (size_t)x) * 3;
            rgb[pixel] = (uint8_t)(x * 7);
            rgb[pixel + 1] = (uint8_t)(y * 11);
            rgb[pixel + 2] = (uint8_t)(x * y);
        }
    char directory[] = "/tmp/h3-png-test-XXXXXX";
    if (!mkdtemp(directory)) die("cannot create a temporary directory");
    char path[128];
    snprintf(path, sizeof(path), "%s/nested/image.png", directory);
    char error[512];
    if (!h3_ffmpeg_write_png_rgb24(path, rgb, WIDTH, HEIGHT, error,
                                   sizeof(error))) die(error);

    FILE *file = fopen(path, "rb");
    if (!file) die("PNG file was not created");
    uint8_t header[24];
    size_t got = fread(header, 1, sizeof(header), file);
    fclose(file);
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    if (got != sizeof(header) || memcmp(header, signature, 8) ||
        memcmp(header + 12, "IHDR", 4)) die("file is not a PNG");
    if (big_endian32(header + 16) != WIDTH ||
        big_endian32(header + 20) != HEIGHT)
        die("PNG header has the wrong size");

    float *decoded = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT, H3_IMAGE_FIT_STRETCH,
                                  &decoded, error, sizeof(error)))
        die(error);
    size_t plane = (size_t)WIDTH * HEIGHT;
    for (size_t pixel = 0; pixel < plane; pixel++)
        for (size_t channel = 0; channel < 3; channel++) {
            double value = (double)decoded[channel * plane + pixel] * 255.0;
            if (fabs(value - (double)rgb[pixel * 3 + channel]) > 0.01)
                die("decoded PNG differs from the written pixels");
        }
    free(decoded);

    if (h3_ffmpeg_write_png_rgb24(path, rgb, 0, HEIGHT, error, sizeof(error)))
        die("a zero width must be rejected");
    if (h3_ffmpeg_write_png_rgb24(path, NULL, WIDTH, HEIGHT, error,
                                  sizeof(error)))
        die("missing pixels must be rejected");
    unlink(path);
    snprintf(path, sizeof(path), "%s/nested", directory);
    rmdir(path);
    rmdir(directory);
    printf("ok: PNG writer round trip at %dx%d\n", WIDTH, HEIGHT);
    return 0;
}
