#include "h3_ffmpeg.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_image_png.c: %s\n", message);
    exit(1);
}

static uint8_t *read_all(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) die("cannot open a PNG");
    long end = ftell(file);
    if (end < 1 || fseek(file, 0, SEEK_SET)) die("empty PNG file");
    uint8_t *data = malloc((size_t)end);
    if (!data || fread(data, 1, (size_t)end, file) != (size_t)end)
        die("cannot read a PNG");
    fclose(file);
    *size = (size_t)end;
    return data;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s WIDTH HEIGHT PNG [SECOND_PNG]\n", argv[0]);
        return 2;
    }
    int width = atoi(argv[1]), height = atoi(argv[2]);
    if (width < 1 || height < 1) die("bad size arguments");
    char error[512];
    int probed_width = 0, probed_height = 0;
    if (!h3_ffprobe_visual_size(argv[3], &probed_width, &probed_height, error,
                                sizeof(error))) die(error);
    if (probed_width != width || probed_height != height)
        die("the PNG has the wrong size");
    float *pixels = NULL;
    if (!h3_ffmpeg_read_image_f32(argv[3], width, height, H3_IMAGE_FIT_STRETCH,
                                  &pixels, error, sizeof(error))) die(error);
    size_t count = (size_t)width * height * 3;
    int constant = 1;
    for (size_t index = 1; index < count; index++)
        if (pixels[index] != pixels[0]) {
            constant = 0;
            break;
        }
    free(pixels);
    if (constant) die("the image is constant");
    printf("%s: %dx%d RGB, not constant\n", argv[3], width, height);
    if (argc > 4) {
        size_t first_size, second_size;
        uint8_t *first = read_all(argv[3], &first_size);
        uint8_t *second = read_all(argv[4], &second_size);
        if (first_size != second_size || memcmp(first, second, first_size))
            die("the two runs are not byte-identical");
        printf("%s and %s are byte-identical (%zu bytes)\n", argv[3], argv[4],
               first_size);
        free(first);
        free(second);
    }
    printf("ok: image PNG check\n");
    return 0;
}
