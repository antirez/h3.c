#include "h3_ffmpeg.h"

#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_png.c: %s\n", message);
    exit(1);
}

static uint32_t big_endian32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16 |
           (uint32_t)bytes[2] << 8 | bytes[3];
}

static const uint8_t PNG_SIGNATURE[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};

static void check_hostile_names(const uint8_t *rgb, int width, int height) {
    static const char *const names[] = {
        "-leading.png", "with space.png", "quo'te\"d.png",
        "$(touch pwned).png", "pipe:1", "back`tick`.png", "sub/relative.png"
    };
    enum { NAME_COUNT = sizeof(names) / sizeof(names[0]) };
    char error[512];
    for (size_t index = 0; index < NAME_COUNT; index++) {
        if (!h3_ffmpeg_write_png_rgb24(names[index], rgb, width, height,
                                       error, sizeof(error))) die(error);
        FILE *named = fopen(names[index], "rb");
        if (!named) die("hostile PNG name did not land at its literal path");
        uint8_t magic[8];
        size_t got = fread(magic, 1, sizeof(magic), named);
        fclose(named);
        if (got != sizeof(magic) || memcmp(magic, PNG_SIGNATURE, 8))
            die("hostile PNG name holds no PNG");
    }
    uint8_t frames[2 * 16 * 16 * 3] = {0};
    if (!h3_ffmpeg_write_rgb24("-video.mp4", frames, 2, 16, 16, 24, error,
                               sizeof(error))) die(error);
    struct stat status;
    if (stat("-video.mp4", &status) != 0 || status.st_size < 100)
        die("hostile MP4 name did not land at its literal path");
    size_t entries = 0;
    DIR *listing = opendir(".");
    if (!listing) die("cannot list the temporary directory");
    for (struct dirent *entry; (entry = readdir(listing)) != NULL;)
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, ".."))
            entries++;
    closedir(listing);
    /* Six PNG files, the directory sub, -video.mp4, and the directory nested
     * of the first test. */
    if (entries != 9) die("a hostile name created or removed other files");
    for (size_t index = 0; index < NAME_COUNT; index++) unlink(names[index]);
    unlink("-video.mp4");
    rmdir("sub");
}

static void check_failed_encode_removes_output(const uint8_t *rgb, int width,
                                               int height) {
    char error[512];
    FILE *partial = fopen("partial.png", "wb");
    if (!partial) die("cannot create a partial file");
    fputs("partial", partial);
    fclose(partial);
    setenv("H3_FFMPEG", "/usr/bin/false", 1);
    if (h3_ffmpeg_write_png_rgb24("partial.png", rgb, width, height, error,
                                  sizeof(error)))
        die("a failing encoder must be reported");
    unsetenv("H3_FFMPEG");
    if (access("partial.png", F_OK) == 0)
        die("a failed encode left its output file behind");
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
    if (got != sizeof(header) || memcmp(header, PNG_SIGNATURE, 8) ||
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
    if (access(path, F_OK) != 0)
        die("a rejected request must not remove an existing file");

    if (chdir(directory) != 0) die("cannot enter the temporary directory");
    check_hostile_names(rgb, WIDTH, HEIGHT);
    check_failed_encode_removes_output(rgb, WIDTH, HEIGHT);
    if (chdir("/") != 0) die("cannot leave the temporary directory");

    unlink(path);
    snprintf(path, sizeof(path), "%s/nested", directory);
    rmdir(path);
    rmdir(directory);
    printf("ok: PNG writer round trip at %dx%d, hostile names, failed encode\n",
           WIDTH, HEIGHT);
    return 0;
}
