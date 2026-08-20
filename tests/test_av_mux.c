#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_av_mux.c: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    enum { WIDTH = 32, HEIGHT = 32, FRAMES = 8, SAMPLES = 64000 };
    const char *path = argc > 1 ? argv[1] : "/tmp/h3-av-mux-test.mp4";
    uint8_t *rgb = malloc(WIDTH * HEIGHT * 3 * FRAMES);
    float *pcm = malloc(2 * SAMPLES * sizeof(*pcm));
    if (!rgb || !pcm) die("out of memory creating mux fixture");
    for (int frame = 0; frame < FRAMES; frame++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t pixel = ((size_t)frame * HEIGHT * WIDTH +
                                (size_t)y * WIDTH + (size_t)x) * 3;
                rgb[pixel] = (uint8_t)(x * 8);
                rgb[pixel + 1] = (uint8_t)(y * 8);
                rgb[pixel + 2] = (uint8_t)(frame * 32);
            }
    for (int channel = 0; channel < 2; channel++)
        for (int sample = 0; sample < SAMPLES; sample++)
            pcm[(size_t)channel * SAMPLES + (size_t)sample] =
                1.5f * sinf(2.0f * 3.14159265358979323846f *
                             (float)(220 + channel * 110) * (float)sample /
                             32000.0f);
    char error[512];
    if (!h3_ffmpeg_write_av_rgb24_f32(path, rgb, FRAMES, WIDTH, HEIGHT, 24,
                                      pcm, SAMPLES, 2, 32000,
                                      error, sizeof(error))) die(error);
    struct stat status;
    if (stat(path, &status) != 0 || status.st_size < 1000)
        die("FFmpeg did not create a nonempty A/V container");
    int probed_width = 0, probed_height = 0;
    if (!h3_ffprobe_visual_size(path, &probed_width, &probed_height,
                                error, sizeof(error))) die(error);
    if (probed_width != WIDTH || probed_height != HEIGHT)
        die("FFprobe returned unexpected visual dimensions");
    float *decoded = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT,
                                  H3_IMAGE_FIT_STRETCH, &decoded,
                                  error, sizeof(error))) die(error);
    double red = 0.0, green = 0.0, blue = 0.0;
    for (size_t pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
        if (!isfinite(decoded[pixel]) || decoded[pixel] < 0.0f ||
            decoded[pixel] > 1.0f ||
            !isfinite(decoded[WIDTH * HEIGHT + pixel]) ||
            !isfinite(decoded[2 * WIDTH * HEIGHT + pixel]))
            die("decoded FFmpeg image contains invalid pixels");
        red += decoded[pixel];
        green += decoded[WIDTH * HEIGHT + pixel];
        blue += decoded[2 * WIDTH * HEIGHT + pixel];
    }
    if (red < 100.0 || green < 100.0 || blue > 80.0)
        die("decoded FFmpeg image has unexpected channel-major content");
    free(decoded);
    decoded = NULL;
    if (!h3_ffmpeg_read_image_f32(path, WIDTH, HEIGHT * 2,
                                  H3_IMAGE_FIT_COVER, &decoded,
                                  error, sizeof(error))) die(error);
    free(decoded);
    decoded = NULL;
    int decoded_frames = 0;
    if (!h3_ffmpeg_read_video_f32(path, WIDTH, HEIGHT, FRAMES,
                                  &decoded, &decoded_frames,
                                  error, sizeof(error))) die(error);
    if (decoded_frames != 5)
        die("FFmpeg video input did not align to 5+17k frames");
    size_t decoded_values = (size_t)3 * decoded_frames * WIDTH * HEIGHT;
    for (size_t index = 0; index < decoded_values; index++) {
        if (!isfinite(decoded[index]) || decoded[index] < 0.0f ||
            decoded[index] > 1.0f)
            die("decoded FFmpeg video contains invalid pixels");
    }
    free(decoded);
    float *decoded_pcm = NULL;
    int decoded_samples = 0;
    if (!h3_ffmpeg_read_audio_f32(path, SAMPLES, 1,
                                  &decoded_pcm, &decoded_samples,
                                  error, sizeof(error))) die(error);
    if (decoded_samples != SAMPLES)
        die("FFmpeg audio input returned an unexpected sample count");
    double left_energy = 0.0, right_energy = 0.0;
    float peak = 0.0f;
    for (int sample = 0; sample < decoded_samples; sample++) {
        float left = decoded_pcm[sample];
        float right = decoded_pcm[decoded_samples + sample];
        if (!isfinite(left) || !isfinite(right))
            die("decoded FFmpeg audio contains non-finite PCM");
        left_energy += (double)left * left;
        right_energy += (double)right * right;
        peak = fmaxf(peak, fmaxf(fabsf(left), fabsf(right)));
    }
    if (left_energy < 1.0 || right_energy < 1.0)
        die("decoded FFmpeg audio has no stereo signal");
    if (peak > 1.0f)
        die("decoded FFmpeg audio exceeds the anti-clipping ceiling");
    free(decoded_pcm);
    printf("ok: concurrent FFmpeg video/PCM pipes created %s (%lld bytes)\n",
           path, (long long)status.st_size);
    free(rgb);
    free(pcm);
    return 0;
}
