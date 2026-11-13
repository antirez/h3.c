#include "h3_image_vae_encode.h"

#include "h3_host.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 24,
    SPATIAL_RATIO = 16,
    PATCH = 2,
    COEFFICIENT_BITS = 22
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

typedef struct {
    int ksize;
    int *bounds;
    int *kernel;
} resample_axis;

static double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= M_PI;
    return sin(x) / x;
}

static double lanczos(double x) {
    if (-3.0 <= x && x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

static void resample_axis_free(resample_axis *axis) {
    free(axis->bounds);
    free(axis->kernel);
    memset(axis, 0, sizeof(*axis));
}

/* Pillow's precompute_coeffs and normalize_coeffs_8bpc for the whole input
 * range. */
static int resample_axis_build(int in_size, int out_size,
                               resample_axis *axis) {
    double scale = (double)in_size / out_size;
    double filter_scale = scale < 1.0 ? 1.0 : scale;
    double support = 3.0 * filter_scale;
    int ksize = (int)ceil(support) * 2 + 1;
    axis->ksize = ksize;
    axis->bounds = malloc((size_t)out_size * 2 * sizeof(*axis->bounds));
    axis->kernel = malloc((size_t)out_size * (size_t)ksize *
                          sizeof(*axis->kernel));
    double *weights = malloc((size_t)ksize * sizeof(*weights));
    if (!axis->bounds || !axis->kernel || !weights) {
        free(weights);
        resample_axis_free(axis);
        return 0;
    }
    double inverse = 1.0 / filter_scale;
    for (int out = 0; out < out_size; out++) {
        double center = (out + 0.5) * scale;
        int low = (int)(center - support + 0.5);
        if (low < 0) low = 0;
        int high = (int)(center + support + 0.5);
        if (high > in_size) high = in_size;
        high -= low;
        double total = 0.0;
        for (int tap = 0; tap < high; tap++) {
            weights[tap] = lanczos((tap + low - center + 0.5) * inverse);
            total += weights[tap];
        }
        int *row = axis->kernel + (size_t)out * (size_t)ksize;
        for (int tap = 0; tap < ksize; tap++) {
            double weight = tap >= high ? 0.0 :
                            total != 0.0 ? weights[tap] / total : weights[tap];
            double scaled = weight * (double)(1 << COEFFICIENT_BITS);
            row[tap] = (int)(weight < 0 ? -0.5 + scaled : 0.5 + scaled);
        }
        axis->bounds[out * 2] = low;
        axis->bounds[out * 2 + 1] = high;
    }
    free(weights);
    return 1;
}

static uint8_t clip_byte(int32_t accumulated) {
    int32_t value = accumulated >> COEFFICIENT_BITS;
    return (uint8_t)(value < 0 ? 0 : value > 255 ? 255 : value);
}

int h3_reference_resize_lanczos(const uint8_t *planar, int height, int width,
                                int out_height, int out_width, uint8_t *out) {
    resample_axis columns = {0}, rows = {0};
    uint8_t *middle = NULL;
    int ok = resample_axis_build(width, out_width, &columns) &&
             resample_axis_build(height, out_height, &rows);
    if (ok && width != out_width && height != out_height) {
        middle = malloc((size_t)3 * (size_t)height * (size_t)out_width);
        ok = middle != NULL;
    }
    for (int channel = 0; ok && channel < 3; channel++) {
        const uint8_t *plane = planar + (size_t)channel * height * width;
        uint8_t *target = out + (size_t)channel * out_height * out_width;
        const uint8_t *vertical_source = plane;
        int vertical_width = width;
        if (width != out_width) {
            uint8_t *destination = middle ? middle +
                (size_t)channel * height * out_width : target;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < out_width; x++) {
                    const int *kernel = columns.kernel +
                        (size_t)x * (size_t)columns.ksize;
                    const uint8_t *row = plane + (size_t)y * width +
                                         columns.bounds[x * 2];
                    int32_t sum = 1 << (COEFFICIENT_BITS - 1);
                    for (int tap = 0; tap < columns.bounds[x * 2 + 1]; tap++)
                        sum += row[tap] * kernel[tap];
                    destination[(size_t)y * out_width + x] = clip_byte(sum);
                }
            vertical_source = destination;
            vertical_width = out_width;
        }
        if (height != out_height) {
            for (int y = 0; y < out_height; y++) {
                const int *kernel = rows.kernel +
                    (size_t)y * (size_t)rows.ksize;
                for (int x = 0; x < vertical_width; x++) {
                    const uint8_t *column = vertical_source +
                        (size_t)rows.bounds[y * 2] * vertical_width + x;
                    int32_t sum = 1 << (COEFFICIENT_BITS - 1);
                    for (int tap = 0; tap < rows.bounds[y * 2 + 1]; tap++)
                        sum += column[(size_t)tap * vertical_width] *
                               kernel[tap];
                    target[(size_t)y * vertical_width + x] = clip_byte(sum);
                }
            }
        } else if (vertical_source != target) {
            memcpy(target, vertical_source,
                   (size_t)out_height * out_width);
        }
    }
    free(middle);
    resample_axis_free(&columns);
    resample_axis_free(&rows);
    return ok;
}

void h3_reference_crop_u8(const uint8_t *planar, int channels, int height,
                          int width, int crop_height, int crop_width,
                          uint8_t *out) {
    for (int channel = 0; channel < channels; channel++)
        for (int y = 0; y < crop_height; y++)
            memcpy(out + ((size_t)channel * crop_height + y) * crop_width,
                   planar + ((size_t)channel * height + y) * width,
                   (size_t)crop_width);
}

void h3_reference_crop_f32(const float *planar, int channels, int height,
                           int width, int crop_height, int crop_width,
                           float *out) {
    for (int channel = 0; channel < channels; channel++)
        for (int y = 0; y < crop_height; y++)
            memcpy(out + ((size_t)channel * crop_height + y) * crop_width,
                   planar + ((size_t)channel * height + y) * width,
                   (size_t)crop_width * sizeof(*out));
}

float h3_round_to_f16(float value) {
    return (float)(_Float16)value;
}

void h3_posterior_sample(const float *moments, const float *noise,
                         size_t count, float *sample) {
    const float *mean = moments, *logvar = moments + count;
    for (size_t index = 0; index < count; index++) {
        float clamped = logvar[index];
        clamped = clamped < -30.0f ? -30.0f : clamped > 20.0f ? 20.0f : clamped;
        float deviation = expf(0.5f * clamped);
        float scaled = deviation * noise[index];
        sample[index] = mean[index] + scaled;
    }
}

void h3_reference_normalize(float *latents, int channels, size_t plane,
                            const float *mean, const float *std) {
    for (int channel = 0; channel < channels; channel++)
        for (size_t index = 0; index < plane; index++) {
            float *value = latents + (size_t)channel * plane + index;
            *value = (*value - mean[channel]) / std[channel];
        }
}

void h3_reference_scale_noise(const float *latents, const float *noise,
                              float t, size_t count, float *out) {
    float remainder = 1.0f - t;
    for (size_t index = 0; index < count; index++) {
        float kept = t * latents[index];
        float mixed = remainder * noise[index];
        out[index] = kept + mixed;
    }
}

void h3_reference_latents_free(h3_reference_latents *latents) {
    if (!latents) return;
    free(latents->latents);
    free(latents->moments);
    memset(latents, 0, sizeof(*latents));
}

static int encode_reference(h3_image_vae_encoder *session,
                            const char *weight_directory,
                            const char *shader_source_path,
                            const uint8_t *rgb_planar, int height, int width,
                            const float *noise, uint64_t seed,
                            h3_image_vae_progress progress,
                            void *progress_opaque,
                            h3_reference_latents *output,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !shader_source_path || !rgb_planar || !output ||
        height < 2 * SPATIAL_RATIO || width < 2 * SPATIAL_RATIO) {
        fail(error, error_size, "invalid reference encoder arguments");
        return 0;
    }
    int pixel_height = height / SPATIAL_RATIO * SPATIAL_RATIO;
    int pixel_width = width / SPATIAL_RATIO * SPATIAL_RATIO;
    int moments_height = pixel_height / SPATIAL_RATIO;
    int moments_width = pixel_width / SPATIAL_RATIO;
    size_t plane = (size_t)moments_height * moments_width;
    size_t pixel_count = (size_t)3 * pixel_height * pixel_width;

    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    uint8_t *cropped = malloc(pixel_count);
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    float *drawn = noise ? NULL : malloc(LATENT_CHANNELS * plane * sizeof(*drawn));
    float *sample = malloc(LATENT_CHANNELS * plane * sizeof(*sample));
    h3_video_latent encoded = {0};
    int ok = cropped && pixels && sample && (noise || drawn);
    if (!ok) fail(error, error_size, "out of memory encoding a reference image");
    ok = ok && h3_image_vae_load_latent_normalization(
        weight_directory, mean, deviation, error, error_size);
    if (ok) {
        h3_reference_crop_u8(rgb_planar, 3, height, width, pixel_height,
                             pixel_width, cropped);
        for (size_t index = 0; index < pixel_count; index++)
            pixels[index] = (float)cropped[index] / 255.0f;
        ok = session ?
            h3_image_vae_encoder_moments(session, pixels, pixel_height,
                                         pixel_width, progress, progress_opaque,
                                         &encoded, error, error_size) :
            h3_image_vae_encode_moments(weight_directory, shader_source_path,
                                        pixels, pixel_height, pixel_width,
                                        progress, progress_opaque, &encoded,
                                        error, error_size);
    }
    if (ok && !noise) {
        h3_rng rng;
        h3_rng_seed(&rng, seed);
        h3_rng_fill_normal(&rng, drawn, LATENT_CHANNELS * plane);
        noise = drawn;
    }
    if (ok) {
        h3_posterior_sample(encoded.values, noise, LATENT_CHANNELS * plane,
                            sample);
        for (size_t index = 0; index < LATENT_CHANNELS * plane; index++)
            sample[index] = h3_round_to_f16(sample[index]);
        output->latent_height = moments_height / PATCH * PATCH;
        output->latent_width = moments_width / PATCH * PATCH;
        output->latents = malloc((size_t)LATENT_CHANNELS *
            output->latent_height * output->latent_width *
            sizeof(*output->latents));
        output->moments = encoded.values;
        encoded.values = NULL;
        output->moments_height = moments_height;
        output->moments_width = moments_width;
        output->gpu_stats = encoded.gpu_stats;
        if (!output->latents) {
            fail(error, error_size, "out of memory encoding a reference image");
            ok = 0;
        }
    }
    if (ok) {
        h3_reference_crop_f32(sample, LATENT_CHANNELS, moments_height,
                              moments_width, output->latent_height,
                              output->latent_width, output->latents);
        h3_reference_normalize(output->latents, LATENT_CHANNELS,
                               (size_t)output->latent_height *
                                   output->latent_width, mean, deviation);
    }
    free(cropped);
    free(pixels);
    free(drawn);
    free(sample);
    h3_video_latent_free(&encoded);
    if (!ok) h3_reference_latents_free(output);
    return ok;
}

int h3_image_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const uint8_t *rgb_planar, int height, int width,
                        const float *noise, uint64_t seed,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_reference_latents *output,
                        char *error, size_t error_size) {
    return encode_reference(NULL, weight_directory, shader_source_path,
                            rgb_planar, height, width, noise, seed, progress,
                            progress_opaque, output, error, error_size);
}

int h3_image_vae_encode_session(h3_image_vae_encoder *session,
                                const char *weight_directory,
                                const uint8_t *rgb_planar, int height,
                                int width, const float *noise, uint64_t seed,
                                h3_image_vae_progress progress,
                                void *progress_opaque,
                                h3_reference_latents *output,
                                char *error, size_t error_size) {
    return encode_reference(session, weight_directory, "", rgb_planar, height,
                            width, noise, seed, progress, progress_opaque,
                            output, error, error_size);
}
