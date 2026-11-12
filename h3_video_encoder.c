#include "h3_video_encoder.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RGB_CHANNELS = 3,
    LATENT_CHANNELS = 24,
    MOMENT_CHANNELS = 48,
    LEVELS = 6,
    BLOCKS = 2,
    GROUPS = 32,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

static const uint32_t level_channels[LEVELS] = {128, 256, 256, 512, 512, 1024};
static const uint32_t space_strides[LEVELS] = {2, 2, 2, 2, 1, 1};
static const uint32_t time_strides[LEVELS] = {1, 2, 2, 1, 1, 1};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel;
    uint32_t stride_t;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t depth_front;
    uint32_t height_before;
    uint32_t height_after;
    uint32_t width_before;
    uint32_t width_after;
} encoder_conv;

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} encoder_norm;

typedef struct {
    encoder_norm norm1;
    encoder_conv conv1;
    encoder_norm norm2;
    encoder_conv conv2;
    encoder_conv shortcut;
    int has_shortcut;
} encoder_block;

typedef struct {
    encoder_block blocks[BLOCKS];
    encoder_conv downsample;
    int has_downsample;
} encoder_level;

typedef struct {
    const char *block;
    const char *shortcut;
    const char *downsample;
} encoder_names;

static const encoder_names video_names = {
    "encoder.down.%d.block.%d", "nin_shortcut",
    "encoder.down.%d.downsample.conv"
};
static const encoder_names image_names = {
    "encoder.down_blocks.%d.resnets.%d", "conv_shortcut",
    "encoder.down_blocks.%d.downsamplers.0.conv"
};

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *store;
    const encoder_names *names;
    /* Video: the mean channels, normalized per tile. Image: all quant_conv
     * moments, as they are. */
    int raw_moments;
    int output_channels;
    encoder_conv conv_in;
    encoder_level levels[LEVELS];
    encoder_norm norm_out;
    encoder_conv conv_out;
    encoder_conv quant;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
} encoder_context;

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(encoder_context *encoder, int ok, char *error,
                  size_t error_size, const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(encoder->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static void free_conv(encoder_conv *conv) {
    free_tensor(&conv->weight);
    free_tensor(&conv->bias);
}

static void free_norm(encoder_norm *norm) {
    free_tensor(&norm->weight);
    free_tensor(&norm->bias);
}

static void cleanup(encoder_context *encoder) {
    if (!encoder) return;
    free_conv(&encoder->conv_in);
    for (int level = 0; level < LEVELS; level++) {
        for (int block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            free_norm(&item->norm1);
            free_conv(&item->conv1);
            free_norm(&item->norm2);
            free_conv(&item->conv2);
            free_conv(&item->shortcut);
        }
        free_conv(&encoder->levels[level].downsample);
    }
    free_norm(&encoder->norm_out);
    free_conv(&encoder->conv_out);
    free_conv(&encoder->quant);
    h3_weight_store_free(encoder->store);
    h3_gpu_free(encoder->gpu);
    memset(encoder, 0, sizeof(*encoder));
}

static h3_gpu_tensor *load_f32(encoder_context *encoder, const char *name,
                               int ndim, const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(encoder->store, encoder->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(encoder_context *encoder, const char *name,
                         uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(encoder, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f5(encoder_context *encoder, const char *name,
                         uint64_t output_channels, uint64_t input_channels,
                         uint64_t kernel, char *error, size_t error_size) {
    uint64_t shape[] = {
        output_channels, input_channels, kernel, kernel, kernel
    };
    return load_f32(encoder, name, 5, shape, error, error_size);
}

static int load_conv(encoder_context *encoder, encoder_conv *conv,
                     const char *prefix, uint32_t input_channels,
                     uint32_t output_channels, uint32_t kernel,
                     uint32_t stride_t, uint32_t stride_h,
                     uint32_t stride_w, uint32_t depth_front,
                     uint32_t height_before, uint32_t height_after,
                     uint32_t width_before, uint32_t width_after,
                     char *error, size_t error_size) {
    char name[192];
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    conv->stride_t = stride_t;
    conv->stride_h = stride_h;
    conv->stride_w = stride_w;
    conv->depth_front = depth_front;
    conv->height_before = height_before;
    conv->height_after = height_after;
    conv->width_before = width_before;
    conv->width_after = width_after;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    conv->weight = f5(encoder, name, output_channels, input_channels, kernel,
                      error, error_size);
    if (!conv->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    conv->bias = f1(encoder, name, output_channels, error, error_size);
    return conv->bias != NULL;
}

static int load_norm(encoder_context *encoder, encoder_norm *norm,
                     const char *prefix, uint32_t channels,
                     char *error, size_t error_size) {
    char name[192];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    norm->weight = f1(encoder, name, channels, error, error_size);
    if (!norm->weight) return 0;
    snprintf(name, sizeof(name), "%s.bias", prefix);
    norm->bias = f1(encoder, name, channels, error, error_size);
    return norm->bias != NULL;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_normalization(encoder_context *encoder,
                              const char *weight_directory,
                              char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", encoder->latent_mean,
                               LATENT_CHANNELS, error, error_size) &&
             parse_float_array(json, "latents_std", encoder->latent_std,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        if (encoder->latent_std[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent deviation is invalid");
            return 0;
        }
    return ok;
}

static int load_weights(encoder_context *encoder, char *error,
                        size_t error_size) {
    if (!load_conv(encoder, &encoder->conv_in, "encoder.conv_in", 3, 128, 3,
                   1, 1, 1, 2, 1, 1, 1, 1, error, error_size)) return 0;
    uint32_t previous = 128;
    for (int level = 0; level < LEVELS; level++) {
        uint32_t channels = level_channels[level];
        for (int block = 0; block < BLOCKS; block++) {
            encoder_block *item = &encoder->levels[level].blocks[block];
            uint32_t input_channels = block ? channels : previous;
            char prefix[192];
            snprintf(prefix, sizeof(prefix), encoder->names->block, level,
                     block);
            char name[224];
            snprintf(name, sizeof(name), "%s.norm1", prefix);
            if (!load_norm(encoder, &item->norm1, name, input_channels,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.conv1", prefix);
            if (!load_conv(encoder, &item->conv1, name, input_channels,
                           channels, 3, 1, 1, 1, 2, 1, 1, 1, 1,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.norm2", prefix);
            if (!load_norm(encoder, &item->norm2, name, channels,
                           error, error_size)) return 0;
            snprintf(name, sizeof(name), "%s.conv2", prefix);
            if (!load_conv(encoder, &item->conv2, name, channels, channels, 3,
                           1, 1, 1, 2, 1, 1, 1, 1,
                           error, error_size)) return 0;
            if (input_channels != channels) {
                snprintf(name, sizeof(name), "%s.%s", prefix,
                         encoder->names->shortcut);
                if (!load_conv(encoder, &item->shortcut, name, input_channels,
                               channels, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                               error, error_size)) return 0;
                item->has_shortcut = 1;
            }
        }
        if (space_strides[level] * time_strides[level] > 1) {
            char name[192];
            snprintf(name, sizeof(name), encoder->names->downsample, level);
            uint32_t spatial_tail = space_strides[level] == 2 ? 1 : 0;
            if (!load_conv(encoder, &encoder->levels[level].downsample, name,
                           channels, channels, 3, time_strides[level],
                           space_strides[level], space_strides[level], 2, 0,
                           spatial_tail, 0, spatial_tail, error,
                           error_size)) return 0;
            encoder->levels[level].has_downsample = 1;
        }
        previous = channels;
    }
    if (!load_norm(encoder, &encoder->norm_out, "encoder.norm_out", 1024,
                   error, error_size) ||
        !load_conv(encoder, &encoder->conv_out, "encoder.conv_out", 1024,
                   MOMENT_CHANNELS, 3, 1, 1, 1, 2, 1, 1, 1, 1,
                   error, error_size) ||
        !load_conv(encoder, &encoder->quant, "quant_conv", MOMENT_CHANNELS,
                   MOMENT_CHANNELS, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                   error, error_size)) return 0;
    return 1;
}

static size_t tensor_elements(uint32_t depth, uint32_t height, uint32_t width,
                              uint32_t channels) {
    return (size_t)depth * height * width * channels;
}

static int conv_op(encoder_context *encoder, h3_gpu_tensor *output,
                   const h3_gpu_tensor *input, const encoder_conv *conv,
                   uint32_t depth, uint32_t height, uint32_t width,
                   h3_gpu_tensor *padded, char *error, size_t error_size) {
    const h3_gpu_tensor *source = input;
    uint32_t source_depth = depth, source_height = height, source_width = width;
    if (padded) {
        if (!gpu_op(encoder, h3_gpu_vae_encoder_pad_f32(
                encoder->gpu, padded, input, 1, depth, height, width,
                conv->input_channels, conv->depth_front, conv->height_before,
                conv->height_after, conv->width_before, conv->width_after),
                error, error_size, "visual encoder causal padding")) return 0;
        source = padded;
        source_depth += conv->depth_front;
        source_height += conv->height_before + conv->height_after;
        source_width += conv->width_before + conv->width_after;
    }
    return gpu_op(encoder, h3_gpu_conv3d_f32(
        encoder->gpu, output, source, conv->weight, conv->bias, 1,
        source_depth, source_height, source_width, conv->input_channels,
        conv->output_channels, conv->kernel, conv->kernel, conv->kernel,
        conv->stride_t, conv->stride_h, conv->stride_w), error, error_size,
        "visual encoder Conv3d");
}

static h3_gpu_tensor *run_conv(encoder_context *encoder,
                               h3_gpu_tensor *input,
                               const encoder_conv *conv,
                               uint32_t depth, uint32_t height, uint32_t width,
                               uint32_t *output_depth, uint32_t *output_height,
                               uint32_t *output_width, char *error,
                               size_t error_size) {
    uint32_t padded_d = depth + conv->depth_front;
    uint32_t padded_h = height + conv->height_before + conv->height_after;
    uint32_t padded_w = width + conv->width_before + conv->width_after;
    *output_depth = (padded_d - conv->kernel) / conv->stride_t + 1;
    *output_height = (padded_h - conv->kernel) / conv->stride_h + 1;
    *output_width = (padded_w - conv->kernel) / conv->stride_w + 1;
    h3_gpu_tensor *padded = NULL;
    if (conv->depth_front || conv->height_before || conv->height_after ||
        conv->width_before || conv->width_after)
        padded = h3_gpu_tensor_new_f32(
            encoder->gpu, tensor_elements(padded_d, padded_h, padded_w,
                                           conv->input_channels));
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        encoder->gpu, tensor_elements(*output_depth, *output_height,
                                      *output_width, conv->output_channels));
    int ok = output && (!((conv->depth_front || conv->height_before ||
                           conv->height_after || conv->width_before ||
                           conv->width_after)) || padded);
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder convolution");
    } else {
        ok = gpu_op(encoder, h3_gpu_begin(encoder->gpu), error, error_size,
                    "begin visual encoder convolution") &&
             conv_op(encoder, output, input, conv, depth, height, width, padded,
                     error, error_size) &&
             gpu_op(encoder, h3_gpu_submit(encoder->gpu), error, error_size,
                    "submit visual encoder convolution");
    }
    h3_gpu_tensor_free(padded);
    if (!ok) {
        h3_gpu_tensor_free(output);
        return NULL;
    }
    return output;
}

static h3_gpu_tensor *run_block(encoder_context *encoder,
                                h3_gpu_tensor *input,
                                const encoder_block *block, uint32_t depth,
                                uint32_t height, uint32_t width,
                                uint32_t input_channels,
                                uint32_t output_channels, char *error,
                                size_t error_size) {
    size_t input_count = tensor_elements(depth, height, width, input_channels);
    size_t output_count = tensor_elements(depth, height, width, output_channels);
    size_t pad1_count = tensor_elements(depth + 2, height + 2, width + 2,
                                        input_channels);
    size_t pad2_count = tensor_elements(depth + 2, height + 2, width + 2,
                                        output_channels);
    h3_gpu_tensor *norm1 = h3_gpu_tensor_new_f32(encoder->gpu, input_count);
    h3_gpu_tensor *pad1 = h3_gpu_tensor_new_f32(encoder->gpu, pad1_count);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(encoder->gpu, output_count);
    h3_gpu_tensor *norm2 = h3_gpu_tensor_new_f32(encoder->gpu, output_count);
    h3_gpu_tensor *pad2 = h3_gpu_tensor_new_f32(encoder->gpu, pad2_count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(encoder->gpu, output_count);
    h3_gpu_tensor *shortcut = block->has_shortcut ?
        h3_gpu_tensor_new_f32(encoder->gpu, output_count) : NULL;
    int ok = norm1 && pad1 && hidden && norm2 && pad2 && output &&
             (!block->has_shortcut || shortcut);
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder residual block");
        goto done;
    }
    ok = gpu_op(encoder, h3_gpu_begin(encoder->gpu), error, error_size,
                "begin visual encoder residual block") &&
         gpu_op(encoder, h3_gpu_vae_encoder_group_norm_silu_f32(
             encoder->gpu, norm1, input, block->norm1.weight, block->norm1.bias,
             1, depth, height, width, input_channels, GROUPS, 1e-6f),
             error, error_size, "visual encoder group norm 1") &&
         conv_op(encoder, hidden, norm1, &block->conv1, depth, height, width,
                 pad1, error, error_size) &&
         gpu_op(encoder, h3_gpu_vae_encoder_group_norm_silu_f32(
             encoder->gpu, norm2, hidden, block->norm2.weight, block->norm2.bias,
             1, depth, height, width, output_channels, GROUPS, 1e-6f),
             error, error_size, "visual encoder group norm 2") &&
         conv_op(encoder, output, norm2, &block->conv2, depth, height, width,
                 pad2, error, error_size);
    const h3_gpu_tensor *residual = input;
    if (ok && block->has_shortcut) {
        ok = conv_op(encoder, shortcut, input, &block->shortcut, depth, height,
                     width, NULL, error, error_size);
        residual = shortcut;
    }
    if (ok) ok = gpu_op(encoder, h3_gpu_add_scaled_f32(
        encoder->gpu, output, residual, output, 1.0f, 1.0f,
        (uint32_t)output_count), error, error_size,
        "visual encoder residual add");
    if (ok) ok = gpu_op(encoder, h3_gpu_submit(encoder->gpu), error, error_size,
                        "submit visual encoder residual block");

done:
    h3_gpu_tensor_free(norm1);
    h3_gpu_tensor_free(pad1);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm2);
    h3_gpu_tensor_free(pad2);
    h3_gpu_tensor_free(shortcut);
    if (!ok) {
        h3_gpu_tensor_free(output);
        return NULL;
    }
    return output;
}

static float *encode_tile(encoder_context *encoder, const float *pixels,
                          int frames, int height, int width,
                          int *latent_time, char *error, size_t error_size) {
    size_t pixel_count = (size_t)frames * height * width * RGB_CHANNELS;
    float *normalized = malloc(pixel_count * sizeof(*normalized));
    if (!normalized) {
        fail(error, error_size, "out of memory normalizing visual anchor");
        return NULL;
    }
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    size_t destination = 0;
    for (int time = 0; time < frames; time++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int channel = 0; channel < RGB_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * frames + time) * height +
                                     y) * width + x;
                    normalized[destination++] =
                        (pixels[source] - mean[channel]) / deviation[channel];
                }
    h3_gpu_tensor *hidden = h3_gpu_tensor_from_f32(
        encoder->gpu, normalized, pixel_count);
    free(normalized);
    if (!hidden) {
        fail(error, error_size, "cannot allocate visual encoder pixels");
        return NULL;
    }
    uint32_t depth = (uint32_t)frames, h = (uint32_t)height, w = (uint32_t)width;
    uint32_t next_d, next_h, next_w;
    h3_gpu_tensor *next = run_conv(encoder, hidden, &encoder->conv_in,
                                   depth, h, w, &next_d, &next_h, &next_w,
                                   error, error_size);
    free_tensor(&hidden);
    hidden = next;
    if (!hidden) return NULL;
    depth = next_d; h = next_h; w = next_w;
    uint32_t previous = 128;
    for (int level = 0; level < LEVELS && hidden; level++) {
        uint32_t channels = level_channels[level];
        for (int block = 0; block < BLOCKS && hidden; block++) {
            uint32_t input_channels = block ? channels : previous;
            next = run_block(encoder, hidden,
                             &encoder->levels[level].blocks[block], depth, h, w,
                             input_channels, channels, error, error_size);
            free_tensor(&hidden);
            hidden = next;
        }
        if (hidden && encoder->levels[level].has_downsample) {
            next = run_conv(encoder, hidden,
                            &encoder->levels[level].downsample, depth, h, w,
                            &next_d, &next_h, &next_w, error, error_size);
            free_tensor(&hidden);
            hidden = next;
            depth = next_d; h = next_h; w = next_w;
        }
        previous = channels;
    }
    if (!hidden) return NULL;

    size_t hidden_count = tensor_elements(depth, h, w, 1024);
    size_t padded_count = tensor_elements(depth + 2, h + 2, w + 2, 1024);
    size_t moment_count = tensor_elements(depth, h, w, MOMENT_CHANNELS);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_f32(encoder->gpu, hidden_count);
    h3_gpu_tensor *padded = h3_gpu_tensor_new_f32(encoder->gpu, padded_count);
    h3_gpu_tensor *moments = h3_gpu_tensor_new_f32(encoder->gpu, moment_count);
    h3_gpu_tensor *quant = h3_gpu_tensor_new_f32(encoder->gpu, moment_count);
    int ok = norm && padded && moments && quant;
    if (!ok) {
        fail(error, error_size, "cannot allocate visual encoder output");
    } else {
        ok = gpu_op(encoder, h3_gpu_begin(encoder->gpu), error, error_size,
                    "begin visual encoder output") &&
             gpu_op(encoder, h3_gpu_vae_encoder_group_norm_silu_f32(
                 encoder->gpu, norm, hidden, encoder->norm_out.weight,
                 encoder->norm_out.bias, 1, depth, h, w, 1024, GROUPS, 1e-6f),
                 error, error_size, "visual encoder output norm") &&
             conv_op(encoder, moments, norm, &encoder->conv_out, depth, h, w,
                     padded, error, error_size) &&
             conv_op(encoder, quant, moments, &encoder->quant, depth, h, w,
                     NULL, error, error_size) &&
             gpu_op(encoder, h3_gpu_submit(encoder->gpu), error, error_size,
                    "submit visual encoder output");
    }
    float *raw = ok ? malloc(moment_count * sizeof(*raw)) : NULL;
    size_t latent_count = (size_t)encoder->output_channels * depth * h * w;
    float *latent = ok ? malloc(latent_count * sizeof(*latent)) : NULL;
    if (ok && (!raw || !latent ||
               !h3_gpu_tensor_read_f32(quant, raw, moment_count))) {
        fail(error, error_size, "cannot read visual encoder latent");
        ok = 0;
    }
    if (ok) for (int channel = 0; channel < encoder->output_channels; channel++)
        for (uint32_t time = 0; time < depth; time++)
            for (uint32_t y = 0; y < h; y++)
                for (uint32_t x = 0; x < w; x++) {
                    size_t source = (((size_t)time * h + y) * w + x) *
                                    MOMENT_CHANNELS + channel;
                    size_t target = (((size_t)channel * depth + time) * h + y) *
                                    w + x;
                    latent[target] = encoder->raw_moments ? raw[source] :
                        (raw[source] - encoder->latent_mean[channel]) /
                        encoder->latent_std[channel];
                }
    free(raw);
    free_tensor(&hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(padded);
    h3_gpu_tensor_free(moments);
    h3_gpu_tensor_free(quant);
    if (!ok) {
        free(latent);
        return NULL;
    }
    *latent_time = (int)depth;
    return latent;
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_axis_build(int extent, tile_axis *axis, char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < SPATIAL_RATIO || extent % SPATIAL_RATIO) {
        fail(error, error_size,
             "visual encoder extent must be a multiple of %d", SPATIAL_RATIO);
        return 0;
    }
    if (extent <= TILE_PIXELS) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing encoder tiles");
            return 0;
        }
        return 1;
    }
    int count = (extent + TILE_PIXELS - 1) / TILE_PIXELS;
    while (TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing encoder tiles");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = TILE_PIXELS * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + TILE_PIXELS -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = TILE_PIXELS;
    return 1;
}

static float *extract_pixel_tile(const float *pixels, int frames,
                                 int full_h, int full_w, int start_y,
                                 int start_x, int tile_h, int tile_w,
                                 char *error, size_t error_size) {
    size_t count = (size_t)RGB_CHANNELS * frames * tile_h * tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting visual anchor tile");
        return NULL;
    }
    for (int channel = 0; channel < RGB_CHANNELS; channel++)
        for (int time = 0; time < frames; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * frames + time) * full_h +
                                  start_y + y) * full_w + start_x;
                size_t destination = (((size_t)channel * frames + time) *
                                       tile_h + y) * tile_w;
                memcpy(tile + destination, pixels + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

static int stitch_latents(float **tiles, int channels, int latent_time,
                          const tile_axis *y_axis, const tile_axis *x_axis,
                          h3_video_latent *output, char *error,
                          size_t error_size) {
    int tile_h = y_axis->length / SPATIAL_RATIO;
    int tile_w = x_axis->length / SPATIAL_RATIO;
    int full_h = (y_axis->starts[y_axis->count - 1] + y_axis->length) /
                 SPATIAL_RATIO;
    int full_w = (x_axis->starts[x_axis->count - 1] + x_axis->length) /
                 SPATIAL_RATIO;
    size_t count = (size_t)channels * latent_time * full_h * full_w;
    float *values = malloc(count * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory stitching visual latents");
        return 0;
    }
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ?
                y_axis->overlaps[tile_y - 1] / SPATIAL_RATIO : 0;
            int overlap_x = tile_x ?
                x_axis->overlaps[tile_x - 1] / SPATIAL_RATIO : 0;
            int keep_h = tile_h - (tile_y + 1 < y_axis->count ?
                y_axis->overlaps[tile_y] / SPATIAL_RATIO : 0);
            int keep_w = tile_w - (tile_x + 1 < x_axis->count ?
                x_axis->overlaps[tile_x] / SPATIAL_RATIO : 0);
            int destination_y = y_axis->starts[tile_y] / SPATIAL_RATIO;
            int destination_x = x_axis->starts[tile_x] / SPATIAL_RATIO;
            for (int channel = 0; channel < channels; channel++)
                for (int time = 0; time < latent_time; time++)
                    for (int y = 0; y < keep_h; y++)
                        for (int x = 0; x < keep_w; x++) {
                            size_t local = (((size_t)channel * latent_time + time) *
                                            tile_h + y) * tile_w + x;
                            float value = current[local];
                            if (above && y < overlap_y) {
                                size_t source = (((size_t)channel * latent_time +
                                    time) * tile_h + tile_h - overlap_y + y) *
                                    tile_w + x;
                                float weight = (float)y / (float)overlap_y;
                                value = above[source] * (1.0f - weight) +
                                        value * weight;
                            }
                            if (left && x < overlap_x) {
                                size_t source = (((size_t)channel * latent_time +
                                    time) * tile_h + y) * tile_w +
                                    tile_w - overlap_x + x;
                                float weight = (float)x / (float)overlap_x;
                                value = left[source] * (1.0f - weight) +
                                        value * weight;
                            }
                            size_t destination =
                                (((size_t)channel * latent_time + time) * full_h +
                                  destination_y + y) * full_w +
                                 destination_x + x;
                            values[destination] = value;
                        }
        }
    output->time = latent_time;
    output->height = full_h;
    output->width = full_w;
    output->values = values;
    return 1;
}

static int open_encoder(encoder_context *encoder, const encoder_names *names,
                        int raw_moments, const char *weight_directory,
                        const char *shader_source_path,
                        char *error, size_t error_size) {
    memset(encoder, 0, sizeof(*encoder));
    if (!weight_directory || !*weight_directory || !shader_source_path ||
        !*shader_source_path) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    encoder->names = names;
    encoder->raw_moments = raw_moments;
    encoder->output_channels = raw_moments ? MOMENT_CHANNELS : LATENT_CHANNELS;
    encoder->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (encoder->gpu)
        h3_gpu_profile_set_label(encoder->gpu, raw_moments ?
                                 "image VAE encoder" : "video VAE encoder");
    if (encoder->gpu)
        encoder->store = h3_weight_store_open(weight_directory, error,
                                              error_size);
    int ok = encoder->gpu && encoder->store &&
         (raw_moments ||
          load_normalization(encoder, weight_directory, error, error_size)) &&
         load_weights(encoder, error, error_size);
    if (!ok) cleanup(encoder);
    return ok;
}

static int encode_with(encoder_context *encoder, const float *pixels,
                       int frames, int height, int width,
                       h3_video_encoder_progress progress,
                       void *progress_opaque, h3_video_latent *output,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!pixels || !output || frames < 1 || height < 32 || width < 32 ||
        height % SPATIAL_RATIO || width % SPATIAL_RATIO) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    tile_axis y_axis, x_axis;
    int ok = tile_axis_build(height, &y_axis, error, error_size) &&
             tile_axis_build(width, &x_axis, error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis);
        tile_axis_free(&x_axis);
        return 0;
    }
    int tile_count = y_axis.count * x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory allocating visual encoder tiles");
        ok = 0;
    }
    int latent_time = 0;
    for (int y = 0, completed = 0; ok && y < y_axis.count; y++)
        for (int x = 0; ok && x < x_axis.count; x++, completed++) {
            float *tile = extract_pixel_tile(
                pixels, frames, height, width, y_axis.starts[y], x_axis.starts[x],
                y_axis.length, x_axis.length, error, error_size);
            int current_time = 0;
            if (tile) tiles[completed] = encode_tile(
                encoder, tile, frames, y_axis.length, x_axis.length,
                &current_time, error, error_size);
            free(tile);
            ok = tiles[completed] != NULL &&
                 (!latent_time || latent_time == current_time);
            if (ok) latent_time = current_time;
            if (ok && progress) progress(completed + 1, tile_count, progress_opaque);
        }
    if (ok) ok = stitch_latents(tiles, encoder->output_channels, latent_time,
                                &y_axis, &x_axis, output, error,
                                error_size) &&
                 h3_gpu_get_stats(encoder->gpu, &output->gpu_stats);
    if (tiles) for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    tile_axis_free(&y_axis);
    tile_axis_free(&x_axis);
    if (!ok) h3_video_latent_free(output);
    return ok;
}

static int encode_visual(const encoder_names *names, int raw_moments,
                         const char *weight_directory,
                         const char *shader_source_path,
                         const float *pixels, int frames, int height, int width,
                         h3_video_encoder_progress progress,
                         void *progress_opaque, h3_video_latent *output,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!pixels || !output || frames < 1 || height < 32 || width < 32 ||
        height % SPATIAL_RATIO || width % SPATIAL_RATIO) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    encoder_context encoder;
    if (!open_encoder(&encoder, names, raw_moments, weight_directory,
                      shader_source_path, error, error_size)) return 0;
    int ok = encode_with(&encoder, pixels, frames, height, width, progress,
                         progress_opaque, output, error, error_size);
    cleanup(&encoder);
    return ok;
}

struct h3_image_vae_encoder {
    encoder_context context;
};

h3_image_vae_encoder *h3_image_vae_encoder_open(const char *weight_directory,
                                                const char *shader_source_path,
                                                char *error,
                                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_image_vae_encoder *encoder = calloc(1, sizeof(*encoder));
    if (!encoder) {
        fail(error, error_size, "out of memory opening the image VAE encoder");
        return NULL;
    }
    if (!open_encoder(&encoder->context, &image_names, 1, weight_directory,
                      shader_source_path, error, error_size)) {
        free(encoder);
        return NULL;
    }
    return encoder;
}

int h3_image_vae_encoder_moments(h3_image_vae_encoder *encoder,
                                 const float *pixels, int height, int width,
                                 h3_video_encoder_progress progress,
                                 void *progress_opaque, h3_video_latent *output,
                                 char *error, size_t error_size) {
    if (!encoder) {
        fail(error, error_size, "invalid visual encoder arguments");
        return 0;
    }
    return encode_with(&encoder->context, pixels, 1, height, width, progress,
                       progress_opaque, output, error, error_size);
}

void h3_image_vae_encoder_close(h3_image_vae_encoder *encoder) {
    if (!encoder) return;
    cleanup(&encoder->context);
    free(encoder);
}

int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size) {
    return encode_visual(&video_names, 0, weight_directory, shader_source_path,
                         pixels, frames, height, width, progress,
                         progress_opaque, output, error, error_size);
}

int h3_image_vae_encode_moments(const char *weight_directory,
                                const char *shader_source_path,
                                const float *pixels, int height, int width,
                                h3_video_encoder_progress progress,
                                void *progress_opaque, h3_video_latent *output,
                                char *error, size_t error_size) {
    return encode_visual(&image_names, 1, weight_directory, shader_source_path,
                         pixels, 1, height, width, progress, progress_opaque,
                         output, error, error_size);
}

void h3_video_latent_free(h3_video_latent *latent) {
    if (!latent) return;
    free(latent->values);
    memset(latent, 0, sizeof(*latent));
}
