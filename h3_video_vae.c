#include "h3_video_vae.h"

#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 24,
    CHUNK_LATENT_TIME = 7,
    FIRST_CHUNK_FRAMES = 22,
    FRAME_OFFSET = 3,
    HIDDEN = 2048,
    LAYERS = 36,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    REGISTERS = 4,
    SUFFIX = 5,
    ROPE_HALF = 24,
    OUTPUT_PATCH = 3 * 4 * 16 * 16,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *scale1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *w1;
    h3_gpu_tensor *w1_b;
    h3_gpu_tensor *w2;
    h3_gpu_tensor *w2_b;
    h3_gpu_tensor *scale2;
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    vae_block blocks[LAYERS];
    h3_gpu_tensor *post_w;
    h3_gpu_tensor *post_b;
    h3_gpu_tensor *embed_w;
    h3_gpu_tensor *embed_b;
    h3_gpu_tensor *registers;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *proj_w;
    h3_gpu_tensor *proj_b;
    h3_gpu_tensor *latent;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *projected;
    uint32_t patches;
    uint32_t sequence;
    /* When >1, activations hold tile_batch independent tiles back to back
     * ([tile][sequence][...]); row-wise ops run on tile_batch*sequence rows
     * and attention runs with a leading batch dimension. */
    uint32_t tile_batch;
    int latent_h;
    int latent_w;
    int latent_t;
    int output_frames;
} vae_context;

/* One independent set of per-tile decode activations. Two slots let the CPU
 * encode tile k+1 while the GPU still runs tile k; the ordered Metal queue and
 * default buffer hazard tracking keep execution order and bytes identical. */
typedef struct {
    h3_gpu_tensor *latent;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *projected;
} vae_activation_slot;

struct h3_video_vae_decoder {
    vae_context vae;
    tile_axis y_axis;
    tile_axis x_axis;
    int latent_h;
    int latent_w;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
    h3_video_vae_progress progress;
    void *progress_opaque;
    /* Spatial tiles each run the full block stack, so a raw per-tile count
     * would restart the bar once per tile; these carry the tile's place in
     * the whole decode so the reported figure only moves forward. */
    int progress_tile;
    int progress_tiles;
    /* Encode-ahead pipeline state. slots[0] aliases the vae_context's own
     * activation tensors; slots[1] is an extra set owned here. */
    int pipeline;
    vae_activation_slot slots[2];
};

static void slot_capture(vae_context *vae, vae_activation_slot *slot) {
    slot->latent = vae->latent;
    slot->post = vae->post;
    slot->patch_hidden = vae->patch_hidden;
    slot->hidden = vae->hidden;
    slot->norm = vae->norm;
    slot->qkv = vae->qkv;
    slot->query = vae->query;
    slot->key = vae->key;
    slot->value = vae->value;
    slot->heads = vae->heads;
    slot->branch = vae->branch;
    slot->ff1 = vae->ff1;
    slot->activated = vae->activated;
    slot->projected = vae->projected;
}

static void slot_bind(vae_context *vae, const vae_activation_slot *slot) {
    vae->latent = slot->latent;
    vae->post = slot->post;
    vae->patch_hidden = slot->patch_hidden;
    vae->hidden = slot->hidden;
    vae->norm = slot->norm;
    vae->qkv = slot->qkv;
    vae->query = slot->query;
    vae->key = slot->key;
    vae->value = slot->value;
    vae->heads = slot->heads;
    vae->branch = slot->branch;
    vae->ff1 = slot->ff1;
    vae->activated = slot->activated;
    vae->projected = slot->projected;
}

static void free_tensor(h3_gpu_tensor **tensor);

static uint32_t vae_batch(const vae_context *vae);

static void slot_free(vae_activation_slot *slot) {
    free_tensor(&slot->latent);
    free_tensor(&slot->post);
    free_tensor(&slot->patch_hidden);
    free_tensor(&slot->hidden);
    free_tensor(&slot->norm);
    free_tensor(&slot->qkv);
    free_tensor(&slot->query);
    free_tensor(&slot->key);
    free_tensor(&slot->value);
    free_tensor(&slot->heads);
    free_tensor(&slot->branch);
    free_tensor(&slot->ff1);
    free_tensor(&slot->activated);
    free_tensor(&slot->projected);
}

/* Forwards a tile's block progress as a position in the whole decode. */
static void decoder_tile_progress(int completed, int total, void *opaque) {
    h3_video_vae_decoder *decoder = opaque;
    if (!decoder || !decoder->progress || total <= 0) return;
    int tiles = decoder->progress_tiles > 0 ? decoder->progress_tiles : 1;
    decoder->progress(decoder->progress_tile * total + completed,
                      tiles * total, decoder->progress_opaque);
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(vae_context *vae, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(vae->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *load_f32(vae_context *vae, const char *name, int ndim,
                               const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_f32(vae, name, 2, shape, error, error_size);
}

/* Rounds an F32 weight tensor to BF16 storage in place of the original.
 * Downstream h3_gpu_linear_f32 detects the BF16 weight and switches to a
 * mixed-precision MPSGraph product with F32 activations, halving weight
 * bandwidth. Default on; H3_DISABLE_VAE_BF16_GEMM restores full F32.
 * Measured at 768x1344 (15-tile chunk, M5 Pro): decode 36.9s -> 16.3s
 * (2.27x), peak 9.6 -> 5.1 GiB, PSNR vs F32 57.1 dB min over 22 frames
 * with 0.004% of pixels differing by more than two 8-bit levels. */
static h3_gpu_tensor *tensor_to_bf16(vae_context *vae, h3_gpu_tensor *f32,
                                     size_t elements, char *error,
                                     size_t error_size) {
    float *values = malloc(elements * sizeof(*values));
    uint16_t *rounded = malloc(elements * sizeof(*rounded));
    h3_gpu_tensor *result = NULL;
    if (values && rounded &&
        h3_gpu_tensor_read_f32(f32, values, elements)) {
        for (size_t index = 0; index < elements; index++) {
            uint32_t bits;
            memcpy(&bits, &values[index], sizeof(bits));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            rounded[index] = (uint16_t)(bits >> 16);
        }
        result = h3_gpu_tensor_new_bf16(vae->gpu, elements);
        if (result && !h3_gpu_tensor_write_bf16(result, rounded, elements)) {
            h3_gpu_tensor_free(result);
            result = NULL;
        }
    }
    free(values);
    free(rounded);
    if (!result) {
        fail(error, error_size, "cannot round video VAE weight to BF16");
        return NULL;
    }
    h3_gpu_tensor_free(f32);
    return result;
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
}

static void cleanup(vae_context *vae) {
    if (!vae) return;
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->norm_out_w);
    free_tensor(&vae->norm_out_b); free_tensor(&vae->proj_w);
    free_tensor(&vae->proj_b); free_tensor(&vae->latent); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F2(qkv_w, "attn.to_qkv.weight", INNER * 3, HIDDEN);
    F1(qkv_b, "attn.to_qkv.bias", INNER * 3);
    F2(out_w, "attn.to_out.weight", HIDDEN, INNER);
    F1(out_b, "attn.to_out.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F2(w1, "ff.w1.weight", FFN * 2, HIDDEN);
    F1(w1_b, "ff.w1.bias", FFN * 2);
    F2(w2, "ff.w2.weight", HIDDEN, FFN);
    F1(w2_b, "ff.w2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
#undef F1
#undef F2
    if (!getenv("H3_DISABLE_VAE_BF16_GEMM")) {
        if (!(block->qkv_w = tensor_to_bf16(vae, block->qkv_w,
                  (size_t)INNER * 3 * HIDDEN, error, error_size)) ||
            !(block->out_w = tensor_to_bf16(vae, block->out_w,
                  (size_t)HIDDEN * INNER, error, error_size)) ||
            !(block->w1 = tensor_to_bf16(vae, block->w1,
                  (size_t)FFN * 2 * HIDDEN, error, error_size)) ||
            !(block->w2 = tensor_to_bf16(vae, block->w2,
                  (size_t)HIDDEN * FFN, error, error_size)))
            return 0;
    }
    return 1;
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
    vae->post_w = load_f32(vae, "post_quant_conv.weight", 5, post_shape,
                           error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                      LATENT_CHANNELS, error, error_size);
    vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                      error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_f32(vae, "decoder.register_tokens", 3,
                              register_shape, error, error_size);
    return vae->post_w && vae->post_b && vae->embed_w && vae->embed_b &&
           vae->registers;
}

static int load_output_weights(vae_context *vae, char *error,
                               size_t error_size) {
    vae->norm_out_w = f1(vae, "decoder.norm_out.weight", HIDDEN,
                          error, error_size);
    vae->norm_out_b = f1(vae, "decoder.norm_out.bias", HIDDEN,
                          error, error_size);
    vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH, HIDDEN,
                      error, error_size);
    vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                      error, error_size);
    return vae->norm_out_w && vae->norm_out_b && vae->proj_w && vae->proj_b;
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

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
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
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static void prepare_input_rows(const vae_context *vae, const float *input,
                               const float *mean, const float *deviation,
                               float *rows) {
    size_t row = 0;
    for (int time = 0; time < CHUNK_LATENT_TIME; time++) {
        int source_t = time < vae->latent_t ? time : vae->latent_t - 1;
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)vae->latent_t +
                        (size_t)source_t) *
                        (size_t)vae->latent_h + (size_t)height) *
                        (size_t)vae->latent_w + (size_t)width;
                    rows[row++] = input[source] * deviation[channel] +
                                  mean[channel];
                }
    }
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory padding video latent");
        return 0;
    }
    prepare_input_rows(vae, input, mean, deviation, rows);
    vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows, patch_elements);
    free(rows);
    return vae->latent != NULL;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    uint32_t batch = vae_batch(vae);
    size_t tile_count = (size_t)vae->sequence * ROPE_HALF;
    size_t count = tile_count * batch;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating video VAE RoPE");
        return 0;
    }
    uint32_t row = 0;
    for (int t = 0; t < CHUNK_LATENT_TIME; t++)
        for (int h = 0; h < vae->latent_h; h++)
            for (int w = 0; w < vae->latent_w; w++, row++) {
                float axes[] = {
                    2.0f * (((float)t + 0.5f) /
                            (float)CHUNK_LATENT_TIME) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
                };
                for (uint32_t axis = 0; axis < 3; axis++)
                    for (uint32_t frequency = 0; frequency < 8; frequency++) {
                        float inverse = 1.0f /
                            powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = (size_t)row * ROPE_HALF +
                                        axis * 8 + frequency;
                        cosines[offset] = cosf(angle);
                        sines[offset] = sinf(angle);
                    }
            }
    while (row < vae->sequence) {
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
        row++;
    }
    for (uint32_t copy = 1; copy < batch; copy++) {
        memcpy(cosines + (size_t)copy * tile_count, cosines,
               tile_count * sizeof(*cosines));
        memcpy(sines + (size_t)copy * tile_count, sines,
               tile_count * sizeof(*sines));
    }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    free(cosines); free(sines);
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate video VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static uint32_t vae_batch(const vae_context *vae) {
    return vae->tile_batch ? vae->tile_batch : 1;
}

/* Tiles per batched decode pass. Bit-exact and 2.4x lighter on GPU wait
 * than per-tile submits, but MPS charges a CPU encode cost that grows with
 * shape and stays serial with GPU execution on current macOS, so the wall
 * time is neutral here; opt in with H3_VAE_TILE_BATCH=N (0/1 disables). */
static uint32_t vae_tile_batch_for(int tile_total) {
    if (tile_total <= 1 || getenv("H3_DISABLE_VAE_TILE_BATCH")) return 1;
    long batch = 1;
    const char *text = getenv("H3_VAE_TILE_BATCH");
    if (text && *text) {
        char *end = NULL;
        long parsed = strtol(text, &end, 10);
        if (end && !*end && parsed >= 0 && parsed <= 24) batch = parsed;
    }
    if (batch <= 1) return 1;
    if (batch > tile_total) batch = tile_total;
    return (uint32_t)batch;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    size_t batch = vae_batch(vae);
    size_t patches = (size_t)vae->patches * batch;
    size_t sequence = (size_t)vae->sequence * batch;
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * HIDDEN),
        F32(hidden, sequence * HIDDEN),
        F32(norm, sequence * HIDDEN),
        F32(qkv, sequence * INNER * 3),
        F32(query, sequence * INNER), F32(key, sequence * INNER),
        F32(value, sequence * INNER), F32(heads, sequence * INNER),
        F32(branch, sequence * HIDDEN), F32(ff1, sequence * FFN * 2),
        F32(activated, sequence * FFN),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate video VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    return 1;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t batch = vae_batch(vae);
    uint32_t rows = vae->sequence * batch;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "video VAE attention norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->qkv, vae->norm, weight->qkv_w,
        weight->qkv_b, rows, HIDDEN, INNER * 3), "video VAE QKV");
    OP(h3_gpu_video_qkv_rope_f32(vae->gpu, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
    OP(h3_gpu_sdpa_batch_f32(vae->gpu, vae->heads, vae->query, vae->key,
        vae->value, batch, vae->sequence, HEADS, HEAD_DIM,
        1.0f / sqrtf((float)HEAD_DIM)),
       "video VAE attention");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->heads, weight->out_w,
        weight->out_b, rows, INNER, HIDDEN), "video VAE attention output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "video VAE attention residual");
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm2,
        rows, HIDDEN, 1e-5f), "video VAE MLP norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->ff1, vae->norm, weight->w1,
        weight->w1_b, rows, HIDDEN, FFN * 2), "video VAE MLP input");
    OP(h3_gpu_swiglu_f32(vae->gpu, vae->activated, vae->ff1, rows, FFN),
       "video VAE SwiGLU");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->activated, weight->w2,
        weight->w2_b, rows, FFN, HIDDEN), "video VAE MLP output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "video VAE MLP residual");
#undef OP
    return 1;
}

static int run_decoder(vae_context *vae, h3_video_vae_progress progress,
                       void *progress_opaque, char *error,
                       size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin video VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack video VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers);

    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_begin(vae->gpu), "begin video VAE transformer block");
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit video VAE transformer block");
        free_block(&vae->blocks[index]);
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    OP(h3_gpu_begin(vae->gpu), "begin video VAE output projection");
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "video VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE decoder");
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
#undef OP
    return 1;
}

static int load_resident_weights(vae_context *vae,
                                 h3_video_vae_progress progress,
                                 void *progress_opaque, char *error,
                                 size_t error_size) {
    if (!load_input_weights(vae, error, error_size)) return 0;
    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    return load_output_weights(vae, error, error_size);
}

/* Tiled decoding keeps every VAE weight resident for the duration of the phase.
 * This uses about 9 GiB after the DiT has been retired, avoids rereading 9 GiB
 * per spatial tile, and permits one command-buffer chain per tile. The caller
 * owns the command chain: it must call h3_gpu_begin()/h3_gpu_continue() before
 * and h3_gpu_continue()/h3_gpu_submit() after, which lets the pipelined path
 * encode one tile while the previous one still runs on the GPU. */
static int run_resident_tile(vae_context *vae,
                             h3_video_vae_progress progress,
                             void *progress_opaque, char *error,
                             size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "tiled video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "tiled video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack tiled video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack tiled video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack tiled video VAE suffix");
    for (int index = 0; index < LAYERS; index++) {
        if (!run_block(vae, index, error, error_size)) {
            h3_gpu_tensor_free(zero);
            return 0;
        }
        /* Decoding costs about as much as loading did, so it reports too;
         * without this a caller's phase sits at its last value for minutes
         * and reads as a hang. */
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "tiled video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "tiled video VAE output projection");
    h3_gpu_tensor_free(zero);
#undef OP
    return 1;
}

static int unpack_frame_range(vae_context *vae, uint32_t tile_index,
                              int first_frame,
                              int frame_count, h3_video_frames *output,
                              char *error, size_t error_size) {
    if (first_frame < 0 || frame_count < 1 ||
        first_frame > vae->output_frames - frame_count ||
        tile_index >= vae_batch(vae)) {
        fail(error, error_size, "invalid video VAE output frame range");
        return 0;
    }
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = malloc(projected_count * sizeof(*rows));
    int pixel_h = vae->latent_h * 16;
    int pixel_w = vae->latent_w * 16;
    size_t rgb_count = (size_t)frame_count * (size_t)pixel_h *
                       (size_t)pixel_w * 3;
    float *rgb = malloc(rgb_count * sizeof(*rgb));
    if (!rows || !rgb) {
        free(rows); free(rgb);
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32_range(vae->projected,
                                      (size_t)tile_index * projected_count,
                                      rows, projected_count)) {
        free(rows); free(rgb);
        fail(error, error_size, "cannot read video VAE output patches");
        return 0;
    }
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    for (int output_frame = 0; output_frame < frame_count; output_frame++) {
        int frame = first_frame + output_frame;
        int decoded_t = frame + FRAME_OFFSET;
        if (vae->output_frames == FIRST_CHUNK_FRAMES && frame >= 17)
            decoded_t += 3;
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        for (int y = 0; y < pixel_h; y++) {
            int patch_y = y / 16, within_y = y % 16;
            for (int x = 0; x < pixel_w; x++) {
                int patch_x = x / 16, within_x = x % 16;
                size_t patch = ((size_t)patch_t * (size_t)vae->latent_h +
                    (size_t)patch_y) * (size_t)vae->latent_w +
                    (size_t)patch_x;
                for (int channel = 0; channel < 3; channel++) {
                    size_t component = ((((size_t)channel * 4 +
                        (size_t)within_t) * 16 + (size_t)within_y) * 16 +
                        (size_t)within_x);
                    float value = rows[patch * OUTPUT_PATCH + component] *
                                  deviation[channel] + mean[channel];
                    if (value < 0.0f) value = 0.0f;
                    if (value > 1.0f) value = 1.0f;
                    size_t destination = (((size_t)output_frame *
                        (size_t)pixel_h +
                        (size_t)y) * (size_t)pixel_w + (size_t)x) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
            }
        }
    }
    free(rows);
    output->frames = frame_count;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    return h3_gpu_get_stats(vae->gpu, &output->gpu_stats);
}

static int unpack_frames(vae_context *vae, uint32_t tile_index,
                         h3_video_frames *output,
                         char *error, size_t error_size) {
    return unpack_frame_range(vae, tile_index, 0, vae->output_frames, output,
                              error, error_size);
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

static int configured_tile_pixels(int pixel_height, int pixel_width) {
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    if (value && *value) {
        char *end = NULL;
        long pixels = strtol(value, &end, 10);
        if (end && !*end && pixels >= TILE_PIXELS && pixels <= 512 &&
            pixels % SPATIAL_RATIO == 0) return (int)pixels;
    }
    int best = TILE_PIXELS;
    uint64_t best_score = UINT64_MAX;
    for (int pixels = TILE_PIXELS; pixels <= 320;
         pixels += SPATIAL_RATIO) {
        uint64_t tiles = (uint64_t)tile_count_for_extent(pixel_height, pixels) *
            (uint64_t)tile_count_for_extent(pixel_width, pixels);
        /* The resident VAE is dominated by linears and activation traffic;
         * measured tile cost follows area more closely than cubic sequence
         * growth at these shapes. Attention is still included in each tile. */
        uint64_t score = tiles * (uint64_t)pixels * (uint64_t)pixels;
        if (score < best_score) {
            best = pixels;
            best_score = score;
        }
    }
    return best;
}

static int tile_axis_build(int extent, int tile_pixels, tile_axis *axis,
                           char *error,
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
    int count = tile_count_for_extent(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
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

static float *extract_latent_tile(const float *latent, int full_t,
                                  int full_h, int full_w, int start_t,
                                  int start_y, int start_x, int tile_t,
                                  int tile_h, int tile_w, char *error,
                                  size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_t *
                   (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting video VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int time = 0; time < tile_t; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * (size_t)full_t +
                    (size_t)(start_t + time)) * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)start_x;
                size_t destination = (((size_t)channel * (size_t)tile_t +
                    (size_t)time) * (size_t)tile_h + (size_t)y) *
                    (size_t)tile_w;
                memcpy(tile + destination, latent + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

static int stitch_tiles(float **tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count,
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

/* Decode every spatial tile of one temporal chunk into tiles[]. With
 * pipeline set, tile k+1's latent extraction and command encoding overlap
 * tile k's GPU execution through h3_gpu_continue()/h3_gpu_flight_limit(),
 * alternating between two activation slots. Output bytes are unchanged:
 * per-tile op order is identical and the ordered queue plus default buffer
 * hazard tracking serialize GPU execution exactly as before. */
/* Decodes every spatial tile of one temporal chunk in a single batched GPU
 * pass: activations hold all tiles back to back, row-wise kernels see one
 * large row count, and attention runs with a leading tile batch dimension.
 * This lifts the small per-tile GEMMs to efficient shapes and collapses
 * per-tile command encoding and submission. */
static int decode_chunk_batched(vae_context *vae, const tile_axis *y_axis,
                                const tile_axis *x_axis,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int full_latent_h, int full_latent_w,
                                const float *latent_mean,
                                const float *latent_std, int selected_frame,
                                uint32_t first_tile, uint32_t batch,
                                float **tiles,
                                h3_video_vae_decoder *progress_decoder,
                                char *error, size_t error_size) {
    size_t tile_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows_host = malloc(tile_elements * batch * sizeof(*rows_host));
    if (!rows_host) {
        fail(error, error_size, "out of memory batching video VAE latents");
        return 0;
    }
    for (uint32_t index = 0; index < batch; index++) {
        int tile_y = (int)(first_tile + index) / x_axis->count;
        int tile_x = (int)(first_tile + index) % x_axis->count;
        float *input = extract_latent_tile(
            normalized_latent, latent_time, full_latent_h, full_latent_w,
            chunk * 5, y_axis->starts[tile_y] / SPATIAL_RATIO,
            x_axis->starts[tile_x] / SPATIAL_RATIO,
            CHUNK_LATENT_TIME, vae->latent_h, vae->latent_w,
            error, error_size);
        if (!input) {
            free(rows_host);
            return 0;
        }
        prepare_input_rows(vae, input, latent_mean, latent_std,
                           rows_host + (size_t)index * tile_elements);
        free(input);
    }
    free_tensor(&vae->latent);
    vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows_host,
                                         tile_elements * batch);
    free(rows_host);
    if (!vae->latent) {
        fail(error, error_size, "cannot allocate batched video VAE latent");
        return 0;
    }
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    uint32_t all_patches = vae->patches * batch;
    OP(h3_gpu_begin(vae->gpu), "begin batched video VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, all_patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "batched video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, all_patches, LATENT_CHANNELS, HIDDEN),
       "batched video VAE latent embedding");
    for (uint32_t index = 0; index < batch; index++) {
        size_t hidden_base = (size_t)index * vae->sequence * HIDDEN;
        OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, hidden_base,
            vae->patch_hidden, (size_t)index * vae->patches * HIDDEN,
            (size_t)vae->patches * HIDDEN), "pack batched video VAE patches");
        OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
            hidden_base + (size_t)vae->patches * HIDDEN, vae->registers, 0,
            REGISTERS * HIDDEN), "pack batched video VAE registers");
        OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
            hidden_base + (size_t)(vae->patches + REGISTERS) * HIDDEN,
            zero, 0, HIDDEN), "pack batched video VAE suffix");
    }
    if (progress_decoder) {
        progress_decoder->progress_tiles = 1;
        progress_decoder->progress_tile = 0;
    }
    for (int index = 0; index < LAYERS; index++) {
        if (!run_block(vae, index, error, error_size)) {
            h3_gpu_tensor_free(zero);
            return 0;
        }
        /* Commit early so GPU execution overlaps the remaining CPU command
         * encoding; the ordered queue keeps execution order unchanged. */
        if (index % 6 == 5 &&
            !gpu_op(vae, h3_gpu_continue(vae->gpu), error, error_size,
                    "continue batched video VAE decoder")) {
            h3_gpu_tensor_free(zero);
            return 0;
        }
        if (progress_decoder && progress_decoder->progress)
            decoder_tile_progress(index + 1, LAYERS, progress_decoder);
    }
    uint32_t all_rows = vae->sequence * batch;
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, all_rows, HIDDEN, 1e-5f),
       "batched video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, all_rows, HIDDEN, OUTPUT_PATCH),
       "batched video VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit batched video VAE decoder");
    h3_gpu_tensor_free(zero);
#undef OP
    for (uint32_t index = 0; index < batch; index++) {
        h3_video_frames tile;
        memset(&tile, 0, sizeof(tile));
        int ok = selected_frame >= 0 ?
            unpack_frame_range(vae, index, selected_frame, 1, &tile,
                               error, error_size) :
            unpack_frames(vae, index, &tile, error, error_size);
        if (!ok) return 0;
        tiles[first_tile + index] = tile.rgb;
    }
    return 1;
}

static int decode_chunk_tiles(vae_context *vae, vae_activation_slot slots[2],
                              int pipeline, const tile_axis *y_axis,
                              const tile_axis *x_axis,
                              const float *normalized_latent, int latent_time,
                              int chunk, int full_latent_h, int full_latent_w,
                              const float *latent_mean,
                              const float *latent_std, int selected_frame,
                              float **tiles,
                              h3_video_vae_decoder *progress_decoder,
                              char *error, size_t error_size) {
    int tile_count = y_axis->count * x_axis->count;
    if (vae_batch(vae) > 1) {
        uint32_t stride = vae_batch(vae);
        for (uint32_t first = 0; first < (uint32_t)tile_count;
             first += stride) {
            uint32_t group = (uint32_t)tile_count - first < stride ?
                (uint32_t)tile_count - first : stride;
            if (!decode_chunk_batched(vae, y_axis, x_axis, normalized_latent,
                                      latent_time, chunk, full_latent_h,
                                      full_latent_w, latent_mean, latent_std,
                                      selected_frame, first, group, tiles,
                                      progress_decoder, error, error_size))
                return 0;
        }
        return 1;
    }
    int committed = 0;
    int ok = 1;
    for (int index = 0; index < tile_count && ok; index++) {
        int tile_y = index / x_axis->count;
        int tile_x = index % x_axis->count;
        vae_activation_slot *slot = &slots[pipeline ? (index & 1) : 0];
        slot_bind(vae, slot);
        h3_gpu_set_graph_salt(vae->gpu, pipeline ? (index & 1) : 0);
        if (progress_decoder) progress_decoder->progress_tile = index;
        float *input = extract_latent_tile(
            normalized_latent, latent_time, full_latent_h, full_latent_w,
            chunk * 5, y_axis->starts[tile_y] / SPATIAL_RATIO,
            x_axis->starts[tile_x] / SPATIAL_RATIO,
            CHUNK_LATENT_TIME, vae->latent_h, vae->latent_w,
            error, error_size);
        if (!input) {
            ok = 0;
            break;
        }
        free_tensor(&vae->latent);
        ok = prepare_input(vae, input, latent_mean, latent_std,
                           error, error_size);
        free(input);
        slot->latent = vae->latent;
        if (!ok) break;
        if (!pipeline || index == 0)
            ok = gpu_op(vae, h3_gpu_begin(vae->gpu), error, error_size,
                        "begin tiled video VAE decoder");
        ok = ok && run_resident_tile(
            vae,
            progress_decoder && progress_decoder->progress ?
                decoder_tile_progress : NULL,
            progress_decoder, error, error_size);
        if (!ok) break;
        if (!pipeline) {
            ok = gpu_op(vae, h3_gpu_submit(vae->gpu), error, error_size,
                        "submit tiled video VAE decoder");
            if (!ok) break;
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(vae, 0, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(vae, 0, &tile, error, error_size);
            if (ok) tiles[index] = tile.rgb;
            continue;
        }
        /* Commit this tile without waiting, so its GPU execution overlaps
         * the next tile's CPU latent extraction and command encoding. */
        ok = gpu_op(vae, h3_gpu_continue(vae->gpu), error, error_size,
                    "commit tiled video VAE tile");
        if (!ok) break;
        committed = index + 1;
        if (index == 0) continue;
        /* The previous tile's buffer is the oldest in flight; wait for it
         * alone and read it back while the current tile keeps running. */
        ok = gpu_op(vae, h3_gpu_flight_limit(vae->gpu, 1), error, error_size,
                    "wait prior video VAE tile");
        if (!ok) break;
        slot_bind(vae, &slots[(index - 1) & 1]);
        h3_video_frames tile;
        memset(&tile, 0, sizeof(tile));
        ok = selected_frame >= 0 ?
            unpack_frame_range(vae, 0, selected_frame, 1, &tile,
                               error, error_size) :
            unpack_frames(vae, 0, &tile, error, error_size);
        if (ok) tiles[index - 1] = tile.rgb;
    }
    if (pipeline && committed) {
        /* Close the chain: submit the trailing empty buffer, wait for and
         * validate everything still in flight, then read the last tile. */
        int drained = gpu_op(vae, h3_gpu_submit(vae->gpu), error, error_size,
                             "submit tiled video VAE decoder");
        ok = ok && drained;
        if (ok) {
            slot_bind(vae, &slots[(committed - 1) & 1]);
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(vae, 0, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(vae, 0, &tile, error, error_size);
            if (ok) tiles[committed - 1] = tile.rgb;
        }
    }
    slot_bind(vae, &slots[0]);
    h3_gpu_set_graph_salt(vae->gpu, 0);
    return ok;
}

static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    int frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    decoder->progress_tiles = tile_count;
    int ok = decode_chunk_tiles(
        &decoder->vae, decoder->slots, decoder->pipeline,
        &decoder->y_axis, &decoder->x_axis, normalized_latent, latent_time,
        chunk, decoder->latent_h, decoder->latent_w, decoder->latent_mean,
        decoder->latent_std, selected_frame, tiles, decoder,
        error, error_size);
    if (ok) ok = stitch_tiles(tiles, &decoder->y_axis, &decoder->x_axis,
                              frame_count, output, error, error_size);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE arguments");
        return NULL;
    }
    h3_video_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating resident video VAE");
        return NULL;
    }
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    int ok = load_latent_normalization(
        weight_directory, decoder->latent_mean, decoder->latent_std,
        error, error_size) &&
        tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                        &decoder->y_axis, error, error_size) &&
        tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                        &decoder->x_axis, error, error_size);
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    vae_context *vae = &decoder->vae;
    if (ok) {
        vae->latent_h = decoder->y_axis.length / SPATIAL_RATIO;
        vae->latent_w = decoder->x_axis.length / SPATIAL_RATIO;
        vae->latent_t = CHUNK_LATENT_TIME;
        vae->output_frames = FIRST_CHUNK_FRAMES;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * vae->latent_h *
                                  vae->latent_w);
        vae->sequence = vae->patches + SUFFIX;
        int tile_total = decoder->y_axis.count * decoder->x_axis.count;
        vae->tile_batch = vae_tile_batch_for(tile_total);
        vae->weights = h3_weight_store_open(weight_directory,
                                             error, error_size);
        if (vae->weights)
            vae->gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (vae->gpu)
            h3_gpu_profile_set_label(vae->gpu, "resident video VAE decoder");
        ok = vae->weights && vae->gpu &&
             load_resident_weights(vae, progress, progress_opaque,
                                   error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size);
    }
    if (ok) {
        slot_capture(vae, &decoder->slots[0]);
        decoder->pipeline = vae->tile_batch <= 1 &&
            getenv("H3_VAE_PIPELINE") != NULL;
        if (decoder->pipeline) {
            /* A second activation set lets tile k+1 encode while tile k runs.
             * On failure fall back to the single-slot serial path. */
            vae_activation_slot blank;
            memset(&blank, 0, sizeof(blank));
            slot_bind(vae, &blank);
            char scratch[256];
            if (allocate_activations(vae, scratch, sizeof(scratch))) {
                slot_capture(vae, &decoder->slots[1]);
            } else {
                slot_capture(vae, &decoder->slots[1]);
                slot_free(&decoder->slots[1]);
                decoder->pipeline = 0;
            }
            slot_bind(vae, &decoder->slots[0]);
        }
    }
    if (!ok) {
        h3_video_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

void h3_video_vae_decoder_set_progress(h3_video_vae_decoder *decoder,
                                       h3_video_vae_progress progress,
                                       void *progress_opaque) {
    if (!decoder) return;
    decoder->progress = progress;
    decoder->progress_opaque = progress_opaque;
}

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || !output_frame_index ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid video VAE preview arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int local_frame = FIRST_CHUNK_FRAMES / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + local_frame;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    int ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, local_frame, output,
                                  error, error_size);
    if (ok) *output_frame_index = global_frame;
    return ok;
}

int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = malloc(output_elements * sizeof(*final_rgb));
    float *temporal_overlap = malloc(5 * frame_elements *
                                     sizeof(*temporal_overlap));
    if (!final_rgb || !temporal_overlap) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = 1;
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, -1, &decoded, error, error_size);
        if (!ok) break;
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder) {
    if (!decoder) return;
    if (decoder->slots[0].hidden)
        slot_bind(&decoder->vae, &decoder->slots[0]);
    slot_free(&decoder->slots[1]);
    cleanup(&decoder->vae);
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    free(decoder);
}

static int decode_chunked(const char *weight_directory,
                          const char *shader_source_path,
                          const float *normalized_latent, int latent_time,
                          int latent_height, int latent_width,
                          const float *latent_mean, const float *latent_std,
                          int tile_pixels,
                          h3_video_vae_progress progress, void *progress_opaque,
                          h3_video_frames *output, char *error,
                          size_t error_size) {
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int ok = tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                             &y_axis,
                             error, error_size) &&
             tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                             &x_axis,
                             error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: video VAE tiles %dx%d at %d pixels\n",
                x_axis.count, y_axis.count, tile_pixels);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = y_axis.length / SPATIAL_RATIO;
    vae.latent_w = x_axis.length / SPATIAL_RATIO;
    vae.latent_t = CHUNK_LATENT_TIME;
    vae.output_frames = FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    int chunked_tiles = y_axis.count * x_axis.count;
    vae.tile_batch = vae_tile_batch_for(chunked_tiles);
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (vae.weights)
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    ok = vae.weights && vae.gpu &&
         load_resident_weights(&vae, progress, progress_opaque,
                               error, error_size) &&
         prepare_rope(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size);
    vae_activation_slot slots[2];
    memset(slots, 0, sizeof(slots));
    int pipeline = 0;
    if (ok) {
        slot_capture(&vae, &slots[0]);
        pipeline = vae.tile_batch <= 1 &&
            getenv("H3_VAE_PIPELINE") != NULL;
        if (pipeline) {
            /* A second activation set lets tile k+1 encode while tile k runs.
             * On failure fall back to the single-slot serial path. */
            vae_activation_slot blank;
            memset(&blank, 0, sizeof(blank));
            slot_bind(&vae, &blank);
            char scratch[256];
            if (allocate_activations(&vae, scratch, sizeof(scratch))) {
                slot_capture(&vae, &slots[1]);
            } else {
                slot_capture(&vae, &slots[1]);
                slot_free(&slots[1]);
                pipeline = 0;
            }
            slot_bind(&vae, &slots[0]);
        }
    }
    int tile_count = y_axis.count * x_axis.count;
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = ok ? malloc(output_elements * sizeof(*final_rgb)) : NULL;
    float *temporal_overlap = ok ? malloc(5 * frame_elements *
                                          sizeof(*temporal_overlap)) : NULL;
    if (ok && (!final_rgb || !temporal_overlap)) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        ok = 0;
    }
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
        if (!tiles) {
            fail(error, error_size, "out of memory retaining video VAE tiles");
            ok = 0;
            break;
        }
        for (int index = 0; index < tile_count; index++) tiles[index] = NULL;
        ok = decode_chunk_tiles(&vae, slots, pipeline, &y_axis, &x_axis,
                                normalized_latent, latent_time, chunk,
                                latent_height, latent_width,
                                latent_mean, latent_std, -1, tiles,
                                NULL, error, error_size);
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        if (ok) ok = stitch_tiles(tiles, &y_axis, &x_axis,
                                  FIRST_CHUNK_FRAMES, &decoded,
                                  error, error_size);
        for (int index = 0; index < tile_count; index++) free(tiles[index]);
        free(tiles);
        if (!ok) {
            h3_video_frames_free(&decoded);
            break;
        }
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (slots[0].hidden) slot_bind(&vae, &slots[0]);
    slot_free(&slots[1]);
    cleanup(&vae);
    tile_axis_free(&y_axis); tile_axis_free(&x_axis);
    return ok;
}

void h3_video_frames_free(h3_video_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_time < 2 ||
        (latent_time != 2 && (latent_time < CHUNK_LATENT_TIME ||
                              (latent_time - 2) % 5)) ||
        latent_height < 1 || latent_width < 1 ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX ||
        (uint32_t)(latent_height * latent_width) >
            (UINT32_MAX - SUFFIX) / CHUNK_LATENT_TIME) {
        fail(error, error_size, "invalid first-chunk video VAE arguments");
        return 0;
    }
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.latent_t = latent_time;
    vae.output_frames = latent_time == 2 ? 5 : FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * latent_height * latent_width);
    vae.sequence = vae.patches + SUFFIX;
    float latent_mean[LATENT_CHANNELS], latent_std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, latent_mean, latent_std,
                                   error, error_size)) return 0;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    if (latent_time == 2 &&
        (latent_height > TILE_PIXELS / SPATIAL_RATIO ||
         latent_width > TILE_PIXELS / SPATIAL_RATIO)) {
        fail(error, error_size,
             "the two-token diagnostic VAE path supports one spatial tile");
        return 0;
    }
    if (latent_time > CHUNK_LATENT_TIME ||
        latent_height > tile_pixels / SPATIAL_RATIO ||
        latent_width > tile_pixels / SPATIAL_RATIO) {
        int ok = decode_chunked(weight_directory, shader_source_path,
                                normalized_latent, latent_time, latent_height,
                                latent_width, latent_mean, latent_std,
                                tile_pixels, progress,
                                progress_opaque, output, error, error_size);
        if (!ok) h3_video_frames_free(output);
        return ok;
    }
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) return 0;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    int ok = vae.gpu &&
        load_input_weights(&vae, error, error_size) &&
        prepare_input(&vae, normalized_latent, latent_mean, latent_std,
                      error, error_size) &&
        prepare_rope(&vae, error, error_size) &&
        allocate_activations(&vae, error, error_size) &&
        run_decoder(&vae, progress, progress_opaque, error, error_size) &&
        unpack_frames(&vae, 0, output, error, error_size);
    if (!ok) h3_video_frames_free(output);
    cleanup(&vae);
    return ok;
}
