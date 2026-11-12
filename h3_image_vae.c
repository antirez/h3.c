#include "h3_image_vae.h"

#include "h3_image.h"
#include "h3_vae_tiles.h"
#include "h3_weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = 24,
    LAYERS = 24,
    HEADS = 16,
    HEAD_DIM = 64,
    HIDDEN = HEADS * HEAD_DIM,
    FFN = 4096,
    REGISTERS = 4,
    SUFFIX = REGISTERS + 1,
    ROPE_AXES = 2,
    ROPE_HALF = ROPE_AXES * H3_IMAGE_VAE_ROPE_FREQS,
    PATCH = H3_VAE_SPATIAL_RATIO,
    OUTPUT_PATCH = 3 * PATCH * PATCH,
    TILE_PIXELS = 256,
    MAX_LATENT = 1024
};

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
    h3_gpu_tensor *zero;
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
    int latent_h;
    int latent_w;
} vae_context;

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

/* Read a whole F32 tensor into host memory, to rearrange it before upload. */
static int read_host(vae_context *vae, const char *name, size_t elements,
                     float *destination, char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(vae->weights, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != elements) {
        fail(error, error_size, "weight %s is not %zu F32 elements", name,
             elements);
        return 0;
    }
    return h3_st_read_data(header, tensor, destination,
                           elements * sizeof(*destination), error, error_size);
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
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->zero);
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
    free_tensor(&vae->latent); free_tensor(&vae->post);
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

/* to_q, to_k and to_v become one matrix in the [head][q,k,v][64] row order of
 * the grouped QKV and RoPE kernel; the biases follow the same map. */
static int load_qkv(vae_context *vae, const char *prefix, vae_block *block,
                    char *error, size_t error_size) {
    static const char projection_name[3] = {'q', 'k', 'v'};
    size_t matrix = (size_t)HIDDEN * HIDDEN;
    float *weights = malloc(3 * matrix * sizeof(*weights));
    float *biases = malloc(3 * HIDDEN * sizeof(*biases));
    float *source_w = malloc(3 * matrix * sizeof(*source_w));
    float *source_b = malloc(3 * HIDDEN * sizeof(*source_b));
    int ok = weights && biases && source_w && source_b;
    if (!ok) fail(error, error_size, "out of memory reading image VAE QKV");
    char name[160];
    for (int projection = 0; ok && projection < 3; projection++) {
        snprintf(name, sizeof(name), "%sattn.to_%c.weight", prefix,
                 projection_name[projection]);
        ok = read_host(vae, name, matrix, source_w + projection * matrix,
                       error, error_size);
        if (!ok) break;
        snprintf(name, sizeof(name), "%sattn.to_%c.bias", prefix,
                 projection_name[projection]);
        ok = read_host(vae, name, HIDDEN, source_b + projection * HIDDEN,
                       error, error_size);
    }
    for (size_t row = 0; ok && row < 3 * (size_t)HIDDEN; row++) {
        unsigned projection;
        size_t source;
        ok = h3_image_qkv_source(row, HEADS, HEAD_DIM, &projection, &source);
        if (!ok) break;
        memcpy(weights + row * HIDDEN,
               source_w + projection * matrix + source * HIDDEN,
               HIDDEN * sizeof(*weights));
        biases[row] = source_b[projection * HIDDEN + source];
    }
    if (ok) {
        block->qkv_w = h3_gpu_tensor_from_f32(vae->gpu, weights, 3 * matrix);
        block->qkv_b = h3_gpu_tensor_from_f32(vae->gpu, biases, 3 * HIDDEN);
        ok = block->qkv_w && block->qkv_b;
        if (!ok) fail(error, error_size, "cannot allocate image VAE QKV: %s",
                      h3_gpu_error(vae->gpu));
    }
    free(weights); free(biases); free(source_w); free(source_b);
    return ok;
}

/* diffusers SwiGLU is chunk(2) = (value, gate) = value * silu(gate); the GPU
 * kernel reads (gate, value). Swap the two halves of the rows on load. */
static int load_ff1(vae_context *vae, const char *prefix, vae_block *block,
                    char *error, size_t error_size) {
    size_t half = (size_t)FFN * HIDDEN;
    float *weights = malloc(2 * half * sizeof(*weights));
    float *biases = malloc(2 * FFN * sizeof(*biases));
    int ok = weights && biases;
    if (!ok) fail(error, error_size, "out of memory reading image VAE MLP");
    char name[160];
    if (ok) {
        snprintf(name, sizeof(name), "%sff.net.0.proj.weight", prefix);
        ok = read_host(vae, name, 2 * half, weights, error, error_size);
    }
    if (ok) {
        snprintf(name, sizeof(name), "%sff.net.0.proj.bias", prefix);
        ok = read_host(vae, name, 2 * FFN, biases, error, error_size);
    }
    if (ok) {
        for (size_t index = 0; index < half; index++) {
            float value = weights[index];
            weights[index] = weights[half + index];
            weights[half + index] = value;
        }
        for (size_t index = 0; index < FFN; index++) {
            float value = biases[index];
            biases[index] = biases[FFN + index];
            biases[FFN + index] = value;
        }
        block->w1 = h3_gpu_tensor_from_f32(vae->gpu, weights, 2 * half);
        block->w1_b = h3_gpu_tensor_from_f32(vae->gpu, biases, 2 * FFN);
        ok = block->w1 && block->w1_b;
        if (!ok) fail(error, error_size, "cannot allocate image VAE MLP: %s",
                      h3_gpu_error(vae->gpu));
    }
    free(weights); free(biases);
    return ok;
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                       \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F2(out_w, "attn.to_out.0.weight", HIDDEN, HIDDEN);
    F1(out_b, "attn.to_out.0.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F2(w2, "ff.net.2.weight", HIDDEN, FFN);
    F1(w2_b, "ff.net.2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
#undef F1
#undef F2
    return load_qkv(vae, prefix, block, error, error_size) &&
           load_ff1(vae, prefix, block, error, error_size);
}

static int load_weights(vae_context *vae, char *error, size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1};
    vae->post_w = load_f32(vae, "post_quant_conv.weight", 4, post_shape,
                           error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.proj_in.weight", HIDDEN, LATENT_CHANNELS,
                      error, error_size);
    vae->embed_b = f1(vae, "decoder.proj_in.bias", HIDDEN, error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_f32(vae, "decoder.register_tokens", 3,
                              register_shape, error, error_size);
    if (!vae->post_w || !vae->post_b || !vae->embed_w || !vae->embed_b ||
        !vae->registers) return 0;
    for (int index = 0; index < LAYERS; index++)
        if (!load_block(vae, index, error, error_size)) return 0;
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

int h3_image_vae_parse_float_array(const char *json, const char *key,
                                   float *values, size_t count, char *error,
                                   size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "image VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "image VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (*cursor != (index + 1 < count ? ',' : ']')) {
            fail(error, error_size, "image VAE config has wrong length %s",
                 key);
            return 0;
        }
        cursor++;
    }
    return 1;
}

int h3_image_vae_load_latent_normalization(const char *weight_directory,
                                           float *mean, float *deviation,
                                           char *error, size_t error_size) {
    char path[1024];
    int length = snprintf(path, sizeof(path), "%s/config.json",
                          weight_directory);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fail(error, error_size, "image VAE weight directory path is too long");
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open image VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        return 0;
    }
    long end = ftell(file);
    char *json = end > 0 && end <= 1024 * 1024 && !fseek(file, 0, SEEK_SET) ?
                 malloc((size_t)end + 1) : NULL;
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read image VAE config %s", path);
        free(json);
        fclose(file);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    int ok = h3_image_vae_parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             h3_image_vae_parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    for (int channel = 0; ok && channel < LATENT_CHANNELS; channel++)
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size,
                 "image VAE latent standard deviation is invalid");
            return 0;
        }
    return ok;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    size_t count = (size_t)vae->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating image VAE RoPE");
        return 0;
    }
    float inverse[H3_IMAGE_VAE_ROPE_FREQS];
    h3_image_vae_inv_freq(inverse);
    uint32_t row = 0;
    for (int h = 0; h < vae->latent_h; h++)
        for (int w = 0; w < vae->latent_w; w++, row++) {
            float axes[ROPE_AXES] = {
                2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
            };
            for (int axis = 0; axis < ROPE_AXES; axis++)
                for (int frequency = 0; frequency < H3_IMAGE_VAE_ROPE_FREQS;
                     frequency++) {
                    float angle = 2.0f * 3.14159265358979323846f *
                                  axes[axis] * inverse[frequency];
                    size_t offset = (size_t)row * ROPE_HALF +
                        (size_t)(axis * H3_IMAGE_VAE_ROPE_FREQS + frequency);
                    cosines[offset] = cosf(angle);
                    sines[offset] = sinf(angle);
                }
        }
    for (; row < vae->sequence; row++)
        for (size_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    free(cosines); free(sines);
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate image VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    size_t patches = vae->patches, sequence = vae->sequence;
    float zeros[HIDDEN] = {0};
    vae->zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        vae->zero,
        F32(latent, patches * LATENT_CHANNELS),
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * HIDDEN),
        F32(hidden, sequence * HIDDEN),
        F32(norm, sequence * HIDDEN),
        F32(qkv, sequence * HIDDEN * 3),
        F32(query, sequence * HIDDEN), F32(key, sequence * HIDDEN),
        F32(value, sequence * HIDDEN), F32(heads, sequence * HIDDEN),
        F32(branch, sequence * HIDDEN), F32(ff1, sequence * FFN * 2),
        F32(activated, sequence * FFN),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate image VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    return 1;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "image VAE attention norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->qkv, vae->norm, weight->qkv_w,
        weight->qkv_b, rows, HIDDEN, HIDDEN * 3), "image VAE QKV");
    OP(h3_gpu_video_qkv_rope_f32(vae->gpu, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "image VAE QK norm/RoPE");
    OP(h3_gpu_sdpa_f32(vae->gpu, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "image VAE attention");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->heads, weight->out_w,
        weight->out_b, rows, HIDDEN, HIDDEN), "image VAE attention output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "image VAE attention residual");
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm2,
        rows, HIDDEN, 1e-5f), "image VAE MLP norm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->ff1, vae->norm, weight->w1,
        weight->w1_b, rows, HIDDEN, FFN * 2), "image VAE MLP input");
    OP(h3_gpu_swiglu_f32(vae->gpu, vae->activated, vae->ff1, rows, FFN),
       "image VAE SwiGLU");
    OP(h3_gpu_linear_f32(vae->gpu, vae->branch, vae->activated, weight->w2,
        weight->w2_b, rows, FFN, HIDDEN), "image VAE MLP output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "image VAE MLP residual");
#undef OP
    return 1;
}

static int run_tile(vae_context *vae, char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin image VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "image VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "image VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack image VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, (size_t)vae->patches * HIDDEN,
        vae->registers, 0, REGISTERS * HIDDEN), "pack image VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, vae->zero, 0, HIDDEN),
       "pack image VAE zero token");
    for (int index = 0; index < LAYERS; index++)
        if (!run_block(vae, index, error, error_size)) return 0;
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "image VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "image VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit image VAE decoder");
#undef OP
    return 1;
}

void h3_image_vae_unpack_patches(const float *rows, int latent_height,
                                 int latent_width, float *rgb) {
    int pixel_w = latent_width * PATCH;
    for (int y = 0; y < latent_height * PATCH; y++)
        for (int x = 0; x < pixel_w; x++) {
            size_t patch = (size_t)(y / PATCH) * (size_t)latent_width +
                           (size_t)(x / PATCH);
            for (int channel = 0; channel < 3; channel++) {
                size_t component = (size_t)((channel * PATCH + y % PATCH) *
                                            PATCH + x % PATCH);
                rgb[((size_t)y * (size_t)pixel_w + (size_t)x) * 3 +
                    (size_t)channel] = rows[patch * OUTPUT_PATCH + component];
            }
        }
}

/* Normalised latents of one tile, channel-last rows with mean/std restored. */
static int upload_tile_latent(vae_context *vae, const float *latent,
                              int full_h, int full_w, int start_y, int start_x,
                              const float *mean, const float *deviation,
                              char *error, size_t error_size) {
    float *rows = malloc((size_t)vae->patches * LATENT_CHANNELS * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory preparing image VAE tile");
        return 0;
    }
    size_t row = 0;
    for (int y = 0; y < vae->latent_h; y++)
        for (int x = 0; x < vae->latent_w; x++)
            for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                size_t source = ((size_t)channel * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)(start_x + x);
                rows[row++] = latent[source] * deviation[channel] +
                              mean[channel];
            }
    int ok = h3_gpu_tensor_write_f32(vae->latent, rows, row);
    free(rows);
    if (!ok) fail(error, error_size, "cannot upload image VAE tile: %s",
                  h3_gpu_error(vae->gpu));
    return ok;
}

static void free_tiles(float **tiles, int count) {
    if (!tiles) return;
    for (int index = 0; index < count; index++) free(tiles[index]);
    free(tiles);
}

int h3_image_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent,
                        int latent_height, int latent_width,
                        h3_image_vae_progress progress, void *progress_opaque,
                        h3_image_planes *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !normalized_latent || !output ||
        latent_height < 1 || latent_width < 1) {
        fail(error, error_size, "invalid image VAE decode arguments");
        return 0;
    }
    if (latent_height > MAX_LATENT || latent_width > MAX_LATENT) {
        fail(error, error_size,
             "image VAE latent %dx%d exceeds the %d x %d limit",
             latent_height, latent_width, MAX_LATENT, MAX_LATENT);
        return 0;
    }
    float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
    if (!h3_image_vae_load_latent_normalization(weight_directory, mean,
                                                deviation, error, error_size))
        return 0;
    h3_vae_tile_axis y_axis, x_axis;
    if (!h3_vae_tile_axis_build(latent_height * PATCH, TILE_PIXELS, &y_axis,
                                error, error_size)) return 0;
    if (!h3_vae_tile_axis_build(latent_width * PATCH, TILE_PIXELS, &x_axis,
                                error, error_size)) {
        h3_vae_tile_axis_free(&y_axis);
        return 0;
    }
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = y_axis.length / PATCH;
    vae.latent_w = x_axis.length / PATCH;
    vae.patches = (uint32_t)(vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    int tile_count = y_axis.count * x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    int ok = tiles != NULL;
    if (!ok) fail(error, error_size, "out of memory retaining image VAE tiles");
    if (ok) {
        vae.weights = h3_weight_store_open(weight_directory, error, error_size);
        ok = vae.weights != NULL;
    }
    if (ok) {
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
        ok = vae.gpu != NULL;
    }
    ok = ok && load_weights(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size) &&
         prepare_rope(&vae, error, error_size);
    float *rows = NULL;
    if (ok) {
        rows = malloc((size_t)vae.patches * OUTPUT_PATCH * sizeof(*rows));
        ok = rows != NULL;
        if (!ok) fail(error, error_size, "out of memory reading image VAE tile");
    }
    for (int tile_y = 0; ok && tile_y < y_axis.count; tile_y++)
        for (int tile_x = 0; ok && tile_x < x_axis.count; tile_x++) {
            int index = tile_y * x_axis.count + tile_x;
            size_t pixels = (size_t)y_axis.length * (size_t)x_axis.length;
            tiles[index] = malloc(pixels * 3 * sizeof(float));
            ok = tiles[index] != NULL;
            if (!ok) {
                fail(error, error_size, "out of memory retaining image VAE tiles");
                break;
            }
            ok = upload_tile_latent(&vae, normalized_latent, latent_height,
                    latent_width, y_axis.starts[tile_y] / PATCH,
                    x_axis.starts[tile_x] / PATCH, mean, deviation,
                    error, error_size) &&
                 run_tile(&vae, error, error_size) &&
                 h3_gpu_tensor_read_f32(vae.projected, rows,
                                        (size_t)vae.patches * OUTPUT_PATCH);
            if (ok) {
                h3_image_vae_unpack_patches(rows, vae.latent_h, vae.latent_w,
                                            tiles[index]);
                if (progress) progress(index + 1, tile_count, progress_opaque);
            } else if (error && !error[0]) {
                fail(error, error_size, "cannot read image VAE tile");
            }
        }
    free(rows);
    h3_video_frames stitched;
    memset(&stitched, 0, sizeof(stitched));
    ok = ok && h3_vae_stitch_tiles(tiles, &y_axis, &x_axis, 1, &stitched,
                                   error, error_size);
    free_tiles(tiles, tile_count);
    if (ok) {
        size_t plane = (size_t)stitched.height * (size_t)stitched.width;
        output->data = malloc(plane * 3 * sizeof(*output->data));
        if (!output->data) {
            fail(error, error_size, "out of memory assembling the image");
            ok = 0;
        } else if (!h3_gpu_get_stats(vae.gpu, &output->gpu_stats)) {
            fail(error, error_size, "cannot read image VAE GPU statistics: %s",
                 h3_gpu_error(vae.gpu));
            ok = 0;
        } else {
            output->height = stitched.height;
            output->width = stitched.width;
            for (size_t pixel = 0; pixel < plane; pixel++)
                for (size_t channel = 0; channel < 3; channel++)
                    output->data[channel * plane + pixel] =
                        stitched.rgb[pixel * 3 + channel];
        }
    }
    free(stitched.rgb);
    h3_vae_tile_axis_free(&y_axis);
    h3_vae_tile_axis_free(&x_axis);
    cleanup(&vae);
    if (!ok) h3_image_planes_free(output);
    return ok;
}

void h3_image_planes_free(h3_image_planes *planes) {
    if (!planes) return;
    free(planes->data);
    memset(planes, 0, sizeof(*planes));
}

void h3_image_planes_to_rgb(h3_image_planes *planes) {
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    size_t plane = (size_t)planes->height * (size_t)planes->width;
    for (size_t channel = 0; channel < 3; channel++)
        for (size_t index = 0; index < plane; index++) {
            float *value = &planes->data[channel * plane + index];
            *value = *value * deviation[channel] + mean[channel];
            if (*value < 0.0f) *value = 0.0f;
            if (*value > 1.0f) *value = 1.0f;
        }
}

void h3_image_planes_to_u8(const h3_image_planes *planes, uint8_t *bytes) {
    size_t count = (size_t)planes->height * (size_t)planes->width * 3;
    for (size_t index = 0; index < count; index++) {
        float scaled = planes->data[index] * 255.0f;
        bytes[index] = !(scaled > 0.0f) ? 0 :
                       scaled >= 255.0f ? 255 : (uint8_t)scaled;
    }
}
