/* Builds a pre-quantized int8 cache of every DiT weight matrix (QKV,
 * attention-output, FC1, FC2), so the runtime can either keep the MLP
 * portion resident (as before) or stream all four matrices for every
 * block - the latter is what a long (~15s/362-frame) run needs, since
 * even int8-resident MLP for all 50 blocks (~10.8 GiB) stops being cheap
 * once per-sequence activations grow with frame count. See h3_dit.c's
 * H3_ATTENTION_CACHE / H3_INT8_STREAM_MLP handling for the reader side.
 *
 * Cache layout: a 64-byte header, then per block (in block order):
 *   qkv_int8[INNER*3*HIDDEN]  qkv_scales[INNER*3]   (f32)
 *   out_int8[HIDDEN*INNER]    out_scales[HIDDEN]    (f32)
 *   fc1_int8[FFN*2*HIDDEN]    fc1_scales[FFN*2]     (f32)
 *   fc2_int8[HIDDEN*FFN]      fc2_scales[HIDDEN]    (f32)
 * Quantization uses the exact same GPU routine (h3_gpu_quantize_weight_int8)
 * as the existing resident-int8 path, so results match it bit for bit; they
 * are not expected to match a BF16-only run.
 *
 * The v3 header adds model_kind (FL2VA=1/Ref2VA=2, matching h3_dit.c's
 * h3_cache_model_kind) and model_id (h3_weight_store_fingerprint() of the
 * transformer directory quantized) so h3_dit.c can refuse a cache built
 * for the wrong model rather than silently streaming mismatched weights -
 * a v2 cache (no such tagging) simply fails h3_dit.c's version check and
 * must be rebuilt.
 *
 * Usage:
 *   build_attention_cache <FL2VA/transformer dir> <output cache file>
 *     - single-component mode, backward compatible with v2's usage; the
 *       transformer directory's own path (its last two components, e.g.
 *       ".../FL2VA/transformer") sets model_kind, same convention h3.c's
 *       own dit_path selection writes and h3_dit.c reads back.
 *   build_attention_cache <model root dir> <output cache directory>
 *     - detected when <model root dir>/FL2VA/transformer/config.json
 *       exists: writes <output dir>/fl2va.cache, and, if a Ref2VA
 *       transformer is present too, <output dir>/ref2va.cache - matching
 *       H3_ATTENTION_CACHE_DIR's auto-selection by generation mode.
 */
#include "h3_gpu.h"
#include "h3_weights.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    DIT_BLOCKS = 50,
};

#define CACHE_MAGIC "H3AC"
#define CACHE_VERSION 3u

typedef enum {
    MODEL_UNKNOWN = 0,
    MODEL_FL2VA = 1,
    MODEL_REF2VA = 2,
} model_kind;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t block_count;
    uint32_t hidden;
    uint32_t inner;
    uint32_t ffn;
    uint32_t model_kind;
    uint8_t model_id[32];
    uint32_t reserved[1];
} cache_header;

/* Mirrors h3_dit.c's detect_model_kind(): reads back the layout h3.c's
 * own dit_path selection always produces, rather than guessing at one. */
static model_kind detect_model_kind(const char *transformer_dir) {
    size_t length = strlen(transformer_dir);
    static const char ref2va_suffix[] = "Ref2VA/transformer";
    static const char fl2va_suffix[] = "FL2VA/transformer";
    if (length >= sizeof(ref2va_suffix) - 1 &&
        !strcmp(transformer_dir + length - (sizeof(ref2va_suffix) - 1),
                ref2va_suffix))
        return MODEL_REF2VA;
    if (length >= sizeof(fl2va_suffix) - 1 &&
        !strcmp(transformer_dir + length - (sizeof(fl2va_suffix) - 1),
                fl2va_suffix))
        return MODEL_FL2VA;
    return MODEL_UNKNOWN;
}

static int path_exists(const char *path) {
    struct stat status;
    return stat(path, &status) == 0;
}

static char *join_path(const char *base, const char *suffix) {
    size_t length = strlen(base) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (result) snprintf(result, length, "%s/%s", base, suffix);
    return result;
}

static int quantize_and_write(h3_gpu *gpu, h3_weight_store *store,
                              const char *name, uint32_t rows,
                              uint32_t columns, FILE *out,
                              char *error, size_t error_size) {
    uint64_t shape[2] = { rows, columns };
    h3_gpu_tensor *bf16 = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                              error, error_size);
    if (!bf16) return 0;

    size_t elements = (size_t)rows * columns;
    h3_gpu_tensor *i8 = h3_gpu_tensor_new_i8(gpu, elements);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
    int ok = i8 && scales &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(gpu, i8, scales, bf16, rows,
                                         columns) &&
             h3_gpu_submit(gpu);
    h3_gpu_tensor_free(bf16);
    if (!ok) {
        if (error && error_size && !error[0])
            snprintf(error, error_size, "cannot quantize %s: %s", name,
                     h3_gpu_error(gpu));
        h3_gpu_tensor_free(i8);
        h3_gpu_tensor_free(scales);
        return 0;
    }

    int8_t *i8_host = malloc(elements);
    float *scale_host = malloc((size_t)rows * sizeof(float));
    int result = i8_host && scale_host &&
        h3_gpu_tensor_read_i8(i8, i8_host, elements) &&
        h3_gpu_tensor_read_f32(scales, scale_host, rows) &&
        fwrite(i8_host, 1, elements, out) == elements &&
        fwrite(scale_host, sizeof(float), rows, out) == rows;
    free(i8_host);
    free(scale_host);
    h3_gpu_tensor_free(i8);
    h3_gpu_tensor_free(scales);
    if (!result && error && error_size)
        snprintf(error, error_size, "cannot write cache payload for %s: %s",
                 name, strerror(errno));
    return result;
}

/* Builds one complete attention cache from transformer_dir into
 * output_path. Returns 1 on success, 0 with a message already printed to
 * stderr on failure. */
static int build_one_cache(h3_gpu *gpu, const char *transformer_dir,
                           const char *output_path) {
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(transformer_dir, error,
                                                   sizeof(error));
    if (!store) {
        fprintf(stderr, "h3: %s\n", error);
        return 0;
    }

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "h3: cannot open %s: %s\n", output_path,
                strerror(errno));
        h3_weight_store_free(store);
        return 0;
    }

    cache_header header = {0};
    memcpy(header.magic, CACHE_MAGIC, 4);
    header.version = CACHE_VERSION;
    header.block_count = DIT_BLOCKS;
    header.hidden = HIDDEN;
    header.inner = INNER;
    header.ffn = FFN;
    header.model_kind = (uint32_t)detect_model_kind(transformer_dir);
    h3_weight_store_fingerprint(store, header.model_id);
    if (fwrite(&header, sizeof(header), 1, out) != 1) {
        fprintf(stderr, "h3: cannot write cache header: %s\n",
                strerror(errno));
        fclose(out);
        h3_weight_store_free(store);
        return 0;
    }

    for (uint32_t block = 0; block < DIT_BLOCKS; block++) {
        char name[160];
        struct { const char *suffix; uint32_t rows, columns; } projections[] = {
            {"attn.qkv_proj.weight", INNER * 3, HIDDEN},
            {"attn.out_proj.weight", HIDDEN, INNER},
            {"mlp.fc1.weight", FFN * 2, HIDDEN},
            {"mlp.fc2.weight", HIDDEN, FFN},
        };
        for (size_t p = 0; p < sizeof(projections) / sizeof(*projections); p++) {
            error[0] = '\0';
            snprintf(name, sizeof(name), "blocks.%u.%s", block,
                    projections[p].suffix);
            if (!quantize_and_write(gpu, store, name, projections[p].rows,
                                    projections[p].columns, out, error,
                                    sizeof(error))) {
                fprintf(stderr, "h3: %s\n", error);
                fclose(out);
                h3_weight_store_free(store);
                return 0;
            }
        }
        fprintf(stderr, "h3: attention cache block %2u/%u\n", block + 1,
                DIT_BLOCKS);
    }

    fclose(out);
    h3_weight_store_free(store);
    fprintf(stderr, "h3: wrote attention cache to %s\n", output_path);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <FL2VA/transformer dir> <output cache file>\n"
                "       %s <model root dir> <output cache directory>\n",
                argv[0], argv[0]);
        return 1;
    }
    char *fl2va_dir = join_path(argv[1], "FL2VA/transformer");
    char *fl2va_marker = fl2va_dir ? join_path(fl2va_dir, "config.json") : NULL;
    int model_root_mode = fl2va_marker && path_exists(fl2va_marker);
    free(fl2va_marker);

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "h3: %s\n", error);
        free(fl2va_dir);
        return 1;
    }
    if (!h3_gpu_has_int8_mlp(gpu)) {
        fprintf(stderr,
                "h3: this GPU lacks the int8 path the cache is built for\n");
        h3_gpu_free(gpu);
        free(fl2va_dir);
        return 1;
    }

    int ok;
    if (model_root_mode) {
        mkdir(argv[2], 0755); /* ignore EEXIST - a pre-existing dir is fine */
        char *fl2va_out = join_path(argv[2], "fl2va.cache");
        char *ref2va_dir = join_path(argv[1], "Ref2VA/transformer");
        char *ref2va_marker = ref2va_dir ? join_path(ref2va_dir, "config.json") : NULL;
        ok = fl2va_out && build_one_cache(gpu, fl2va_dir, fl2va_out);
        if (ok) {
            if (ref2va_marker && path_exists(ref2va_marker)) {
                char *ref2va_out = join_path(argv[2], "ref2va.cache");
                ok = ref2va_out && build_one_cache(gpu, ref2va_dir, ref2va_out);
                free(ref2va_out);
            } else {
                fprintf(stderr,
                        "h3: no Ref2VA transformer under %s - built "
                        "fl2va.cache only\n", argv[1]);
            }
        }
        free(fl2va_out);
        free(ref2va_dir);
        free(ref2va_marker);
    } else {
        ok = build_one_cache(gpu, argv[1], argv[2]);
    }

    h3_gpu_free(gpu);
    free(fl2va_dir);
    return ok ? 0 : 1;
}
