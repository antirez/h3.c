/* Public API for the h3-metal MiniMax-H3 inference engine. */
#ifndef H3_H
#define H3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VERSION "0.1.0-dev"
#define H3_DEFAULT_WIDTH 864
#define H3_DEFAULT_HEIGHT 480
#define H3_DEFAULT_FRAMES 56
#define H3_DEFAULT_STEPS 20
#define H3_DEFAULT_DIT_LAYERS 50
#define H3_MIN_DIT_LAYERS 35
#define H3_IMAGE_DEFAULT_SIZE 1024
#define H3_IMAGE_DEFAULT_MIN_AREA 4194304

typedef struct h3_ctx h3_ctx;
typedef struct h3_result h3_result;

typedef struct {
    size_t embedding_entries;
    size_t embedding_bytes;
    int prepared_dit;
    int video_decoder;
} h3_cache_info;

typedef enum {
    H3_REFERENCE_IMAGE = 1,
    H3_REFERENCE_VIDEO = 2,
    H3_REFERENCE_AUDIO = 3,
    H3_REFERENCE_VIDEO_AUDIO = 4
} h3_reference_kind;

typedef struct {
    h3_reference_kind kind;
    const char *path;
    const char *audio_path;
    int include_embedded_audio;
} h3_reference;

typedef enum {
    H3_REFERENCE_IMAGE_MATCH = 0,
    H3_REFERENCE_IMAGE_MAX = 1
} h3_reference_image_size;

typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    /* Non-negative only for an intermediate denoising preview. */
    int denoise_step;
    int denoise_steps;
} h3_frame;

typedef int (*h3_frame_callback)(const h3_frame *frame, void *opaque);
typedef int (*h3_progress_callback)(const char *phase, int completed, int total,
                                    void *opaque);

/* DiT tuning flags shared by the video and image requests, all 0 by default:
 * alternative weight paths and slower, closer-to-reference kernels. The DiT
 * loaders take the struct by pointer. */
typedef struct {
    /* Use one int8 activation scale per FC2 row and the M5 full-K kernel.
     * Faster, but more numerically aggressive than grouped int8. */
    int use_int8_row_fc2;
    /* Keep only two original BF16 DiT blocks in memory and overlap reading the
     * next block from the checkpoint with execution of the current block. */
    int ssd_streaming;
    /* Force the portable close-reference BF16/MPS MLP implementation instead
     * of the fastest validated native MLP supported by the current GPU. */
    int use_slower_bf16_mlp;
    /* Force the portable close-reference BF16 QKV projection. */
    int use_slower_bf16_qkv;
    /* Force the portable BF16 attention-output projection. */
    int use_slower_bf16_attention_output;
    /* Materialize row-major BF16 after SDPA before int8 quantization. */
    int use_slower_row_major_attention_output;
    /* Keep int8 projection-input quantization as standalone kernels. */
    int use_slower_unfused_int8_inputs;
    /* Keep Q/K norm and RoPE as a separate kernel after int8 QKV. */
    int use_slower_unfused_qkv_rope;
    /* Force scalar BF16 loads in the fused Q/K RMS reducer. */
    int use_slower_scalar_qkv_rms;
    /* Reread int8 dequantization scales from device memory per output. */
    int use_slower_uncached_int8_scales;
    /* Use the generic runtime-bound FC1 TensorOps K loop. */
    int use_slower_dynamic_fc1_k;
    /* Force the original 256-thread FC2 grouped activation quantizer. */
    int use_slower_grouped_quantizer;
} h3_dit_tuning;

typedef struct {
    int width;
    int height;
    int frames;
    int steps;
    uint64_t seed;
    const char *output_path;
    const char *first_frame;
    const char *last_frame;
    const h3_reference *references;
    size_t reference_count;
    h3_reference_image_size reference_image_size;
    /* Evaluate one of every N denoiser steps. 1 is the close-reference path,
     * 2 is the validated fast path, and 3 is the aggressive fast path. */
    int denoise_reuse;
    /* Number of gate-ranked DiT residual blocks to retain. 50 is exact,
     * 45 is the validated fast setting, and 40 is more aggressive. */
    int dit_layers;
    /* Recompute the transformer core every N denoiser steps while refreshing
     * the timestep head each step. 1 is exact, 4 fast, and 6 aggressive. */
    int core_reuse;
    /* Pair adjacent horizontal video tokens through middle DiT blocks while
     * preserving their full-resolution residual. Early noisy evaluations use
     * a deeper reduced interval. This is a validated aggressive speed mode. */
    int token_reduction;
    /* Restore the released spatial RoPE grid at 256x256. The default applies
     * a visually validated half-scale grid only at that native canvas. */
    int use_reference_rope;
    /* Optional lower internal model canvas. Both must be zero (exact output
     * canvas) or valid same-aspect dimensions no larger than width/height. */
    int render_width;
    int render_height;
    h3_dit_tuning tuning;
    /* Decode and deliver one representative frame after every Euler step. */
    int preview_denoise;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
} h3_params;

#define H3_PARAMS_DEFAULT { \
    .width = H3_DEFAULT_WIDTH, .height = H3_DEFAULT_HEIGHT, \
    .frames = H3_DEFAULT_FRAMES, .steps = H3_DEFAULT_STEPS, \
    .seed = UINT64_C(42), .reference_image_size = H3_REFERENCE_IMAGE_MATCH, \
    .denoise_reuse = 1, .dit_layers = H3_DEFAULT_DIT_LAYERS, .core_reuse = 1 \
}

#define H3_IMAGE_MAX_REFERENCES 8

/* One edit reference: an image file FFmpeg can read. The prompt names it
 * <Picture N>, N counting from 1 in request order. */
typedef struct {
    const char *path;
} h3_image_reference;

/* Text-to-image request. width and height are the size of the returned image,
 * any positive size; the model runs on a larger canvas when the area is below
 * min_generation_area and the result is resized back. The progress callback
 * reports each phase and each of the six denoising steps; on_frame receives
 * the final image once (frame_index 0, frame_count 1).
 *
 * With reference_count > 0 the request edits: the references (at most
 * H3_IMAGE_MAX_REFERENCES, each with both sides at least 32 pixels) condition
 * the image, and width and height may both be 0 to return the size of the
 * first reference. reference_count 0 is text-to-image. */
typedef struct {
    int width;
    int height;
    int min_generation_area;
    uint64_t seed;
    const char *output_path;
    /* Must be 50: the image transformer has no thinned variants. */
    int dit_layers;
    /* Weight path. The default is resident BF16 (about 36 GiB of GPU buffers,
     * fast forwards). use_int8 keeps the MLP, QKV and attention output
     * resident as int8 on a Metal 4 GPU (about 18.6 GiB, about twice the
     * per-step error). ssd_streaming runs exact BF16 from the checkpoint with
     * about 1.5 GiB of GPU buffers and is slower. use_int8 and ssd_streaming
     * exclude each other. */
    int use_int8;
    h3_dit_tuning tuning;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
    const h3_image_reference *references;
    size_t reference_count;
} h3_image_params;

#define H3_IMAGE_PARAMS_DEFAULT { \
    .width = H3_IMAGE_DEFAULT_SIZE, .height = H3_IMAGE_DEFAULT_SIZE, \
    .min_generation_area = H3_IMAGE_DEFAULT_MIN_AREA, .seed = UINT64_C(42), \
    .dit_layers = H3_DEFAULT_DIT_LAYERS \
}

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
} h3_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} h3_component_info;

typedef struct {
    h3_component_info text_encoder;
    h3_component_info fl2va_transformer;
    h3_component_info ref2va_transformer;
    h3_component_info video_vae;
    h3_component_info audio_vae;
    /* MiniMax-H3-Image checkpoint: the text encoder above is its own
     * text_encoder directory. */
    h3_component_info image_transformer;
    h3_component_info image_vae;
} h3_model_info;

struct h3_result {
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
};

/* Load model metadata and initialize the Metal device. Weights remain unmapped. */
h3_ctx *h3_load_dir(const char *model_dir);
void h3_free(h3_ctx *ctx);

/* Nonzero when the directory is a MiniMax-H3-Image checkpoint, which
 * generates with h3_generate_image() only. */
int h3_is_image_model(const h3_ctx *ctx);

const char *h3_last_error(const h3_ctx *ctx);
const h3_device_info *h3_device(const h3_ctx *ctx);
const h3_model_info *h3_model(const h3_ctx *ctx);

/* Interactive-session reuse. Disabled by default so one-shot callers retain
 * the original phase-by-phase memory lifetime. */
void h3_cache_set_enabled(h3_ctx *ctx, int enabled);
void h3_cache_clear(h3_ctx *ctx);
void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info);
/* Image runs: keep the text embedding of a prompt (and its references) in
 * files under dir, and reuse it in later runs, even in other processes. NULL
 * disables it, which is the default. H3_CACHE_BYTES caps the directory. */
void h3_cache_set_directory(h3_ctx *ctx, const char *dir);

/* Generate media, delivering decoded frames incrementally through on_frame. */
h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params);
/* Text-to-image: tokenize, encode, six denoising steps, VAE decode, resize and
 * PNG. Each model is loaded, used and freed in turn. The result has frames = 1
 * and fps = 0. */
h3_result *h3_generate_image(h3_ctx *ctx, const char *prompt,
                             const h3_image_params *params);
void h3_result_free(h3_result *result);

#ifdef __cplusplus
}
#endif
#endif
