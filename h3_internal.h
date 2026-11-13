#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stdarg.h>

struct h3_dit;
struct h3_video_vae_decoder;

struct h3_ctx {
    char *model_dir;
    char error[512];
    h3_device_info device;
    h3_model_info model;
    int image;
    int cache_enabled;
    char *cache_dir;
    char *conditioning_key;
    size_t conditioning_tokens;
    size_t conditioning_width;
    uint16_t *conditioning_values;
    uint8_t *conditioning_tags;
    float *conditioning_video_rows;
    size_t conditioning_video_elements;
    float *conditioning_audio_rows;
    size_t conditioning_audio_elements;
    h3_layout_ref *conditioning_references;
    size_t conditioning_reference_count;
    int conditioning_present;
    char *dit_key;
    struct h3_dit *dit;
    char *video_decoder_key;
    struct h3_video_vae_decoder *video_decoder;
};

#define H3_CACHE_NOT_USED 0
#define H3_CACHE_MISS 1
#define H3_CACHE_HIT 2

/* Test hooks of h3_generate_image(). */
typedef struct {
    /* Initial latent noise [24][latent_height][latent_width] instead of the
     * seed's. */
    const float *noise;
    /* Receive malloc'd copies; the caller frees them. latent is the final
     * normalised latent [24][latent_height][latent_width], rgb the returned
     * image, interleaved [height][width][3]. */
    float *latent;
    uint8_t *rgb;
    /* The weight path that ran: "stream", "bf16", "int8" or "mixed". */
    const char *weight_mode;
    /* Edits, one entry per reference (NULL array: the real path runs). The
     * posterior noise is [24][crop_h / 16][crop_w / 16] standard normal, the
     * shape the encoder's posterior has before the even-side crop; the
     * augmentation noise is [24][latent_h][latent_w]. */
    const float *const *reference_posterior_noise;
    const float *const *reference_aug_noise;
    /* Planar bytes [3][h][w] used instead of the pixels h3.c resizes itself:
     * the VAE copy (vae_w x vae_h) and the image the Qwen vision tower
     * receives (vision_w x vision_h of the reference's geometry): the vision
     * copy after smart_resize, not the first-stage image. The sizes are not
     * checked here; the caller must match the geometry. */
    const uint8_t *const *vae_pixels;
    const uint8_t *const *vision_pixels;
    /* Edit outputs, malloc'd, the caller frees them: the noised, patchified
     * condition rows [condition_elements / 96][96] and the text presentation
     * length. */
    float *condition_rows;
    size_t condition_elements;
    size_t text_tokens;
    /* Edits: stop after the reference VAE encoder. No text encoder or
     * transformer loads, nothing is generated, condition_rows is set. */
    int references_only;
    /* When the disk cache holds the text embedding, stop right after the
     * lookup and return it in text (the caller frees it with
     * h3_text_embedding_free()). After a miss the run goes on. */
    int stop_after_lookup;
    h3_text_embedding text;
    /* Output: what the text cache lookup did. */
    int cache_result;
} h3_image_hooks;

h3_result *h3_generate_image_with_hooks(h3_ctx *ctx, const char *prompt,
                                        const h3_image_params *params,
                                        h3_image_hooks *hooks);

/* Generation progress and helpers that h3.c shares with h3_image_generate.c. */
typedef struct {
    h3_ctx *ctx;
    h3_progress_callback callback;
    void *opaque;
    const h3_params *params;
    int cancelled;
} h3_generation_progress;

void h3_progress_emit(h3_generation_progress *state, const char *phase,
                      int completed, int total);
void h3_text_progress_bridge(int completed, int total, void *opaque);
void h3_dit_progress_bridge(const char *phase, int completed, int total,
                            void *opaque);
void h3_vision_progress_bridge(int completed, int total, void *opaque);
/* "root/relative", malloc'd (the caller frees it); NULL when out of memory. */
char *h3_path(const char *root, const char *relative);

/* The text cache key of a request, composed from ctx (h3_image_generate.c). */
char *h3_image_text_cache_key(const h3_ctx *ctx, const char *prompt,
                              size_t count, const int *widths,
                              const int *heights,
                              const uint8_t *const *sources,
                              const int *vision_widths,
                              const int *vision_heights);

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
