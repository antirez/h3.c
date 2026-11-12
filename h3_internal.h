#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"
#include "h3_host.h"

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
} h3_image_hooks;

h3_result *h3_generate_image_with_hooks(h3_ctx *ctx, const char *prompt,
                                        const h3_image_params *params,
                                        h3_image_hooks *hooks);

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
