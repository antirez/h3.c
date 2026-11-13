#include "h3_internal.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_gpu.h"
#include "h3_image.h"
#include "h3_image_cache.h"
#include "h3_image_edit.h"
#include "h3_image_pipeline.h"
#include "h3_image_vae.h"
#include "h3_image_vae_encode.h"
#include "h3_multimodal.h"
#include "h3_text_build_id.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int h3_valid_image_params(h3_ctx *ctx, const h3_image_params *params) {
    if (!params) {
        h3_set_error(ctx, "generation parameters are required");
        return 0;
    }
    if (params->reference_count > H3_IMAGE_MAX_REFERENCES) {
        h3_set_error(ctx, "at most %d reference images are supported, got %zu",
                     H3_IMAGE_MAX_REFERENCES, params->reference_count);
        return 0;
    }
    if (params->reference_count && !params->references) {
        h3_set_error(ctx, "reference images are required");
        return 0;
    }
    for (size_t index = 0; index < params->reference_count; index++)
        if (!params->references[index].path || !*params->references[index].path) {
            h3_set_error(ctx, "reference %zu has an empty path", index + 1);
            return 0;
        }
    int size_from_reference = params->reference_count &&
                              params->width == 0 && params->height == 0;
    if (!size_from_reference &&
        (params->width < 1 || params->height < 1 ||
         params->width > H3_IMAGE_MAX_SIDE ||
         params->height > H3_IMAGE_MAX_SIDE)) {
        h3_set_error(ctx, "image width and height must be in [1, %d]",
                     H3_IMAGE_MAX_SIDE);
        return 0;
    }
    if (params->min_generation_area < H3_IMAGE_MIN_AREA_LOW ||
        params->min_generation_area > H3_IMAGE_MIN_AREA_HIGH) {
        h3_set_error(ctx, "min generation area must be in [%d, %d]",
                     H3_IMAGE_MIN_AREA_LOW, H3_IMAGE_MIN_AREA_HIGH);
        return 0;
    }
    if (params->dit_layers != H3_DEFAULT_DIT_LAYERS) {
        h3_set_error(ctx, "the image transformer runs all 50 blocks");
        return 0;
    }
    if (params->tuning.ssd_streaming != 0 && params->tuning.ssd_streaming != 1) {
        h3_set_error(ctx, "SSD streaming must be zero or one");
        return 0;
    }
    if (params->tuning.use_int8_row_fc2 != 0 && params->tuning.use_int8_row_fc2 != 1) {
        h3_set_error(ctx, "int8 row FC2 must be zero or one");
        return 0;
    }
    if (params->use_int8 != 0 && params->use_int8 != 1) {
        h3_set_error(ctx, "int8 must be zero or one");
        return 0;
    }
    if (params->use_int8 && params->tuning.ssd_streaming) {
        h3_set_error(ctx, "SSD streaming uses original BF16 weights and cannot "
                         "be combined with int8");
        return 0;
    }
    if (params->tuning.ssd_streaming && params->tuning.use_int8_row_fc2) {
        h3_set_error(ctx, "SSD streaming uses original BF16 weights and cannot "
                         "be combined with int8 row FC2");
        return 0;
    }
    if ((params->use_int8 || params->tuning.use_int8_row_fc2) &&
        (params->tuning.use_slower_bf16_mlp || params->tuning.use_slower_bf16_qkv ||
         params->tuning.use_slower_bf16_attention_output)) {
        h3_set_error(ctx, "int8 cannot be combined with the BF16 projections");
        return 0;
    }
    if ((params->use_int8 || params->tuning.use_int8_row_fc2) &&
        !h3_device(ctx)->metal4) {
        h3_set_error(ctx, "int8 requires an M5-class Metal 4 GPU");
        return 0;
    }
    return 1;
}

typedef struct {
    h3_generation_progress *progress;
} h3_image_step_state;

static int h3_image_step_bridge(int completed, int total, void *opaque) {
    h3_image_step_state *state = opaque;
    h3_progress_emit(state->progress, "image denoise", completed, total);
    return state->progress->cancelled;
}

static void h3_image_vae_progress_bridge(int completed, int total,
                                         void *opaque) {
    h3_progress_emit(opaque, "image VAE", completed, total);
}

typedef struct {
    const char *name;
    double seconds, footprint_gib, peak_gib;
} h3_image_phase;

typedef struct {
    h3_image_phase phases[16];
    int count;
    double mark;
} h3_image_phase_log;

static double h3_wall_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static void h3_image_phase_end(h3_image_phase_log *log, const char *name) {
    if (log->count >= (int)(sizeof(log->phases) / sizeof(log->phases[0])))
        return;
    h3_image_phase *phase = &log->phases[log->count++];
    double now = h3_wall_seconds();
    phase->name = name;
    phase->seconds = now - log->mark;
    log->mark = now;
    phase->footprint_gib = phase->peak_gib = 0.0;
    h3_gpu_process_footprint(&phase->footprint_gib, &phase->peak_gib);
}

static void h3_image_phase_print(const h3_image_phase_log *log) {
    double total = 0.0;
    fprintf(stderr, "h3 profile: %-16s %9s %14s %14s\n", "phase", "seconds",
            "footprint GiB", "lifetime max");
    for (int index = 0; index < log->count; index++) {
        const h3_image_phase *phase = &log->phases[index];
        total += phase->seconds;
        fprintf(stderr, "h3 profile: %-16s %9.2f %14.2f %14.2f\n",
                phase->name, phase->seconds, phase->footprint_gib,
                phase->peak_gib);
    }
    fprintf(stderr, "h3 profile: %-16s %9.2f\n", "total", total);
}

typedef struct {
    void *dit;
    h3_image_phase_log *log;
    int enabled;
} h3_image_timed_dit;

static int h3_image_timed_velocity(void *opaque, int step, const float *rows,
                                   float *velocity, char *error,
                                   size_t error_size) {
    h3_image_timed_dit *timed = opaque;
    int ok = h3_image_dit_velocity(timed->dit, step, rows, velocity, error,
                                   error_size);
    if (ok && timed->enabled) {
        static const char *names[] = {"DiT step 1", "DiT step 2", "DiT step 3",
                                      "DiT step 4", "DiT step 5", "DiT step 6"};
        h3_image_phase_end(timed->log, names[step % 6]);
    }
    return ok;
}

static void h3_report_peak(const char *label, const h3_gpu_stats *stats) {
    fprintf(stderr, "h3: %s peak GPU buffers %.2f GiB, %.2f GPU seconds\n",
            label,
            (double)stats->peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
            stats->gpu_seconds);
}

typedef struct {
    size_t count;
    uint8_t **sources;      /* planar bytes, upright, at the file's own size */
    int widths[H3_IMAGE_MAX_REFERENCES];
    int heights[H3_IMAGE_MAX_REFERENCES];
    h3_image_reference_geometry geometry[H3_IMAGE_MAX_REFERENCES];
    uint8_t *vision[H3_IMAGE_MAX_REFERENCES];   /* owned Qwen vision copies */
    h3_tokenizer *tokenizer;
    float *condition_rows;  /* [R][96] F32, noised and patchified */
    size_t condition_elements;
    h3_layout_ref layout[H3_IMAGE_MAX_REFERENCES];
} h3_edit_state;

static void h3_edit_state_free(h3_edit_state *edit) {
    for (size_t index = 0; index < H3_IMAGE_MAX_REFERENCES; index++) {
        if (edit->sources) free(edit->sources[index]);
        free(edit->vision[index]);
    }
    free(edit->sources);
    h3_tokenizer_free(edit->tokenizer);
    free(edit->condition_rows);
    memset(edit, 0, sizeof(*edit));
}

static void h3_edit_warn_tags(const char *prompt, size_t count) {
    size_t max_tag;
    uint32_t mentioned;
    h3_image_edit_scan_tags(prompt, count, &max_tag, &mentioned);
    if (max_tag > count)
        fprintf(stderr, "h3: warning: the prompt names <Picture %zu> but only "
                "%zu reference image%s given\n", max_tag, count,
                count == 1 ? " is" : "s are");
    for (size_t index = 0; index < count; index++)
        if (!(mentioned & ((uint32_t)1 << index)))
            fprintf(stderr, "h3: warning: reference %zu is never named in the "
                    "prompt as <Picture %zu>\n", index + 1, index + 1);
}

static int h3_edit_decode(h3_ctx *ctx, const h3_image_params *params,
                          h3_edit_state *edit) {
    char detail[512];
    edit->count = params->reference_count;
    edit->sources = calloc(H3_IMAGE_MAX_REFERENCES, sizeof(*edit->sources));
    if (!edit->sources) {
        h3_set_error(ctx, "out of memory reading the reference images");
        return 0;
    }
    for (size_t index = 0; index < edit->count; index++)
        if (!h3_ffmpeg_read_image_u8(params->references[index].path,
                                     &edit->widths[index],
                                     &edit->heights[index],
                                     &edit->sources[index], detail,
                                     sizeof(detail))) {
            h3_set_error(ctx, "reference %zu: %s", index + 1, detail);
            return 0;
        }
    return 1;
}

static int h3_edit_check(h3_ctx *ctx, h3_edit_state *edit, int latent_height,
                         int latent_width, size_t presentation_tokens,
                         int size_defaulted) {
    char detail[256];
    if (!h3_image_edit_check_limits(edit->widths, edit->heights, edit->count,
                                    latent_height, latent_width,
                                    presentation_tokens, edit->geometry,
                                    detail, sizeof(detail))) {
        int canvas_related = strstr(detail, "rows") != NULL;
        h3_set_error(ctx, "%s%s", detail, size_defaulted && canvas_related ?
                     " (the size was taken from reference 1; pass --size)" :
                     "");
        return 0;
    }
    return 1;
}

/* The length of the text presentation, from the tokenizer alone. */
static int h3_edit_presentation_tokens(h3_ctx *ctx, const char *prompt,
                                       h3_edit_state *edit, size_t *tokens) {
    char detail[512];
    int grid_h[H3_IMAGE_MAX_REFERENCES], grid_w[H3_IMAGE_MAX_REFERENCES];
    char *tokenizer_path = h3_path(ctx->model_dir, "tokenizer/tokenizer.json");
    if (!tokenizer_path) {
        h3_set_error(ctx, "out of memory resolving the tokenizer path");
        return 0;
    }
    edit->tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    free(tokenizer_path);
    if (!edit->tokenizer) {
        h3_set_error(ctx, "%s", detail);
        return 0;
    }
    for (size_t index = 0; index < edit->count; index++) {
        grid_h[index] = edit->geometry[index].vision_h / 16;
        grid_w[index] = edit->geometry[index].vision_w / 16;
    }
    h3_edit_presentation presentation;
    if (!h3_multimodal_edit_presentation(edit->tokenizer, prompt, grid_h,
                                         grid_w, edit->count, &presentation,
                                         detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        return 0;
    }
    *tokens = presentation.count;
    h3_edit_presentation_free(&presentation);
    return 1;
}

static void h3_image_vae_encode_bridge(int completed, int total,
                                       void *opaque) {
    h3_progress_emit(opaque, "reference VAE", completed, total);
}

/* VAE-encode every reference, one at a time, into the condition rows. */
static int h3_edit_encode_references(h3_ctx *ctx, const h3_image_params *params,
                                     const h3_image_hooks *hooks,
                                     h3_edit_state *edit, const char *vae_path,
                                     h3_generation_progress *progress,
                                     int profile, int need_vision) {
    char detail[512];
    size_t rows = 0;
    h3_image_vae_encoder *encoder = NULL;
    int succeeded = 0;
    for (size_t index = 0; index < edit->count; index++)
        rows += (size_t)edit->geometry[index].dit_rows;
    edit->condition_elements = rows * 96;
    edit->condition_rows = malloc(edit->condition_elements *
                                  sizeof(*edit->condition_rows));
    if (!edit->condition_rows) {
        h3_set_error(ctx, "out of memory allocating the reference rows");
        return 0;
    }
    encoder = h3_image_vae_encoder_open(vae_path, "h3_shaders.metal", detail,
                                        sizeof(detail));
    if (!encoder) {
        h3_set_error(ctx, "%s", detail);
        return 0;
    }
    size_t offset = 0;
    for (size_t index = 0; index < edit->count; index++) {
        const h3_image_reference_geometry *g = &edit->geometry[index];
        uint8_t *vae_copy = NULL;
        const uint8_t *vae_pixels;
        if (!h3_image_edit_copies(edit->sources[index], edit->widths[index],
                                  edit->heights[index], g, &vae_copy,
                                  need_vision ? &edit->vision[index] :
                                                NULL)) {
            h3_set_error(ctx, "reference %zu: cannot resize the image",
                         index + 1);
            goto done;
        }
        free(edit->sources[index]);
        edit->sources[index] = NULL;
        if (hooks && hooks->vae_pixels && hooks->vae_pixels[index])
            vae_pixels = hooks->vae_pixels[index];
        else
            vae_pixels = vae_copy;
        h3_reference_latents latents;
        int encoded = h3_image_vae_encode_session(
            encoder, vae_path, vae_pixels, g->vae_h, g->vae_w,
            hooks && hooks->reference_posterior_noise ?
                hooks->reference_posterior_noise[index] : NULL,
            h3_image_edit_stream(params->seed, 1, index),
            h3_image_vae_encode_bridge, progress, &latents, detail,
            sizeof(detail));
        free(vae_copy);
        if (!encoded) {
            if (!progress->cancelled)
                h3_set_error(ctx, "reference %zu: %s", index + 1, detail);
            goto done;
        }
        if (profile) h3_report_peak("reference VAE", &latents.gpu_stats);
        int ok = 1;
        size_t count = (size_t)H3_IMAGE_LATENT_CHANNELS *
                       (size_t)g->latent_h * (size_t)g->latent_w;
        const float *aug = hooks && hooks->reference_aug_noise ?
                           hooks->reference_aug_noise[index] : NULL;
        float *drawn = NULL;
        if (latents.latent_height != g->latent_h ||
            latents.latent_width != g->latent_w) {
            h3_set_error(ctx, "reference %zu: the VAE returned a %dx%d latent, "
                         "expected %dx%d", index + 1, latents.latent_width,
                         latents.latent_height, g->latent_w, g->latent_h);
            ok = 0;
        } else if (!aug) {
            drawn = malloc(count * sizeof(*drawn));
            if (drawn) {
                h3_image_noise(h3_image_edit_stream(params->seed, 2, index),
                               g->latent_h, g->latent_w, drawn);
                aug = drawn;
            }
        }
        if (ok && !aug) {
            h3_set_error(ctx, "out of memory drawing the reference noise");
            ok = 0;
        }
        if (ok && !h3_image_edit_condition_rows(
                latents.latents, aug, g->latent_h, g->latent_w,
                edit->condition_rows + offset)) {
            h3_set_error(ctx, "reference %zu: cannot build the condition rows",
                         index + 1);
            ok = 0;
        }
        free(drawn);
        h3_reference_latents_free(&latents);
        if (!ok) goto done;
        offset += (size_t)g->dit_rows * 96;
        edit->layout[index] = h3_image_edit_layout_ref(g);
        if (progress->cancelled) goto done;
    }
    succeeded = 1;
done:
    h3_image_vae_encoder_close(encoder);
    return succeeded;
}

/* Qwen vision tower per reference, then the language layers; the tower and the
 * language model are never resident together. */
static int h3_edit_encode_text(h3_ctx *ctx, const char *prompt,
                               h3_edit_state *edit,
                               const h3_image_hooks *hooks,
                               h3_generation_progress *progress, int profile,
                               h3_text_embedding *text) {
    char detail[512];
    char *text_path = h3_path(ctx->model_dir, "text_encoder");
    h3_vision_output *outputs = calloc(edit->count, sizeof(*outputs));
    int ok = text_path && outputs;
    if (!ok) h3_set_error(ctx, "out of memory encoding the reference images");
    for (size_t index = 0; ok && index < edit->count; index++) {
        const h3_image_reference_geometry *g = &edit->geometry[index];
        const uint8_t *bytes = hooks && hooks->vision_pixels &&
                               hooks->vision_pixels[index] ?
            hooks->vision_pixels[index] : edit->vision[index];
        size_t count = (size_t)3 * (size_t)g->vision_h * (size_t)g->vision_w;
        float *pixels = malloc(count * sizeof(*pixels));
        if (!pixels) {
            h3_set_error(ctx, "out of memory encoding the reference images");
            ok = 0;
            break;
        }
        for (size_t element = 0; element < count; element++)
            pixels[element] = (float)bytes[element] / 255.0f;
        ok = h3_vision_encode_bf16(text_path, "h3_shaders.metal", pixels, 1,
                                   g->vision_h, g->vision_w,
                                   h3_vision_progress_bridge, progress,
                                   &outputs[index], detail, sizeof(detail));
        free(pixels);
        free(edit->vision[index]);
        edit->vision[index] = NULL;
        if (!ok) {
            if (!progress->cancelled)
                h3_set_error(ctx, "reference %zu: %s", index + 1, detail);
        } else if (outputs[index].tokens != (size_t)g->vision_tokens) {
            h3_set_error(ctx, "reference %zu: the vision tower returned %zu "
                         "tokens, expected %d", index + 1, outputs[index].tokens,
                         g->vision_tokens);
            ok = 0;
        } else if (progress->cancelled) {
            ok = 0;
        } else if (profile) {
            h3_report_peak("Qwen vision", &outputs[index].gpu_stats);
        }
    }
    if (ok) {
        h3_progress_emit(progress, "text encoder", 0, 50);
        ok = h3_multimodal_encode_edit_bf16(
            edit->tokenizer, text_path, "h3_shaders.metal", prompt, outputs,
            edit->count, h3_text_progress_bridge, progress, text, detail,
            sizeof(detail));
        if (!ok && !progress->cancelled) h3_set_error(ctx, "%s", detail);
    }
    if (outputs)
        for (size_t index = 0; index < edit->count; index++)
            h3_vision_output_free(&outputs[index]);
    free(outputs);
    free(text_path);
    return ok;
}

/* Gather what names the computation (model directory, text encoder inventory,
 * device, build identity) and compose the key with the cache module. */
char *h3_image_text_cache_key(const h3_ctx *ctx, const char *prompt,
                              size_t count, const int *widths,
                              const int *heights,
                              const uint8_t *const *sources,
                              const int *vision_widths,
                              const int *vision_heights) {
    const h3_component_info *encoder = &ctx->model.text_encoder;
    h3_image_cache_inputs inputs = {
        ctx->model_dir, ctx->device.name,
        (unsigned long long)encoder->bytes, encoder->files, encoder->tensors,
        H3_TEXT_BUILD_ID};
    return h3_image_cache_key(&inputs, prompt, count, widths, heights,
                              sources, vision_widths, vision_heights);
}

typedef struct {
    char *key;
    char path[1024];
    const char *label;
    int hit;
} h3_text_cache_state;

/* Look the text embedding up on disk. The key is built from the decoded
 * references, so this runs before the reference VAE frees them. */
static void h3_text_cache_lookup(const h3_ctx *ctx, const char *prompt,
                                 const h3_edit_state *edit,
                                 size_t presentation_tokens,
                                 h3_image_hooks *hooks,
                                 h3_text_cache_state *cache,
                                 h3_text_embedding *text) {
    const uint8_t *sources[H3_IMAGE_MAX_REFERENCES];
    int vision_w[H3_IMAGE_MAX_REFERENCES], vision_h[H3_IMAGE_MAX_REFERENCES];
    if (hooks) hooks->cache_result = H3_CACHE_NOT_USED;
    cache->label = edit->count ? "edit text cache" : "text cache";
    if (!ctx->cache_dir ||
        (hooks && (hooks->vae_pixels || hooks->vision_pixels))) {
        return;
    }
    for (size_t index = 0; index < edit->count; index++) {
        sources[index] = edit->sources[index];
        vision_w[index] = edit->geometry[index].vision_w;
        vision_h[index] = edit->geometry[index].vision_h;
    }
    cache->key = h3_image_text_cache_key(ctx, prompt, edit->count,
                                         edit->widths, edit->heights, sources,
                                         vision_w, vision_h);
    if (!cache->key) return;
    cache->hit = h3_image_cache_load(ctx->cache_dir, cache->key, text,
                                     cache->path, sizeof(cache->path));
    if (cache->hit && edit->count && text->tokens != presentation_tokens) {
        fprintf(stderr, "h3: warning: %s: ignoring %s: %zu rows, the prompt "
                "makes %zu\n", cache->label, cache->path, text->tokens,
                presentation_tokens);
        h3_text_embedding_free(text);
        cache->hit = 0;
    }
    if (hooks) hooks->cache_result = cache->hit ? H3_CACHE_HIT :
                                                  H3_CACHE_MISS;
    if (cache->hit)
        fprintf(stderr, "h3: %s hit %s\n", cache->label, cache->path);
}

static void h3_text_cache_store(const h3_ctx *ctx, h3_text_cache_state *cache,
                                const h3_text_embedding *text) {
    if (cache->key && !cache->hit &&
        h3_image_cache_store(ctx->cache_dir, cache->key, text,
                             h3_image_cache_limit(), cache->path,
                             sizeof(cache->path)))
        fprintf(stderr, "h3: %s miss; stored %s\n", cache->label,
                cache->path);
}

h3_result *h3_generate_image_with_hooks(h3_ctx *ctx, const char *prompt,
                                        const h3_image_params *params,
                                        h3_image_hooks *hooks) {
    if (!ctx) return NULL;
    ctx->error[0] = '\0';
    if (!ctx->image) {
        h3_set_error(ctx, "model directory is not a MiniMax-H3-Image "
                          "checkpoint");
        return NULL;
    }
    if (!prompt || !*prompt) {
        h3_set_error(ctx, "prompt must not be empty");
        return NULL;
    }
    if (!h3_valid_image_params(ctx, params)) return NULL;
    int profile = getenv("H3_PROFILE") != NULL;
    h3_generation_progress progress = {
        ctx, params->on_progress, params->callback_opaque, NULL, 0};
    char detail[512];
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_dit *dit = NULL;
    float *latent = NULL;
    h3_image_planes planes;
    memset(&planes, 0, sizeof(planes));
    uint8_t *planar = NULL, *rgb = NULL;
    h3_result *result = NULL;
    char *dit_path = NULL, *vae_path = NULL;
    int edit = params->reference_count > 0;
    h3_edit_state edit_state;
    memset(&edit_state, 0, sizeof(edit_state));
    h3_text_cache_state text_cache;
    memset(&text_cache, 0, sizeof(text_cache));
    h3_image_phase_log phase_log = {{{0}}, 0, h3_wall_seconds()};
    int request_width = params->width, request_height = params->height;
    int size_defaulted = 0;
    if (edit) {
        h3_edit_warn_tags(prompt, params->reference_count);
        if (!h3_edit_decode(ctx, params, &edit_state)) goto cleanup;
        if (!request_width) {
            size_defaulted = 1;
            request_width = edit_state.widths[0];
            request_height = edit_state.heights[0];
        }
    }
    int canvas_height, canvas_width;
    if (!h3_image_canvas(request_height, request_width,
                         params->min_generation_area, &canvas_height,
                         &canvas_width)) {
        h3_set_error(ctx, "cannot compute the generation canvas");
        goto cleanup;
    }
    const char *canvas_error = h3_image_canvas_error(canvas_height,
                                                     canvas_width);
    if (canvas_error) {
        h3_set_error(ctx, "%s (%dx%d for a %dx%d request)%s", canvas_error,
                     canvas_width, canvas_height, request_width,
                     request_height, size_defaulted ?
                     " (the size was taken from reference 1; pass --size)" :
                     "");
        goto cleanup;
    }
    int latent_height = canvas_height / H3_IMAGE_VAE_SCALE;
    int latent_width = canvas_width / H3_IMAGE_VAE_SCALE;
    fprintf(stderr, "h3: image %dx%d requested, generating on a %dx%d canvas "
            "(latent %dx%d, %d image tokens)\n", request_width, request_height,
            canvas_width, canvas_height, latent_width, latent_height,
            (latent_height / 2) * (latent_width / 2));
    size_t presentation_tokens = 0;
    if (edit) {
        if (!h3_edit_check(ctx, &edit_state, latent_height, latent_width, 0,
                           size_defaulted) ||
            !h3_edit_presentation_tokens(ctx, prompt, &edit_state,
                                         &presentation_tokens) ||
            !h3_edit_check(ctx, &edit_state, latent_height, latent_width,
                           presentation_tokens, size_defaulted)) goto cleanup;
        if (hooks) hooks->text_tokens = presentation_tokens;
    }
    h3_text_cache_lookup(ctx, prompt, &edit_state, presentation_tokens, hooks,
                         &text_cache, &text);
    if (text_cache.hit && hooks && hooks->stop_after_lookup) {
        hooks->text = text;
        memset(&text, 0, sizeof(text));
        goto deliver;
    }
    dit_path = h3_path(ctx->model_dir, "transformer_diffusers");
    vae_path = h3_path(ctx->model_dir, "vae_diffusers");
    if (!dit_path || !vae_path) {
        h3_set_error(ctx, "out of memory resolving image model paths");
        goto cleanup;
    }
    if (edit) {
        h3_progress_emit(&progress, "reference VAE", 0, 1);
        if (!h3_edit_encode_references(ctx, params, hooks, &edit_state,
                                       vae_path, &progress, profile,
                                       !text_cache.hit))
            goto cleanup;
        if (profile) h3_image_phase_end(&phase_log, "reference VAE");
        if (hooks && hooks->references_only) goto deliver;
        if (text_cache.hit) {
            h3_progress_emit(&progress, "text encoder", 50, 50);
        } else {
            if (!h3_edit_encode_text(ctx, prompt, &edit_state, hooks,
                                     &progress, profile, &text)) goto cleanup;
            h3_text_cache_store(ctx, &text_cache, &text);
            if (profile) h3_report_peak("text encoder", &text.gpu_stats);
        }
        if (profile) h3_image_phase_end(&phase_log, "reference text");
    } else {
        if (text_cache.hit) {
            h3_progress_emit(&progress, "text encoder", 50, 50);
        } else {
            h3_progress_emit(&progress, "text encoder", 0, 50);
            if (!h3_image_encode_prompt(ctx->model_dir, prompt,
                                        h3_text_progress_bridge, &progress,
                                        NULL, NULL, &text, detail,
                                        sizeof(detail))) {
                if (!progress.cancelled) h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            h3_text_cache_store(ctx, &text_cache, &text);
            if (profile) h3_report_peak("text encoder", &text.gpu_stats);
        }
        if (profile) h3_image_phase_end(&phase_log, "text encoder");
    }
    if (progress.cancelled) goto cleanup;
    int resident_bf16 = !params->use_int8 && !params->tuning.use_int8_row_fc2 &&
                        !params->tuning.ssd_streaming;
    /* The resident BF16 path runs the close-reference projections. */
    h3_dit_tuning dit_tuning = params->tuning;
    if (resident_bf16) {
        dit_tuning.use_slower_bf16_mlp = 1;
        dit_tuning.use_slower_bf16_qkv = 1;
        dit_tuning.use_slower_bf16_attention_output = 1;
    }
    dit = h3_dit_load_image(
        dit_path, "h3_shaders.metal", &text, latent_height, latent_width,
        edit ? edit_state.layout : NULL, edit ? edit_state.count : 0,
        edit ? edit_state.condition_rows : NULL,
        edit ? edit_state.condition_elements : 0,
        (unsigned)params->dit_layers, &dit_tuning,
        h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    h3_text_embedding_free(&text);
    if (!dit) {
        if (!progress.cancelled) h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (profile) h3_image_phase_end(&phase_log, "DiT load");
    fprintf(stderr, "h3: image DiT weights: %s\n", h3_dit_weight_mode(dit));
    if (hooks) hooks->weight_mode = h3_dit_weight_mode(dit);
    if (progress.cancelled) goto cleanup;
    size_t latent_elements = (size_t)H3_IMAGE_LATENT_CHANNELS *
                             (size_t)latent_height * (size_t)latent_width;
    latent = malloc(latent_elements * sizeof(*latent));
    if (!latent) {
        h3_set_error(ctx, "out of memory allocating the image latent");
        goto cleanup;
    }
    if (hooks && hooks->noise)
        memcpy(latent, hooks->noise, latent_elements * sizeof(*latent));
    else
        h3_image_noise(params->seed, latent_height, latent_width, latent);
    h3_image_step_state step_state = {&progress};
    h3_image_timed_dit timed = {dit, &phase_log, profile};
    if (!h3_image_sample(latent_height, latent_width, latent,
                         h3_image_timed_velocity, &timed, h3_image_step_bridge,
                         &step_state, latent, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_gpu_stats dit_stats;
    if (profile && h3_dit_get_gpu_stats(dit, &dit_stats))
        h3_report_peak("image DiT", &dit_stats);
    h3_dit_free(dit);
    dit = NULL;
    if (profile) h3_image_phase_end(&phase_log, "DiT free");
    h3_progress_emit(&progress, "image VAE", 0, 1);
    if (progress.cancelled) goto cleanup;
    if (!h3_image_vae_decode(vae_path, "h3_shaders.metal", latent,
                             latent_height, latent_width,
                             h3_image_vae_progress_bridge, &progress, &planes,
                             detail, sizeof(detail))) {
        if (!progress.cancelled) h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (profile) h3_report_peak("image VAE", &planes.gpu_stats);
    if (profile) h3_image_phase_end(&phase_log, "image VAE");
    if (planes.height != canvas_height || planes.width != canvas_width) {
        h3_set_error(ctx, "image VAE returned %dx%d, expected %dx%d",
                     planes.width, planes.height, canvas_width, canvas_height);
        goto cleanup;
    }
    size_t canvas_pixels = (size_t)canvas_width * (size_t)canvas_height;
    planar = malloc(canvas_pixels * 3);
    rgb = malloc(canvas_pixels * 3);
    if (!planar || !rgb) {
        h3_set_error(ctx, "out of memory converting the generated image");
        goto cleanup;
    }
    h3_image_planes_to_rgb(&planes);
    h3_image_planes_to_u8(&planes, planar);
    h3_image_planes_free(&planes);
    h3_image_interleave_rgb(planar, canvas_width, canvas_height, rgb);
    free(planar);
    planar = NULL;
    if (progress.cancelled) goto cleanup;
    if (canvas_width != request_width || canvas_height != request_height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(rgb, 1, canvas_width, canvas_height,
                                          request_width, request_height,
                                          &resized)) {
            h3_set_error(ctx, "cannot resize the generated image");
            goto cleanup;
        }
        free(rgb);
        rgb = resized;
    }
    if (params->on_frame) {
        h3_frame frame = {request_width, request_height, request_width * 3,
                          rgb, 0, 1, -1, 0};
        if (params->on_frame(&frame, params->callback_opaque)) {
            h3_set_error(ctx, "generation cancelled while delivering the image");
            goto cleanup;
        }
    }
    if (params->output_path && *params->output_path) {
        h3_progress_emit(&progress, "PNG", 0, 1);
        if (!h3_ffmpeg_write_png_rgb24(params->output_path, rgb, request_width,
                                       request_height, detail,
                                       sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "PNG", 1, 1);
    }
    if (profile) {
        h3_image_phase_end(&phase_log, "convert+PNG");
        h3_image_phase_print(&phase_log);
    }
deliver:
    result = calloc(1, sizeof(*result));
    if (!result) {
        h3_set_error(ctx, "out of memory creating generation result");
        goto cleanup;
    }
    result->width = request_width;
    result->height = request_height;
    result->frames = 1;
    result->seed = params->seed;
    if (hooks) {
        if (edit) {
            hooks->condition_rows = edit_state.condition_rows;
            hooks->condition_elements = edit_state.condition_elements;
            edit_state.condition_rows = NULL;
        }
        hooks->latent = latent;
        latent = NULL;
        hooks->rgb = rgb;
        rgb = NULL;
    }

cleanup:
    h3_edit_state_free(&edit_state);
    free(text_cache.key);
    free(dit_path);
    free(vae_path);
    h3_text_embedding_free(&text);
    h3_dit_free(dit);
    h3_image_planes_free(&planes);
    free(latent);
    free(planar);
    free(rgb);
    return result;
}

h3_result *h3_generate_image(h3_ctx *ctx, const char *prompt,
                             const h3_image_params *params) {
    return h3_generate_image_with_hooks(ctx, prompt, params, NULL);
}
