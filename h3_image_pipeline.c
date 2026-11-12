#include "h3_image_pipeline.h"

#include "h3_host.h"
#include "h3_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

int h3_image_tokenize(const h3_tokenizer *tokenizer, const char *prompt,
                      uint32_t **ids, size_t *count,
                      char *error, size_t error_size) {
    if (!tokenizer || !prompt || !ids || !count) {
        fail(error, error_size, "invalid image tokenization arguments");
        return 0;
    }
    if (!h3_tokenizer_encode(tokenizer, prompt, 0, ids, count, error,
                             error_size)) return 0;
    if (*count == 0) {
        h3_tokenizer_ids_free(*ids);
        *ids = NULL;
        fail(error, error_size, "prompt must not be empty");
        return 0;
    }
    if (*count > H3_IMAGE_MAX_PROMPT_TOKENS) *count = H3_IMAGE_MAX_PROMPT_TOKENS;
    return 1;
}

static char *join_path(const char *root, const char *relative) {
    size_t size = strlen(root) + strlen(relative) + 2;
    char *path = malloc(size);
    if (path) snprintf(path, size, "%s/%s", root, relative);
    return path;
}

int h3_image_encode_prompt(const char *model_directory, const char *prompt,
                           h3_text_progress progress, void *progress_opaque,
                           uint32_t **ids, size_t *count,
                           h3_text_embedding *text,
                           char *error, size_t error_size) {
    if (!model_directory || !prompt || !text) {
        fail(error, error_size, "invalid image text encoding arguments");
        return 0;
    }
    if (ids) *ids = NULL;
    if (count) *count = 0;
    char *tokenizer_path = join_path(model_directory, "tokenizer/tokenizer.json");
    char *encoder_path = join_path(model_directory, "text_encoder");
    h3_tokenizer *tokenizer = NULL;
    uint32_t *token_ids = NULL;
    size_t token_count = 0;
    int ok = 0;
    if (!tokenizer_path || !encoder_path) {
        fail(error, error_size, "out of memory resolving image text paths");
        goto done;
    }
    tokenizer = h3_tokenizer_load(tokenizer_path, error, error_size);
    if (!tokenizer || !h3_image_tokenize(tokenizer, prompt, &token_ids,
                                         &token_count, error, error_size))
        goto done;
    if (!h3_text_encode_bf16(encoder_path, "h3_shaders.metal", token_ids,
                             token_count, progress, progress_opaque, text,
                             error, error_size)) goto done;
    if (ids) {
        *ids = token_ids;
        token_ids = NULL;
    }
    if (count) *count = token_count;
    ok = 1;
done:
    h3_tokenizer_ids_free(token_ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(encoder_path);
    return ok;
}

void h3_image_noise(uint64_t seed, int latent_height, int latent_width,
                    float *noise) {
    h3_rng rng;
    h3_rng_seed(&rng, seed);
    h3_rng_fill_normal(&rng, noise, (size_t)H3_IMAGE_LATENT_CHANNELS *
                       (size_t)latent_height * (size_t)latent_width);
}

int h3_image_sample(int latent_height, int latent_width, const float *noise,
                    h3_image_velocity velocity, void *velocity_opaque,
                    h3_image_step_progress progress, void *progress_opaque,
                    float *latent_out, char *error, size_t error_size) {
    if (!noise || !velocity || !latent_out || latent_height < 2 ||
        latent_width < 2 || latent_height % 2 || latent_width % 2) {
        fail(error, error_size, "invalid image sampler arguments");
        return 0;
    }
    h3_sigma_schedule schedule;
    if (!h3_image_schedule_build(&schedule)) {
        fail(error, error_size, "cannot build the image schedule");
        return 0;
    }
    size_t latent_elements = (size_t)H3_IMAGE_LATENT_CHANNELS *
                             (size_t)latent_height * (size_t)latent_width;
    float *rows = malloc(latent_elements * sizeof(*rows));
    float *step_velocity = malloc(latent_elements * sizeof(*step_velocity));
    int ok = 0;
    if (!rows || !step_velocity) {
        fail(error, error_size, "out of memory allocating image sampler rows");
        goto done;
    }
    if (!h3_dit_patchify_video(noise, H3_IMAGE_LATENT_CHANNELS, 1,
                               latent_height, latent_width, rows,
                               latent_elements)) {
        fail(error, error_size, "cannot patchify the image noise");
        goto done;
    }
    if (progress && progress(0, schedule.steps, progress_opaque)) {
        fail(error, error_size, "generation cancelled before the first step");
        goto done;
    }
    for (int step = 0; step < schedule.steps; step++) {
        if (!velocity(velocity_opaque, step, rows, step_velocity, error,
                      error_size)) goto done;
        if (!h3_euler_velocity_step(rows, step_velocity, latent_elements,
                                    schedule.video[step],
                                    schedule.video[step + 1])) {
            fail(error, error_size, "invalid Euler step");
            goto done;
        }
        if (progress && progress(step + 1, schedule.steps, progress_opaque)) {
            fail(error, error_size, "generation cancelled during denoising");
            goto done;
        }
    }
    if (!h3_dit_unpatchify_video(rows, H3_IMAGE_LATENT_CHANNELS, 1,
                                 latent_height, latent_width, latent_out,
                                 latent_elements)) {
        fail(error, error_size, "cannot unpatchify the image latent");
        goto done;
    }
    ok = 1;
done:
    free(rows);
    free(step_velocity);
    return ok;
}

int h3_image_dit_velocity(void *dit, int step, const float *rows,
                          float *velocity, char *error, size_t error_size) {
    return h3_dit_forward_image(dit, step, rows, velocity, error, error_size);
}

void h3_image_interleave_rgb(const uint8_t *planar, int width, int height,
                             uint8_t *interleaved) {
    size_t plane = (size_t)width * (size_t)height;
    for (size_t pixel = 0; pixel < plane; pixel++)
        for (size_t channel = 0; channel < 3; channel++)
            interleaved[pixel * 3 + channel] = planar[channel * plane + pixel];
}
