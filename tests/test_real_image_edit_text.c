#include "h3_ffmpeg.h"
#include "h3_image_vae_encode.h"
#include "h3_multimodal.h"
#include "h3_safetensors.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { TEXT_DIM = 5120 };

/* Bounds are generous placeholders until the gated window measures them; the
 * presentation (a) is exact. */
#define VISION_REL_L2_BOUND 0.15      /* measured in the gated window */
#define VISION_REL_MAX_BOUND 0.25     /* measured in the gated window */
#define LANGUAGE_ROWS_BOUND 5e-2      /* measured in the gated window */
#define VISION_ROWS_BOUND 0.15        /* measured in the gated window */
#define DRIFT_EMBEDDING_BOUND 0.2     /* measured in the gated window */

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_edit_text.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_edit_text.c: fixture "
                "tensor %s is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes ? bytes : 1);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static float bf16_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef struct {
    double rel_l2, rel_max, max_abs;
} difference;

static difference compare_bf16(const uint16_t *got, const uint16_t *want,
                               size_t count) {
    double square_error = 0.0, square_value = 0.0, maximum = 0.0, scale = 0.0;
    for (size_t index = 0; index < count; index++) {
        double actual = (double)bf16_value(got[index]);
        double expected = (double)bf16_value(want[index]);
        require(isfinite(actual), "non-finite value");
        double delta = actual - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        square_error += delta * delta;
        square_value += expected * expected;
    }
    difference result = {
        sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24)),
        maximum / (scale > 1e-12 ? scale : 1e-12), maximum};
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "  %d/%d layers\n", completed, total);
}

typedef struct {
    const char *name;
    size_t tokens;
    int grid_h, grid_w;
    int height, width;
    char *prompt;
    int32_t *token_ids, *mm_types, *positions;
    uint8_t *tags, *vision_u8, *reference_u8;
    uint16_t *merged, *deepstack, *embeds;
    int reference_height, reference_width;
} fixture_case;

static void load_case(fixture_case *fixture_data, const char *name,
                      const char *path) {
    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(path, &fixture, error, sizeof(error))) die(error);
    fixture_data->name = name;
    const h3_st_tensor *ids = h3_st_find(&fixture, "token_ids");
    const h3_st_tensor *prompt = h3_st_find(&fixture, "prompt");
    const h3_st_tensor *vision = h3_st_find(&fixture, "vision_u8");
    const h3_st_tensor *reference = h3_st_find(&fixture, "reference_u8");
    require(ids && prompt && vision && reference && vision->ndim == 3 &&
            reference->ndim == 3 && vision->shape[0] == 3 &&
            reference->shape[0] == 3, "fixture geometry is malformed");
    size_t tokens = h3_st_tensor_elements(ids);
    size_t prompt_bytes = h3_st_tensor_elements(prompt);
    int32_t *grid = read_fixture(&fixture, "image_grid_thw", H3_DTYPE_I32, 3);
    require(grid[0] == 1 && grid[1] > 0 && grid[2] > 0 && grid[1] % 2 == 0 &&
            grid[2] % 2 == 0, "image_grid_thw is malformed");
    fixture_data->grid_h = grid[1];
    fixture_data->grid_w = grid[2];
    free(grid);
    fixture_data->tokens = tokens;
    fixture_data->height = (int)vision->shape[1];
    fixture_data->width = (int)vision->shape[2];
    fixture_data->reference_height = (int)reference->shape[1];
    fixture_data->reference_width = (int)reference->shape[2];
    require(fixture_data->height == fixture_data->grid_h * 16 &&
            fixture_data->width == fixture_data->grid_w * 16,
            "vision_u8 and image_grid_thw disagree");
    fixture_data->prompt = read_fixture(&fixture, "prompt", H3_DTYPE_U8,
                                        prompt_bytes);
    fixture_data->prompt = realloc(fixture_data->prompt, prompt_bytes + 1);
    require(fixture_data->prompt != NULL, "oom");
    fixture_data->prompt[prompt_bytes] = '\0';
    fixture_data->token_ids = read_fixture(&fixture, "token_ids",
                                           H3_DTYPE_I32, tokens);
    fixture_data->mm_types = read_fixture(&fixture, "mm_token_type_ids",
                                          H3_DTYPE_I32, tokens);
    fixture_data->positions = read_fixture(&fixture, "position_ids",
                                           H3_DTYPE_I32, 3 * tokens);
    fixture_data->tags = read_fixture(&fixture, "token_tags", H3_DTYPE_U8,
                                      tokens);
    size_t vision_elements = (size_t)3 * (size_t)fixture_data->height *
                             (size_t)fixture_data->width;
    size_t reference_elements = (size_t)3 *
        (size_t)fixture_data->reference_height *
        (size_t)fixture_data->reference_width;
    fixture_data->vision_u8 = read_fixture(&fixture, "vision_u8",
                                           H3_DTYPE_U8, vision_elements);
    fixture_data->reference_u8 = read_fixture(&fixture, "reference_u8",
                                              H3_DTYPE_U8, reference_elements);
    size_t vision_tokens = (size_t)fixture_data->grid_h *
                           (size_t)fixture_data->grid_w / 4;
    fixture_data->merged = read_fixture(&fixture, "vision_merged",
                                        H3_DTYPE_BF16, vision_tokens * TEXT_DIM);
    fixture_data->deepstack = read_fixture(&fixture, "vision_deepstack",
                                           H3_DTYPE_BF16,
                                           3 * vision_tokens * TEXT_DIM);
    fixture_data->embeds = read_fixture(&fixture, "prompt_embeds",
                                        H3_DTYPE_BF16, tokens * TEXT_DIM);
    h3_st_free_header(&fixture);
}

static void free_case(fixture_case *fixture_data) {
    free(fixture_data->prompt); free(fixture_data->token_ids);
    free(fixture_data->mm_types); free(fixture_data->positions);
    free(fixture_data->tags); free(fixture_data->vision_u8);
    free(fixture_data->reference_u8); free(fixture_data->merged);
    free(fixture_data->deepstack); free(fixture_data->embeds);
}

/* (a) The token ids, DiT tags, Qwen token types and mRoPE positions h3.c
 * builds for the fixture's grid and prompt equal the reference, exactly. */
static void check_presentation(const h3_tokenizer *tokenizer,
                               const fixture_case *fixture_data) {
    char error[512];
    h3_edit_presentation presentation;
    require(h3_multimodal_edit_presentation(
                tokenizer, fixture_data->prompt, &fixture_data->grid_h,
                &fixture_data->grid_w, 1, &presentation, error,
                sizeof(error)), error);
    size_t tokens = fixture_data->tokens;
    require(presentation.count == tokens, "presentation length differs");
    for (size_t index = 0; index < tokens; index++) {
        require((int32_t)presentation.ids[index] ==
                fixture_data->token_ids[index], "token ids differ");
        require(presentation.tags[index] == fixture_data->tags[index],
                "token tags differ");
        require((int32_t)presentation.mm_token_type_ids[index] ==
                fixture_data->mm_types[index], "mm token types differ");
    }
    for (size_t index = 0; index < 3 * tokens; index++)
        require((int32_t)presentation.positions[index] ==
                fixture_data->positions[index], "rope positions differ");
    h3_edit_presentation_free(&presentation);
    printf("%s (a) presentation: %zu ids, tags, token types and positions "
           "identical\n", fixture_data->name, tokens);
}

static float *pixels_from_u8(const uint8_t *planar, size_t count) {
    float *pixels = malloc(count * sizeof(*pixels));
    require(pixels != NULL, "oom");
    for (size_t index = 0; index < count; index++)
        pixels[index] = (float)planar[index] / 255.0f;
    return pixels;
}

static void encode_vision(const char *encoder_directory, const float *pixels,
                          int height, int width, h3_vision_output *output) {
    char error[512];
    require(h3_vision_encode_bf16(encoder_directory, "h3_shaders.metal",
                                  pixels, 1, height, width, progress, NULL,
                                  output, error, sizeof(error)), error);
}

/* (b) The vision tower on the fixture's resized pixels. */
static void check_vision(const fixture_case *fixture_data,
                         const h3_vision_output *output) {
    size_t vision_tokens = (size_t)fixture_data->grid_h *
                           (size_t)fixture_data->grid_w / 4;
    require(output->tokens == vision_tokens &&
            output->grid_h == fixture_data->grid_h &&
            output->grid_w == fixture_data->grid_w,
            "vision output geometry differs");
    difference merged = compare_bf16(output->merged, fixture_data->merged,
                                     vision_tokens * TEXT_DIM);
    printf("%s (b) vision merged: rel-L2 %.6g, rel-max %.6g, max abs %.6g\n",
           fixture_data->name, merged.rel_l2, merged.rel_max, merged.max_abs);
    require(merged.rel_l2 < VISION_REL_L2_BOUND &&
            merged.rel_max < VISION_REL_MAX_BOUND,
            "vision merged rows are off");
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++) {
        difference stack = compare_bf16(
            output->deepstack[index],
            fixture_data->deepstack + index * vision_tokens * TEXT_DIM,
            vision_tokens * TEXT_DIM);
        printf("%s (b) vision deepstack %u: rel-L2 %.6g, rel-max %.6g, max "
               "abs %.6g\n", fixture_data->name, index, stack.rel_l2,
               stack.rel_max, stack.max_abs);
        require(stack.rel_l2 < VISION_REL_L2_BOUND &&
                stack.rel_max < VISION_REL_MAX_BOUND,
                "vision deepstack rows are off");
    }
}

/* Embedding error over the language rows and over the vision span rows
 * (tag 0), printed separately, against bound_language and bound_vision. */
static void compare_embeddings(const char *label, const fixture_case *fixture_data,
                               const h3_text_embedding *text,
                               double bound_language, double bound_vision) {
    size_t tokens = fixture_data->tokens;
    require(text->tokens == tokens && text->width == TEXT_DIM && text->tags,
            "embedding has the wrong shape");
    double error[2] = {0, 0}, value[2] = {0, 0};
    size_t rows[2] = {0, 0};
    for (size_t row = 0; row < tokens; row++) {
        int kind = fixture_data->tags[row] ? 1 : 0;
        rows[kind]++;
        for (size_t column = 0; column < TEXT_DIM; column++) {
            double got = (double)bf16_value(text->values[row * TEXT_DIM + column]);
            double want = (double)bf16_value(
                fixture_data->embeds[row * TEXT_DIM + column]);
            require(isfinite(got), "non-finite embedding");
            error[kind] += (got - want) * (got - want);
            value[kind] += want * want;
        }
    }
    double language = sqrt(error[1] / (value[1] > 1e-24 ? value[1] : 1e-24));
    double vision = rows[0] ? sqrt(error[0] / (value[0] > 1e-24 ? value[0] :
                                               1e-24)) : 0.0;
    printf("%s %s: language rows (%zu) rel-L2 %.6g, vision rows (%zu) rel-L2 "
           "%.6g\n", fixture_data->name, label, rows[1], language, rows[0],
           vision);
    require(language < bound_language, "language rows are off");
    require(vision < bound_vision, "vision rows are off");
}

/* (c) The full presentation around the tower's output. */
static void check_embeddings(const h3_tokenizer *tokenizer,
                             const char *encoder_directory,
                             const fixture_case *fixture_data,
                             const h3_vision_output *vision) {
    char error[512];
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    require(h3_multimodal_encode_edit_bf16(
                tokenizer, encoder_directory, "h3_shaders.metal",
                fixture_data->prompt, vision, 1, progress, NULL, &text, error,
                sizeof(error)), error);
    require(text.tags != NULL &&
            !memcmp(text.tags, fixture_data->tags, fixture_data->tokens),
            "embedding tags differ from the fixture");
    compare_embeddings("(c) prompt_embeds", fixture_data, &text,
                       LANGUAGE_ROWS_BOUND, VISION_ROWS_BOUND);
    h3_text_embedding_free(&text);
}

/* (d) h3.c's own resize of the reference into the vision copy: PIL-exact
 * Lanczos (what the pipeline uses for the first vision stage) and the FFmpeg
 * scaler. The second vision stage (smart_resize) does nothing when the size is
 * already a multiple of 32 and within the pixel bounds, as for the fixtures. */
static void check_resampling(const h3_tokenizer *tokenizer,
                             const char *encoder_directory,
                             const fixture_case *fixture_data) {
    char error[512];
    size_t count = (size_t)3 * (size_t)fixture_data->height *
                   (size_t)fixture_data->width;
    uint8_t *lanczos = malloc(count);
    require(lanczos != NULL, "oom");
    require(h3_reference_resize_lanczos(
                fixture_data->reference_u8, fixture_data->reference_height,
                fixture_data->reference_width, fixture_data->height,
                fixture_data->width, lanczos), "Lanczos resize failed");
    size_t different = 0;
    int maximum = 0;
    for (size_t index = 0; index < count; index++) {
        int delta = abs((int)lanczos[index] - (int)fixture_data->vision_u8[index]);
        if (delta) different++;
        if (delta > maximum) maximum = delta;
    }
    printf("%s (d) Lanczos resize vs vision_u8: %zu of %zu bytes differ, max "
           "abs %d\n", fixture_data->name, different, count, maximum);

    /* The reference image through FFmpeg: write a PNG, decode it at the
     * vision size, compare in 0..255 units. */
    size_t reference_pixels = (size_t)fixture_data->reference_height *
                              (size_t)fixture_data->reference_width;
    uint8_t *interleaved = malloc(reference_pixels * 3);
    require(interleaved != NULL, "oom");
    for (size_t pixel = 0; pixel < reference_pixels; pixel++)
        for (size_t channel = 0; channel < 3; channel++)
            interleaved[pixel * 3 + channel] =
                fixture_data->reference_u8[channel * reference_pixels + pixel];
    char path[] = "/tmp/h3_edit_text_reference_XXXXXX";
    int descriptor = mkstemp(path);
    require(descriptor >= 0, "cannot create a temporary file");
    close(descriptor);
    if (!h3_ffmpeg_write_png_rgb24(path, interleaved,
                                   fixture_data->reference_width,
                                   fixture_data->reference_height, error,
                                   sizeof(error))) {
        unlink(path);
        printf("%s (d) FFmpeg resize skipped: %s\n", fixture_data->name,
               error);
    } else {
        float *scaled = NULL;
        int ok = h3_ffmpeg_read_image_f32(path, fixture_data->width,
                                          fixture_data->height,
                                          H3_IMAGE_FIT_STRETCH, &scaled,
                                          error, sizeof(error));
        unlink(path);
        require(ok, error);
        double square_error = 0.0, square_value = 0.0, worst = 0.0;
        for (size_t index = 0; index < count; index++) {
            double delta = (double)scaled[index] * 255.0 -
                           (double)fixture_data->vision_u8[index];
            if (fabs(delta) > worst) worst = fabs(delta);
            square_error += delta * delta;
            square_value += (double)fixture_data->vision_u8[index] *
                            (double)fixture_data->vision_u8[index];
        }
        double relative = sqrt(square_error / square_value);
        printf("%s (d) FFmpeg resize vs vision_u8: pixel rel-L2 %.6g, max "
               "abs %.6g of 255\n", fixture_data->name, relative, worst);
        free(scaled);
    }
    free(interleaved);

    if (!different) {
        printf("%s (d) Lanczos pixels are the fixture's: embeddings equal "
               "(c)\n", fixture_data->name);
    } else {
        float *pixels = pixels_from_u8(lanczos, count);
        h3_vision_output vision;
        encode_vision(encoder_directory, pixels, fixture_data->height,
                      fixture_data->width, &vision);
        h3_text_embedding text;
        memset(&text, 0, sizeof(text));
        require(h3_multimodal_encode_edit_bf16(
                    tokenizer, encoder_directory, "h3_shaders.metal",
                    fixture_data->prompt, &vision, 1, progress, NULL, &text,
                    error, sizeof(error)), error);
        compare_embeddings("(d) prompt_embeds from own Lanczos pixels",
                           fixture_data, &text, DRIFT_EMBEDDING_BOUND,
                           DRIFT_EMBEDDING_BOUND);
        h3_text_embedding_free(&text);
        h3_vision_output_free(&vision);
        free(pixels);
    }
    free(lanczos);
}

static void run_case(const h3_tokenizer *tokenizer,
                     const char *encoder_directory, const char *name,
                     const char *path, int resampling) {
    fixture_case fixture_data;
    memset(&fixture_data, 0, sizeof(fixture_data));
    load_case(&fixture_data, name, path);
    printf("%s: prompt \"%s\", %zu text rows, vision %dx%d (%d tokens)\n",
           name, fixture_data.prompt, fixture_data.tokens, fixture_data.width,
           fixture_data.height, fixture_data.grid_h * fixture_data.grid_w / 4);
    check_presentation(tokenizer, &fixture_data);
    float *pixels = pixels_from_u8(
        fixture_data.vision_u8,
        (size_t)3 * (size_t)fixture_data.height * (size_t)fixture_data.width);
    h3_vision_output vision;
    encode_vision(encoder_directory, pixels, fixture_data.height,
                  fixture_data.width, &vision);
    free(pixels);
    check_vision(&fixture_data, &vision);
    check_embeddings(tokenizer, encoder_directory, &fixture_data, &vision);
    h3_vision_output_free(&vision);
    if (resampling) check_resampling(tokenizer, encoder_directory,
                                     &fixture_data);
    free_case(&fixture_data);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        printf("skip: usage %s MODEL_DIR TEXT_FIXTURE TEXT_SMALL_FIXTURE\n",
               argv[0]);
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL_DIR h3_image_edit_text.safetensors "
                "h3_image_edit_text_small.safetensors\n", argv[0]);
        return 2;
    }
    for (int index = 2; index < 4; index++)
        if (access(argv[index], R_OK) != 0) {
            printf("skip: fixture %s is not installed\n", argv[index]);
            return 0;
        }
    char tokenizer_path[1024], encoder_directory[1024], error[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer/tokenizer.json",
             argv[1]);
    snprintf(encoder_directory, sizeof(encoder_directory), "%s/text_encoder",
             argv[1]);
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                sizeof(error));
    if (!tokenizer) die(error);
    run_case(tokenizer, encoder_directory, "small", argv[3], 0);
    run_case(tokenizer, encoder_directory, "1024-token", argv[2], 1);
    h3_tokenizer_free(tokenizer);
    puts("ok: edit text presentation, vision tower and prompt embeddings "
         "match the diffusers reference");
    return 0;
}
