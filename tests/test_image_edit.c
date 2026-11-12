#include "h3.h"
#include "h3_ffmpeg.h"
#include "h3_image.h"
#include "h3_image_edit.h"
#include "h3_image_pipeline.h"
#include "h3_image_vae_encode.h"
#include "h3_internal.h"
#include "h3_multimodal.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static char directory[256];

static void make_directory(void) {
    const char *base = getenv("TMPDIR");
    snprintf(directory, sizeof(directory), "%s/h3_image_edit_XXXXXX",
             base && *base ? base : "/tmp");
    CHECK(mkdtemp(directory) != NULL);
}

static void remove_directory(void) {
    char command[512];
    snprintf(command, sizeof(command), "rm -rf '%s'", directory);
    if (system(command) != 0) fprintf(stderr, "cannot remove %s\n", directory);
}

static char *path_in(const char *name) {
    static char paths[16][512];
    static int next;
    char *path = paths[next++ % 16];
    snprintf(path, 512, "%s/%s", directory, name);
    return path;
}

/* A planar [3][h][w] test picture: channel c at (x, y) is a fixed function. */
static uint8_t *picture(int width, int height) {
    uint8_t *planar = malloc((size_t)3 * (size_t)width * (size_t)height);
    CHECK(planar != NULL);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                planar[((size_t)c * (size_t)height + (size_t)y) *
                       (size_t)width + (size_t)x] =
                    (uint8_t)((x * 3 + y * 5 + c * 40) % 256);
    return planar;
}

static void write_png(const char *path, const uint8_t *planar, int width,
                      int height) {
    size_t area = (size_t)width * (size_t)height;
    uint8_t *interleaved = malloc(area * 3);
    char error[256];
    CHECK(interleaved != NULL);
    for (size_t pixel = 0; pixel < area; pixel++)
        for (size_t c = 0; c < 3; c++)
            interleaved[3 * pixel + c] = planar[c * area + pixel];
    CHECK(h3_ffmpeg_write_png_rgb24(path, interleaved, width, height, error,
                                    sizeof(error)));
    free(interleaved);
}

static uint32_t crc32_of(const uint8_t *bytes, size_t count) {
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < count; index++) {
        crc ^= bytes[index];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

/* Insert an eXIf chunk carrying the orientation right after the IHDR. */
static void add_orientation(const char *path, int orientation) {
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *png = malloc((size_t)size);
    CHECK(png && fread(png, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    uint8_t exif[26] = {'M', 'M', 0, 42, 0, 0, 0, 8, 0, 1,
                        0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    exif[18] = (uint8_t)(orientation >> 8);
    exif[19] = (uint8_t)orientation;
    uint8_t chunk[4 + 4 + 26 + 4] = {0, 0, 0, 26, 'e', 'X', 'I', 'f'};
    memcpy(chunk + 8, exif, sizeof(exif));
    uint32_t crc = crc32_of(chunk + 4, 4 + sizeof(exif));
    for (int shift = 0; shift < 4; shift++)
        chunk[34 + shift] = (uint8_t)(crc >> (24 - 8 * shift));
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(png, 1, 33, file) == 33);
    CHECK(fwrite(chunk, 1, sizeof(chunk), file) == sizeof(chunk));
    CHECK(fwrite(png + 33, 1, (size_t)size - 33, file) == (size_t)size - 33);
    fclose(file);
    free(png);
}

static void corner(const uint8_t *planar, int width, int height, int x, int y,
                   uint8_t rgb[3]) {
    for (size_t c = 0; c < 3; c++)
        rgb[c] = planar[(c * (size_t)height + (size_t)y) * (size_t)width +
                        (size_t)x];
}

static int same_color(const uint8_t *got, uint8_t red, uint8_t green,
                      uint8_t blue) {
    return got[0] == red && got[1] == green && got[2] == blue;
}

/* A 64x32 picture with four coloured quadrants, stored with an EXIF
 * orientation: the reader returns the upright image. */
static void test_exif_orientation(void) {
    enum { W = 64, H = 32 };
    uint8_t *source = malloc((size_t)3 * W * H);
    CHECK(source != NULL);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int right = x >= W / 2, bottom = y >= H / 2;
            /* top-left red, top-right green, bottom-left blue, bottom-right
             * white */
            uint8_t color[3] = {
                (uint8_t)(!right && !bottom ? 255 : bottom && right ? 255 : 0),
                (uint8_t)(right && !bottom ? 255 : bottom && right ? 255 : 0),
                (uint8_t)(!right && bottom ? 255 : bottom && right ? 255 : 0)};
            for (int c = 0; c < 3; c++)
                source[((size_t)c * H + (size_t)y) * W + (size_t)x] = color[c];
        }
    const char *path = path_in("oriented.png");
    char error[256];
    int width, height;
    uint8_t *planar, rgb[3];

    write_png(path, source, W, H);
    CHECK(h3_ffmpeg_read_image_u8(path, &width, &height, &planar, error,
                                  sizeof(error)));
    CHECK(width == W && height == H && !memcmp(planar, source, (size_t)3 * W * H));
    free(planar);

    /* 6: rotate 90 degrees clockwise to display. The upright image is 32x64
     * and its top-left is the stored bottom-left. */
    add_orientation(path, 6);
    CHECK(h3_ffmpeg_read_image_u8(path, &width, &height, &planar, error,
                                  sizeof(error)));
    CHECK(width == H && height == W);
    corner(planar, width, height, 0, 0, rgb);
    CHECK(same_color(rgb, 0, 0, 255));
    corner(planar, width, height, width - 1, 0, rgb);
    CHECK(same_color(rgb, 255, 0, 0));
    corner(planar, width, height, 0, height - 1, rgb);
    CHECK(same_color(rgb, 255, 255, 255));
    corner(planar, width, height, width - 1, height - 1, rgb);
    CHECK(same_color(rgb, 0, 255, 0));
    free(planar);

    /* 3: rotate 180 degrees. */
    write_png(path, source, W, H);
    add_orientation(path, 3);
    CHECK(h3_ffmpeg_read_image_u8(path, &width, &height, &planar, error,
                                  sizeof(error)));
    CHECK(width == W && height == H);
    corner(planar, width, height, 0, 0, rgb);
    CHECK(same_color(rgb, 255, 255, 255));
    corner(planar, width, height, width - 1, height - 1, rgb);
    CHECK(same_color(rgb, 255, 0, 0));
    free(planar);

    CHECK(!h3_ffmpeg_read_image_u8(path_in("missing.png"), &width, &height,
                                   &planar, error, sizeof(error)));
    CHECK(error[0] != '\0' && planar == NULL);
    free(source);
}

static void geometry_of(int width, int height,
                        h3_image_reference_geometry *geometry) {
    CHECK(h3_image_edit_reference_geometry(width, height, geometry));
}

static int near_constant(const uint8_t *planar, size_t count, int value) {
    for (size_t index = 0; index < count; index++)
        if (abs((int)planar[index] - value) > 1) return 0;
    return 1;
}

static void test_edit_copies(void) {
    h3_image_reference_geometry g;
    uint8_t *vae, *vision;

    /* An upscale of 256x256 to 1024x1024 for both consumers, last element
     * included: the buffers hold the whole planar picture. */
    uint8_t *flat = malloc((size_t)3 * 256 * 256);
    CHECK(flat != NULL);
    memset(flat, 77, (size_t)3 * 256 * 256);
    geometry_of(256, 256, &g);
    CHECK(h3_image_edit_copies(flat, 256, 256, &g, &vae, &vision));
    CHECK(near_constant(vae, (size_t)3 * 1024 * 1024, 77));
    CHECK(near_constant(vision, (size_t)3 * 1024 * 1024, 77));
    free(vae);
    free(vision);
    free(flat);

    /* 250x370: both copies 1024x1504 (scale 4.096). */
    flat = malloc((size_t)3 * 250 * 370);
    CHECK(flat != NULL);
    memset(flat, 200, (size_t)3 * 250 * 370);
    geometry_of(250, 370, &g);
    CHECK(g.vae_w == 1024 && g.vae_h == 1504 && g.vision_w == 1024 &&
          g.vision_h == 1504);
    CHECK(h3_image_edit_copies(flat, 250, 370, &g, &vae, &vision));
    CHECK(near_constant(vae, (size_t)3 * 1024 * 1504, 200));
    CHECK(near_constant(vision, (size_t)3 * 1024 * 1504, 200));
    free(vae);
    free(vision);
    free(flat);

    /* 1500x1024: the VAE copy keeps the size; the vision copy goes through
     * smart_resize to 1504x1024. */
    flat = malloc((size_t)3 * 1500 * 1024);
    CHECK(flat != NULL);
    memset(flat, 9, (size_t)3 * 1500 * 1024);
    geometry_of(1500, 1024, &g);
    CHECK(g.vae_w == 1500 && g.vision_w == 1504 && g.vision_h == 1024);
    CHECK(h3_image_edit_copies(flat, 1500, 1024, &g, &vae, &vision));
    CHECK(!memcmp(vae, flat, (size_t)3 * 1500 * 1024));
    CHECK(near_constant(vision, (size_t)3 * 1504 * 1024, 9));
    free(vae);
    free(vision);
    free(flat);

    /* 1024x1024 is its own copy for both. */
    flat = picture(1024, 1024);
    geometry_of(1024, 1024, &g);
    CHECK(h3_image_edit_copies(flat, 1024, 1024, &g, &vae, &vision));
    CHECK(!memcmp(vae, flat, (size_t)3 * 1024 * 1024));
    CHECK(!memcmp(vision, flat, (size_t)3 * 1024 * 1024));
    /* Without a vision output only the VAE copy is made, and it is the same. */
    uint8_t *vae_only = NULL;
    CHECK(h3_image_edit_copies(flat, 1024, 1024, &g, &vae_only, NULL));
    CHECK(!memcmp(vae_only, vae, (size_t)3 * 1024 * 1024));
    free(vae_only);
    geometry_of(250, 370, &g);
    uint8_t *small = picture(250, 370);
    uint8_t *small_vae, *small_vision;
    CHECK(h3_image_edit_copies(small, 250, 370, &g, &small_vae,
                               &small_vision));
    CHECK(h3_image_edit_copies(small, 250, 370, &g, &vae_only, NULL));
    CHECK(vae_only != NULL &&
          !memcmp(vae_only, small_vae, (size_t)3 * (size_t)g.vae_w *
                                       (size_t)g.vae_h));
    free(vae_only);
    free(small_vae);
    free(small_vision);
    free(small);
    free(vae);
    free(vision);
    free(flat);

    CHECK(!h3_image_edit_copies(NULL, 256, 256, &g, &vae, &vision));
}

static void test_condition_rows(void) {
    enum { H = 4, W = 6, C = 24 };
    float latents[C * H * W], noise[C * H * W], rows[C * H * W];
    for (int index = 0; index < C * H * W; index++) {
        latents[index] = (float)(index % 17) * 0.25f - 2.0f;
        noise[index] = (float)(index % 13) * 0.5f - 3.0f;
    }
    CHECK(h3_image_edit_condition_rows(latents, noise, H, W, rows));
    const float kept = 0.999f, mixed = 1.0f - 0.999f;
    CHECK(mixed != 0.001f);
    /* Row (i, j) holds, per channel, the 2x2 block (dh, dw). */
    for (int i = 0; i < H / 2; i++)
        for (int j = 0; j < W / 2; j++)
            for (int c = 0; c < C; c++)
                for (int dh = 0; dh < 2; dh++)
                    for (int dw = 0; dw < 2; dw++) {
                        size_t source = ((size_t)c * H + (size_t)(2 * i + dh)) *
                                        W + (size_t)(2 * j + dw);
                        float expected = kept * latents[source] +
                                         mixed * noise[source];
                        size_t at = ((size_t)(i * (W / 2) + j) * C +
                                     (size_t)c) * 4 + (size_t)(dh * 2 + dw);
                        CHECK(fabsf(rows[at] - expected) <= 1e-6f);
                    }
    CHECK(!h3_image_edit_condition_rows(latents, noise, 3, W, rows));
    CHECK(!h3_image_edit_condition_rows(latents, noise, H, 5, rows));
    CHECK(!h3_image_edit_condition_rows(NULL, noise, H, W, rows));
}

static h3_ctx *fake_context(const char *model_dir) {
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    ctx->image = 1;
    ctx->model_dir = strdup(model_dir);
    CHECK(ctx->model_dir != NULL);
    return ctx;
}

/* The request is refused with `expected` in the message. The model directory
 * holds no weights (at most a tokenizer), so a load attempt would fail with a
 * different message: a matching text means nothing was loaded. */
static void refused(h3_ctx *ctx, const char *prompt,
                    const h3_image_params *params, const char *expected) {
    h3_result *result = h3_generate_image(ctx, prompt, params);
    if (result || !strstr(h3_last_error(ctx), expected)) {
        fprintf(stderr, "FAIL: expected \"%s\", got \"%s\"\n", expected,
                result ? "success" : h3_last_error(ctx));
        exit(1);
    }
    tests_run++;
}

static void test_limits_before_loading(const char *tokenizer_json) {
    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "%s/model", directory);
    h3_ctx *ctx = fake_context(model_dir);
    h3_image_params params = H3_IMAGE_PARAMS_DEFAULT;
    h3_image_reference references[H3_IMAGE_MAX_REFERENCES + 1];
    params.references = references;

    uint8_t *small = picture(16, 40);
    write_png(path_in("small.png"), small, 16, 40);
    free(small);
    uint8_t *ok = picture(64, 64);
    write_png(path_in("ok.png"), ok, 64, 64);
    free(ok);
    uint8_t *thin = picture(32, 7000);
    write_png(path_in("thin.png"), thin, 32, 7000);
    free(thin);
    uint8_t *big = picture(1024, 1024);
    write_png(path_in("big.png"), big, 1024, 1024);
    free(big);
    const char *ok_path = path_in("ok.png");

    params.reference_count = H3_IMAGE_MAX_REFERENCES + 1;
    for (size_t index = 0; index < params.reference_count; index++)
        references[index].path = ok_path;
    refused(ctx, "<Picture 1>", &params, "at most 8 reference images");

    params.reference_count = 2;
    references[1].path = "";
    refused(ctx, "<Picture 1>", &params, "reference 2 has an empty path");
    references[1].path = ok_path;
    params.references = NULL;
    refused(ctx, "<Picture 1>", &params, "reference images are required");
    params.references = references;

    params.width = 100;
    params.height = 0;
    refused(ctx, "<Picture 1>", &params, "image width and height");
    params.width = params.height = 0;

    references[0].path = path_in("missing.png");
    refused(ctx, "<Picture 1>", &params, "reference 1: ");
    references[0].path = path_in("small.png");
    refused(ctx, "<Picture 1>", &params,
            "reference 1 is 16x40 pixels: both sides must be at least 32");
    references[0].path = ok_path;
    references[1].path = path_in("small.png");
    refused(ctx, "<Picture 1> <Picture 2>", &params,
            "reference 2 is 16x40 pixels");
    references[1].path = path_in("thin.png");
    refused(ctx, "<Picture 1> <Picture 2>", &params, "aspect ratio is above 200");

    /* 8192x4096 is the largest canvas: any reference exceeds the row cap. */
    references[0].path = path_in("big.png");
    params.reference_count = 1;
    params.width = 8192;
    params.height = 4096;
    refused(ctx, "<Picture 1>", &params,
            "reference 1 brings the reference and image rows to 33792, above "
            "32768");

    /* No --size: the canvas of a 8300x4200 reference is over the cap, and the
     * message says where the size came from. */
    uint8_t *wide = malloc((size_t)3 * 8300 * 4200);
    CHECK(wide != NULL);
    memset(wide, 90, (size_t)3 * 8300 * 4200);
    write_png(path_in("wide.png"), wide, 8300, 4200);
    free(wide);
    references[0].path = path_in("wide.png");
    params.width = params.height = 0;
    refused(ctx, "<Picture 1>", &params,
            "the size was taken from reference 1; pass --size");

    if (tokenizer_json) {
        /* A canvas of 7936x4096 (31744 rows) and one 1024x1024 reference make
         * 32768 rows, the cap; a 4096-token prompt, the label and the 1024
         * vision tokens bring the sequence to 37895. */
        char tokenizer_directory[512], tokenizer_link[512];
        snprintf(tokenizer_directory, sizeof(tokenizer_directory),
                 "%s/tokenizer", model_dir);
        snprintf(tokenizer_link, sizeof(tokenizer_link),
                 "%s/tokenizer.json", tokenizer_directory);
        char *absolute = realpath(tokenizer_json, NULL);
        CHECK(absolute != NULL);
        char command[1536];
        snprintf(command, sizeof(command), "mkdir -p '%s'",
                 tokenizer_directory);
        CHECK(system(command) == 0);
        CHECK(symlink(absolute, tokenizer_link) == 0);
        free(absolute);
        char *prompt = malloc(16 + 5 * 5000);
        CHECK(prompt != NULL);
        strcpy(prompt, "<Picture 1>");
        for (int word = 0; word < 5000; word++) strcat(prompt, " word");
        references[0].path = path_in("big.png");
        params.width = 7936;
        params.height = 4096;
        refused(ctx, prompt, &params,
                "packed sequence of 37895 rows is above 36864");
        free(prompt);
    }
    h3_free(ctx);
}

/* A real VAE encoder run through the edit pipeline: the hooks stop it before
 * any text encoder or transformer loads. The condition rows equal the
 * hand-built composition of the public pieces, reference by reference, and
 * the hook inputs reproduce them bit for bit. */
static void test_reference_rows(const char *model_dir) {
    enum { COUNT = 2 };
    const int sizes[COUNT][2] = {{256, 256}, {300, 200}};
    h3_image_reference references[COUNT];
    h3_image_reference_geometry geometry[COUNT];
    uint8_t *vae_copies[COUNT], *vision_copies[COUNT];
    float *posterior[COUNT], *aug[COUNT];
    size_t offsets[COUNT + 1] = {0};
    char error[512];
    const uint64_t seed = 7;
    char *vae_directory = malloc(strlen(model_dir) + 32);
    CHECK(vae_directory != NULL);
    sprintf(vae_directory, "%s/vae_diffusers", model_dir);

    for (int index = 0; index < COUNT; index++) {
        uint8_t *source = picture(sizes[index][0], sizes[index][1]);
        char name[32];
        snprintf(name, sizeof(name), "reference%d.png", index);
        write_png(path_in(name), source, sizes[index][0], sizes[index][1]);
        references[index].path = strdup(path_in(name));
        geometry_of(sizes[index][0], sizes[index][1], &geometry[index]);
        CHECK(h3_image_edit_copies(source, sizes[index][0], sizes[index][1],
                                   &geometry[index], &vae_copies[index],
                                   &vision_copies[index]));
        free(source);
        offsets[index + 1] = offsets[index] +
                             (size_t)geometry[index].dit_rows * 96;
        const h3_image_reference_geometry *g = &geometry[index];
        size_t posterior_count = (size_t)24 * (size_t)(g->crop_h / 16) *
                                 (size_t)(g->crop_w / 16);
        size_t aug_count = (size_t)24 * (size_t)g->latent_h *
                           (size_t)g->latent_w;
        posterior[index] = malloc(posterior_count * sizeof(float));
        aug[index] = malloc(aug_count * sizeof(float));
        CHECK(posterior[index] && aug[index]);
        h3_image_noise(h3_image_edit_stream(seed, 1, (size_t)index),
                       g->crop_h / 16, g->crop_w / 16, posterior[index]);
        h3_image_noise(h3_image_edit_stream(seed, 2, (size_t)index),
                       g->latent_h, g->latent_w, aug[index]);
    }
    float *expected = malloc(offsets[COUNT] * sizeof(float));
    CHECK(expected != NULL);
    for (int index = 0; index < COUNT; index++) {
        const h3_image_reference_geometry *g = &geometry[index];
        h3_reference_latents latents;
        CHECK(h3_image_vae_encode(vae_directory, "h3_shaders.metal",
                                  vae_copies[index], g->vae_h, g->vae_w, NULL,
                                  h3_image_edit_stream(seed, 1, (size_t)index),
                                  NULL, NULL, &latents, error, sizeof(error)));
        CHECK(latents.latent_height == g->latent_h &&
              latents.latent_width == g->latent_w);
        CHECK(h3_image_edit_condition_rows(latents.latents, aug[index],
                                           g->latent_h, g->latent_w,
                                           expected + offsets[index]));
        h3_reference_latents_free(&latents);
    }

    h3_ctx *ctx = h3_load_dir(model_dir);
    CHECK(ctx != NULL);
    h3_image_params params = H3_IMAGE_PARAMS_DEFAULT;
    params.width = params.height = 0;
    params.min_generation_area = 1024;
    params.seed = seed;
    params.references = references;
    params.reference_count = COUNT;
    const char *prompt = "<Picture 2> in the style of <Picture 1>";

    h3_image_hooks hooks;
    memset(&hooks, 0, sizeof(hooks));
    hooks.references_only = 1;
    h3_result *result = h3_generate_image_with_hooks(ctx, prompt, &params,
                                                     &hooks);
    CHECK(result != NULL);
    CHECK(result->width == 256 && result->height == 256);
    CHECK(hooks.condition_elements == offsets[COUNT]);
    CHECK(hooks.condition_rows != NULL &&
          !memcmp(hooks.condition_rows, expected,
                  offsets[COUNT] * sizeof(float)));
    CHECK(hooks.latent == NULL && hooks.rgb == NULL);

    char *tokenizer_path = malloc(strlen(model_dir) + 32);
    CHECK(tokenizer_path != NULL);
    sprintf(tokenizer_path, "%s/tokenizer/tokenizer.json", model_dir);
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                sizeof(error));
    CHECK(tokenizer != NULL);
    int grid_h[COUNT], grid_w[COUNT];
    for (int index = 0; index < COUNT; index++) {
        grid_h[index] = geometry[index].vision_h / 16;
        grid_w[index] = geometry[index].vision_w / 16;
    }
    h3_edit_presentation presentation;
    CHECK(h3_multimodal_edit_presentation(tokenizer, prompt, grid_h, grid_w,
                                          COUNT, &presentation, error,
                                          sizeof(error)));
    CHECK(hooks.text_tokens == presentation.count);
    h3_edit_presentation_free(&presentation);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    free(hooks.condition_rows);
    h3_result_free(result);

    /* Hook inputs: the stream noises and VAE copies that differ from h3's
     * own (inverted), so a hook that is ignored gives other rows. */
    uint8_t *inverted[COUNT];
    float *expected_inverted = malloc(offsets[COUNT] * sizeof(float));
    CHECK(expected_inverted != NULL);
    for (int index = 0; index < COUNT; index++) {
        const h3_image_reference_geometry *g = &geometry[index];
        size_t bytes = (size_t)3 * (size_t)g->vae_w * (size_t)g->vae_h;
        inverted[index] = malloc(bytes);
        CHECK(inverted[index] != NULL);
        for (size_t at = 0; at < bytes; at++)
            inverted[index][at] = (uint8_t)(255 - vae_copies[index][at]);
        h3_reference_latents latents;
        CHECK(h3_image_vae_encode(vae_directory, "h3_shaders.metal",
                                  inverted[index], g->vae_h, g->vae_w,
                                  posterior[index], 0, NULL, NULL, &latents,
                                  error, sizeof(error)));
        CHECK(h3_image_edit_condition_rows(latents.latents, aug[index],
                                           g->latent_h, g->latent_w,
                                           expected_inverted + offsets[index]));
        h3_reference_latents_free(&latents);
    }
    CHECK(memcmp(expected, expected_inverted, offsets[COUNT] * sizeof(float)));
    const uint8_t *vae_hook[COUNT] = {inverted[0], inverted[1]};
    const uint8_t *vision_hook[COUNT] = {vision_copies[0], vision_copies[1]};
    const float *posterior_hook[COUNT] = {posterior[0], posterior[1]};
    const float *aug_hook[COUNT] = {aug[0], aug[1]};
    memset(&hooks, 0, sizeof(hooks));
    hooks.references_only = 1;
    hooks.vae_pixels = vae_hook;
    hooks.vision_pixels = vision_hook;
    hooks.reference_posterior_noise = posterior_hook;
    hooks.reference_aug_noise = aug_hook;
    params.seed = seed + 1000;
    result = h3_generate_image_with_hooks(ctx, prompt, &params, &hooks);
    CHECK(result != NULL);
    CHECK(hooks.condition_elements == offsets[COUNT]);
    CHECK(hooks.condition_rows != NULL &&
          !memcmp(hooks.condition_rows, expected_inverted,
                  offsets[COUNT] * sizeof(float)));
    free(hooks.condition_rows);
    h3_result_free(result);

    for (int index = 0; index < COUNT; index++) {
        free(inverted[index]);
        free((void *)references[index].path);
        free(vae_copies[index]);
        free(vision_copies[index]);
        free(posterior[index]);
        free(aug[index]);
    }
    free(expected);
    free(expected_inverted);
    free(vae_directory);
    h3_free(ctx);
}

int main(int argc, char **argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [TOKENIZER_JSON [MODEL_DIR]]\n", argv[0]);
        return 2;
    }
    make_directory();
    test_exif_orientation();
    test_edit_copies();
    test_condition_rows();
    test_limits_before_loading(argc > 1 ? argv[1] : NULL);
    if (argc > 2) test_reference_rows(argv[2]);
    else printf("skip: the reference VAE encoder run needs MODEL_DIR\n");
    remove_directory();
    printf("ok: %d image edit host checks\n", tests_run);
    return 0;
}
