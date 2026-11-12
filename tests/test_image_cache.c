#include "h3_ffmpeg.h"
#include "h3_image.h"
#include "h3_image_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
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
    snprintf(directory, sizeof(directory), "%s/h3_image_cache_XXXXXX",
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

static void write_file(const char *path, const void *bytes, size_t count) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(bytes, 1, count, file) == count);
    CHECK(fclose(file) == 0);
}

static uint8_t *read_file(const char *path, size_t *count) {
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    long size = ftell(file);
    CHECK(size > 0);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    uint8_t *bytes = malloc((size_t)size);
    CHECK(bytes != NULL);
    CHECK(fread(bytes, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    *count = (size_t)size;
    return bytes;
}

static h3_ctx *fake_context(const char *model_dir) {
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    ctx->image = 1;
    ctx->model_dir = strdup(model_dir);
    CHECK(ctx->model_dir != NULL);
    ctx->model.text_encoder.bytes = 1000;
    ctx->model.text_encoder.files = 14;
    ctx->model.text_encoder.tensors = 400;
    return ctx;
}

static uint8_t *picture(int width, int height, int salt) {
    size_t count = (size_t)3 * (size_t)width * (size_t)height;
    uint8_t *planar = malloc(count);
    CHECK(planar != NULL);
    for (size_t index = 0; index < count; index++)
        planar[index] = (uint8_t)((index * 7 + (size_t)salt) % 251);
    return planar;
}

static h3_text_embedding make_text(size_t tokens, int salt, int with_tags) {
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    text.tokens = tokens;
    text.width = H3_TEXT_HIDDEN_SIZE;
    text.values = malloc(tokens * text.width * sizeof(*text.values));
    CHECK(text.values != NULL);
    for (size_t index = 0; index < tokens * text.width; index++)
        text.values[index] = (uint16_t)(index * 31 + (size_t)salt);
    if (with_tags) {
        text.tags = malloc(tokens);
        CHECK(text.tags != NULL);
        for (size_t index = 0; index < tokens; index++)
            text.tags[index] = (uint8_t)((index + (size_t)salt) % 2);
    }
    return text;
}

static int same_text(const h3_text_embedding *a, const h3_text_embedding *b) {
    return a->tokens == b->tokens && a->width == b->width &&
           !a->tags == !b->tags &&
           !memcmp(a->values, b->values,
                   a->tokens * a->width * sizeof(*a->values)) &&
           (!a->tags || !memcmp(a->tags, b->tags, a->tokens));
}

static char *key_of(h3_ctx *ctx, const char *prompt, size_t count,
                    uint8_t *const *sources, const int *widths,
                    const int *heights) {
    int vision_w[2] = {64, 96}, vision_h[2] = {64, 32};
    char *key = h3_image_cache_key(ctx, prompt, count, widths, heights,
                                   (const uint8_t *const *)sources, vision_w,
                                   vision_h);
    CHECK(key != NULL);
    return key;
}

static void set_age(const char *path, int seconds_ago) {
    struct timeval times[2];
    gettimeofday(&times[0], NULL);
    times[0].tv_sec -= seconds_ago;
    times[1] = times[0];
    CHECK(utimes(path, times) == 0);
}

static void test_key(void) {
    char model[512];
    snprintf(model, sizeof(model), "%s/model", directory);
    CHECK(mkdir(model, 0755) == 0);
    CHECK(mkdir(path_in("model/tokenizer"), 0755) == 0);
    write_file(path_in("model/tokenizer/tokenizer.json"), "{\"a\":1}", 7);
    h3_ctx *ctx = fake_context(model);

    int widths[2] = {16, 24}, heights[2] = {8, 4};
    uint8_t *first = picture(16, 8, 1);
    uint8_t *second = picture(24, 4, 2);
    uint8_t *sources[2] = {first, second};
    char *key = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                       heights);

    /* The same pixels again, as other memory. */
    uint8_t *first_copy = malloc((size_t)3 * 16 * 8);
    CHECK(first_copy != NULL);
    memcpy(first_copy, first, (size_t)3 * 16 * 8);
    uint8_t *copies[2] = {first_copy, second};
    char *same = key_of(ctx, "make <Picture 1> red", 2, copies, widths,
                        heights);
    CHECK(!strcmp(key, same));
    free(same);

    first_copy[100] ^= 1;
    char *changed = key_of(ctx, "make <Picture 1> red", 2, copies, widths,
                           heights);
    CHECK(strcmp(key, changed));
    free(changed);
    first_copy[100] ^= 1;

    uint8_t *swapped[2] = {second, first};
    int swapped_w[2] = {24, 16}, swapped_h[2] = {4, 8};
    char *order = key_of(ctx, "make <Picture 1> red", 2, swapped, swapped_w,
                         swapped_h);
    CHECK(strcmp(key, order));
    free(order);

    char *prompt = key_of(ctx, "make <Picture 1> blue", 2, sources, widths,
                          heights);
    CHECK(strcmp(key, prompt));
    free(prompt);

    char *one = key_of(ctx, "make <Picture 1> red", 1, sources, widths,
                       heights);
    CHECK(strcmp(key, one));
    free(one);

    char *text_only = key_of(ctx, "make <Picture 1> red", 0, NULL, NULL, NULL);
    CHECK(strcmp(key, text_only));
    free(text_only);

    write_file(path_in("model/tokenizer/tokenizer.json"), "{\"a\":2}", 7);
    char *tokenizer = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                             heights);
    CHECK(strcmp(key, tokenizer));
    free(tokenizer);
    write_file(path_in("model/tokenizer/tokenizer.json"), "{\"a\":1}", 7);

    ctx->model.text_encoder.tensors++;
    char *inventory = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                             heights);
    CHECK(strcmp(key, inventory));
    free(inventory);
    ctx->model.text_encoder.tensors--;

    char *again = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                         heights);
    CHECK(!strcmp(key, again));
    free(again);

    /* Everything that can change the embedding changes the key. */
    int other_vision_w[2] = {64, 96}, other_vision_h[2] = {64, 64};
    char *vision = h3_image_cache_key(ctx, "make <Picture 1> red", 2, widths,
                                      heights,
                                      (const uint8_t *const *)sources,
                                      other_vision_w, other_vision_h);
    CHECK(vision && strcmp(key, vision));
    free(vision);

    CHECK(mkdir(path_in("model/text_encoder"), 0755) == 0);
    write_file(path_in("model/text_encoder/a.safetensors"), "weights", 7);
    char *with_weights = key_of(ctx, "make <Picture 1> red", 2, sources,
                                widths, heights);
    CHECK(strcmp(key, with_weights));
    set_age(path_in("model/text_encoder/a.safetensors"), 5000);
    char *older = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                         heights);
    CHECK(strcmp(with_weights, older));
    write_file(path_in("model/text_encoder/b.safetensors"), "more", 4);
    char *more = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                        heights);
    CHECK(strcmp(older, more));
    CHECK(unlink(path_in("model/text_encoder/b.safetensors")) == 0);
    free(with_weights);
    free(older);
    free(more);

    CHECK(mkdir(path_in("model2"), 0755) == 0);
    h3_ctx *elsewhere = fake_context(path_in("model2"));
    char *model_key = key_of(elsewhere, "make <Picture 1> red", 2, sources,
                             widths, heights);
    CHECK(strcmp(key, model_key));
    free(model_key);
    free(elsewhere->model_dir);
    free(elsewhere);

    strcpy(ctx->device.name, "Apple M5");
    char *device = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                          heights);
    CHECK(strcmp(key, device));
    free(device);
    ctx->device.name[0] = '\0';

    unsetenv("H3_NAX");
    char *plain_env = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                             heights);
    setenv("H3_PROFILE", "1", 1);
    setenv("H3_CACHE_DIR", "/nowhere", 1);
    char *ignored = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                           heights);
    CHECK(!strcmp(plain_env, ignored));
    unsetenv("H3_PROFILE");
    unsetenv("H3_CACHE_DIR");
    setenv("H3_NAX", "0", 1);
    char *nax = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                       heights);
    CHECK(strcmp(plain_env, nax));
    unsetenv("H3_NAX");
    free(plain_env);
    free(ignored);
    free(nax);

    h3_image_cache_set_build_id("build one");
    char *build_one = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                             heights);
    h3_image_cache_set_build_id("build two");
    char *build_two = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                             heights);
    CHECK(strcmp(build_one, build_two));
    h3_image_cache_set_build_id(NULL);
    char *executable = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                              heights);
    CHECK(strcmp(build_one, executable) && strcmp(build_two, executable));
    free(build_one);
    free(build_two);
    free(executable);

    /* The shader file is read from the working directory. */
    char saved[1024];
    CHECK(getcwd(saved, sizeof(saved)) != NULL);
    CHECK(mkdir(path_in("shaders"), 0755) == 0);
    CHECK(chdir(path_in("shaders")) == 0);
    write_file("h3_shaders.metal", "kernel one", 10);
    char *shader_one = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                              heights);
    write_file("h3_shaders.metal", "kernel two", 10);
    char *shader_two = key_of(ctx, "make <Picture 1> red", 2, sources, widths,
                              heights);
    CHECK(chdir(saved) == 0);
    CHECK(strcmp(shader_one, shader_two));
    free(shader_one);
    free(shader_two);

    char expected_start[32];
    snprintf(expected_start, sizeof(expected_start), "edit-cache=%d|",
             H3_EDIT_CACHE_VERSION);
    CHECK(!strncmp(key, expected_start, strlen(expected_start)));
    free(key);
    free(first);
    free(second);
    free(first_copy);
    free(ctx->model_dir);
    free(ctx);
}

static void test_round_trip(void) {
    const char *cache = path_in("cache/nested");
    char file[1024];
    h3_text_embedding loaded;
    h3_text_embedding tagged = make_text(5, 3, 1);
    h3_text_embedding plain = make_text(4, 9, 0);

    CHECK(!h3_image_cache_load(cache, "key one", &loaded, file, sizeof(file)));
    CHECK(h3_image_cache_store(cache, "key one", &tagged,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(h3_image_cache_load(cache, "key one", &loaded, file, sizeof(file)));
    CHECK(same_text(&tagged, &loaded));
    h3_text_embedding_free(&loaded);

    CHECK(h3_image_cache_store(cache, "key two", &plain,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(h3_image_cache_load(cache, "key two", &loaded, file, sizeof(file)));
    CHECK(same_text(&plain, &loaded) && loaded.tags == NULL);
    h3_text_embedding_free(&loaded);

    /* One file per key: a stored key is replaced, another key is not. */
    h3_text_embedding other = make_text(2, 5, 1);
    CHECK(h3_image_cache_store(cache, "key one", &other,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(h3_image_cache_load(cache, "key one", &loaded, file, sizeof(file)));
    CHECK(same_text(&other, &loaded));
    h3_text_embedding_free(&loaded);
    CHECK(h3_image_cache_load(cache, "key two", &loaded, file, sizeof(file)));
    CHECK(same_text(&plain, &loaded));
    h3_text_embedding_free(&loaded);

    /* Damaged files are misses. */
    CHECK(h3_image_cache_path(cache, "key two", file, sizeof(file)));
    size_t size;
    uint8_t *good = read_file(file, &size);
    uint8_t *bad = malloc(size);
    CHECK(bad != NULL);
    const size_t lengths[] = {0, 7, 20, 60, size - 1};
    for (size_t index = 0; index < sizeof(lengths) / sizeof(lengths[0]);
         index++) {
        write_file(file, good, lengths[index]);
        CHECK(!h3_image_cache_load(cache, "key two", &loaded, file,
                                   sizeof(file)));
    }
    const size_t flips[] = {0, 4, 8, 16, 24, 40, 100, 5000, 0};
    for (size_t index = 0; index < sizeof(flips) / sizeof(flips[0]);
         index++) {
        memcpy(bad, good, size);
        bad[flips[index] < size ? flips[index] : size - 1] ^= 0x40;
        write_file(file, bad, size);
        CHECK(!h3_image_cache_load(cache, "key two", &loaded, file,
                                   sizeof(file)));
    }
    memcpy(bad, good, size);
    bad[size - 1] ^= 1;
    write_file(file, bad, size);
    CHECK(!h3_image_cache_load(cache, "key two", &loaded, file, sizeof(file)));

    /* The entry of "key two" copied to the file name of "key three": the key
     * inside the file differs, so it is a miss. */
    char other_file[1024];
    CHECK(h3_image_cache_path(cache, "key three", other_file,
                              sizeof(other_file)));
    write_file(other_file, good, size);
    CHECK(!h3_image_cache_load(cache, "key three", &loaded, other_file,
                               sizeof(other_file)));
    write_file(file, good, size);
    CHECK(h3_image_cache_load(cache, "key two", &loaded, file, sizeof(file)));
    h3_text_embedding_free(&loaded);

    /* A newer format number. */
    memcpy(bad, good, size);
    bad[4] = 2;
    write_file(file, bad, size);
    CHECK(!h3_image_cache_load(cache, "key two", &loaded, file, sizeof(file)));

    /* An unwritable directory is a failed store, not a crash. */
    write_file(path_in("blocker"), "x", 1);
    CHECK(!h3_image_cache_store(path_in("blocker/sub"), "key", &plain,
                                H3_EDIT_CACHE_DEFAULT_BYTES, file,
                                sizeof(file)));
    free(good);
    free(bad);
    h3_text_embedding_free(&tagged);
    h3_text_embedding_free(&plain);
    h3_text_embedding_free(&other);
}

static int exists(const char *path) {
    struct stat status;
    return stat(path, &status) == 0;
}

static void test_eviction(void) {
    const char *cache = path_in("evict");
    char files[5][1024], file[1024];
    char keys[5][16];
    h3_text_embedding text = make_text(3, 1, 1);
    h3_text_embedding loaded;
    struct stat status;
    uint64_t limit = 0;

    for (int index = 0; index < 5; index++) {
        snprintf(keys[index], sizeof(keys[index]), "entry %d", index);
        CHECK(h3_image_cache_path(cache, keys[index], files[index],
                                  sizeof(files[index])));
    }
    CHECK(h3_image_cache_store(cache, keys[0], &text,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(stat(files[0], &status) == 0);
    limit = 3 * (uint64_t)status.st_size;
    /* Entry 0 is the oldest, entry 3 the newest; the cap holds three. */
    set_age(files[0], 400);
    for (int index = 1; index < 4; index++) {
        CHECK(h3_image_cache_store(cache, keys[index], &text,
                                   H3_EDIT_CACHE_DEFAULT_BYTES, file,
                                   sizeof(file)));
        set_age(files[index], 400 - 100 * index);
    }
    /* A hit on entry 0 makes it the newest: entry 1 is now the oldest. */
    CHECK(h3_image_cache_load(cache, keys[0], &loaded, file, sizeof(file)));
    h3_text_embedding_free(&loaded);
    CHECK(h3_image_cache_store(cache, keys[4], &text, limit, file,
                               sizeof(file)));
    CHECK(exists(files[0]) && !exists(files[1]) && !exists(files[2]) &&
          exists(files[3]) && exists(files[4]));

    set_age(files[0], 300);
    set_age(files[3], 100);
    set_age(files[4], 50);
    h3_text_embedding other = make_text(3, 2, 1);
    CHECK(h3_image_cache_store(cache, keys[1], &other,
                               2 * (uint64_t)status.st_size, file,
                               sizeof(file)));
    CHECK(!exists(files[0]) && !exists(files[3]) && exists(files[4]) &&
          exists(files[1]));
    CHECK(h3_image_cache_store(cache, keys[2], &text, 1, file, sizeof(file)));
    CHECK(!exists(files[2]));
    h3_text_embedding_free(&text);
    h3_text_embedding_free(&other);
}

static void test_store_hygiene(void) {
    const char *cache = path_in("hygiene/sub");
    char file[1024];
    struct stat status;
    h3_text_embedding text = make_text(2, 4, 1);
    CHECK(!h3_image_cache_store("", "key", &text, H3_EDIT_CACHE_DEFAULT_BYTES,
                                file, sizeof(file)));
    CHECK(h3_image_cache_store(cache, "key", &text,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(stat(file, &status) == 0 && (status.st_mode & 0777) == 0600);
    CHECK(stat(cache, &status) == 0 && (status.st_mode & 0777) == 0700);

    /* A temporary file left by a crashed store goes after an hour. */
    char old_file[1024], new_file[1024];
    snprintf(old_file, sizeof(old_file), "%s/abc.h3ec.tmp.1", cache);
    snprintf(new_file, sizeof(new_file), "%s/abc.h3ec.tmp.2", cache);
    write_file(old_file, "x", 1);
    write_file(new_file, "x", 1);
    set_age(old_file, 7200);
    CHECK(h3_image_cache_store(cache, "key", &text,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    CHECK(stat(old_file, &status) != 0 && stat(new_file, &status) == 0);
    h3_text_embedding_free(&text);
}

static void test_limit(void) {
    unsetenv("H3_CACHE_BYTES");
    CHECK(h3_image_cache_limit() == H3_EDIT_CACHE_DEFAULT_BYTES);
    setenv("H3_CACHE_BYTES", "12345", 1);
    CHECK(h3_image_cache_limit() == 12345);
    setenv("H3_CACHE_BYTES", "-5", 1);
    CHECK(h3_image_cache_limit() == H3_EDIT_CACHE_DEFAULT_BYTES);
    setenv("H3_CACHE_BYTES", "12x", 1);
    CHECK(h3_image_cache_limit() == H3_EDIT_CACHE_DEFAULT_BYTES);
    setenv("H3_CACHE_BYTES", "0", 1);
    CHECK(h3_image_cache_limit() == H3_EDIT_CACHE_DEFAULT_BYTES);
    unsetenv("H3_CACHE_BYTES");
}

/* The fake model directory holds a tokenizer and nothing else, so any weight
 * load fails: a run that gets past the cache lookup stops at the reference VAE
 * encoder, which opens first. */
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

static void test_set_directory(void) {
    h3_ctx *ctx = fake_context("model");
    CHECK(ctx->cache_dir == NULL);
    h3_cache_set_directory(ctx, "first");
    CHECK(ctx->cache_dir && !strcmp(ctx->cache_dir, "first"));
    h3_cache_set_directory(ctx, "second");
    CHECK(!strcmp(ctx->cache_dir, "second"));
    h3_cache_set_directory(ctx, "");
    CHECK(ctx->cache_dir == NULL);
    h3_cache_set_directory(ctx, "third");
    h3_cache_set_directory(ctx, NULL);
    CHECK(ctx->cache_dir == NULL);
    h3_cache_set_directory(NULL, "ignored");
    free(ctx->model_dir);
    free(ctx);
}

static int at_vae_open(const h3_ctx *ctx) {
    return strstr(h3_last_error(ctx), "vae_diffusers") != NULL;
}

static void test_wiring(const char *tokenizer_json) {
    char model[512], cache[512], link[512];
    snprintf(model, sizeof(model), "%s/wire", directory);
    snprintf(cache, sizeof(cache), "%s/wire_cache", directory);
    snprintf(link, sizeof(link), "%s/tokenizer", model);
    CHECK(mkdir(model, 0755) == 0);
    CHECK(mkdir(link, 0755) == 0);
    snprintf(link, sizeof(link), "%s/tokenizer/tokenizer.json", model);
    char *absolute = realpath(tokenizer_json, NULL);
    CHECK(absolute != NULL);
    CHECK(symlink(absolute, link) == 0);
    free(absolute);

    uint8_t *pixels = picture(64, 64, 4);
    const char *reference_path = path_in("wire_reference.png");
    write_png(reference_path, pixels, 64, 64);
    free(pixels);
    int width, height;
    uint8_t *decoded = NULL;
    char error[256];
    CHECK(h3_ffmpeg_read_image_u8(reference_path, &width, &height, &decoded,
                                  error, sizeof(error)));
    h3_image_reference_geometry geometry;
    CHECK(h3_image_edit_reference_geometry(width, height, &geometry));

    h3_ctx *ctx = fake_context(model);
    const char *prompt = "Keep <Picture 1> as it is, at night.";
    const uint8_t *sources[1] = {decoded};
    int vision_w[1] = {geometry.vision_w}, vision_h[1] = {geometry.vision_h};
    char *edit_key = h3_image_cache_key(ctx, prompt, 1, &width, &height,
                                        sources, vision_w, vision_h);
    char *text_key = h3_image_cache_key(ctx, "a red fox", 0, NULL, NULL, NULL,
                                        NULL, NULL);
    CHECK(edit_key && text_key);
    h3_text_embedding plain = make_text(3, 8, 0);
    char file[1024];
    CHECK(h3_image_cache_store(cache, text_key, &plain,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));

    h3_image_reference reference = {reference_path};
    h3_image_params params = H3_IMAGE_PARAMS_DEFAULT;
    params.references = &reference;
    params.reference_count = 1;
    params.width = params.height = 0;
    params.min_generation_area = 1024;
    params.seed = 1;
    h3_image_hooks hooks;
    ctx->cache_dir = strdup(cache);
    CHECK(ctx->cache_dir != NULL);

    /* An empty cache: a lookup miss. The run reports the presentation length,
     * which an entry must match. */
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.cache_result == H3_CACHE_MISS && at_vae_open(ctx));
    size_t rows = hooks.text_tokens;
    CHECK(rows > 0);

    /* An entry of another length under the right key is not trusted. */
    h3_text_embedding wrong = make_text(rows + 1, 6, 1);
    CHECK(h3_image_cache_store(cache, edit_key, &wrong,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.cache_result == H3_CACHE_MISS && hooks.text.values == NULL &&
          at_vae_open(ctx));
    h3_text_embedding_free(&wrong);

    /* A hit returns the stored embedding before any weight loads, whatever
     * the seed and the size. */
    h3_text_embedding stored = make_text(rows, 6, 1);
    CHECK(h3_image_cache_store(cache, edit_key, &stored,
                               H3_EDIT_CACHE_DEFAULT_BYTES, file,
                               sizeof(file)));
    for (int run = 0; run < 3; run++) {
        params.seed = run == 0 ? 1 : 987654321;
        params.width = params.height = run == 2 ? 128 : 0;
        memset(&hooks, 0, sizeof(hooks));
        hooks.stop_after_lookup = 1;
        h3_result *result = h3_generate_image_with_hooks(ctx, prompt, &params,
                                                         &hooks);
        CHECK(result != NULL);
        CHECK(same_text(&stored, &hooks.text));
        CHECK(hooks.cache_result == H3_CACHE_HIT);
        CHECK(hooks.text_tokens > 0 && hooks.condition_rows == NULL &&
              hooks.latent == NULL && hooks.rgb == NULL);
        h3_text_embedding_free(&hooks.text);
        h3_result_free(result);
    }
    params.width = params.height = 0;

    /* A hit without the stop hook goes on to the reference VAE. */
    memset(&hooks, 0, sizeof(hooks));
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.text.values == NULL && at_vae_open(ctx));
    CHECK(hooks.cache_result == H3_CACHE_HIT);

    /* Another prompt, another reference, a disabled cache and test hooks that
     * replace pixels all miss: the run reaches the reference VAE. */
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, "Keep <Picture 1> as it is, by day.",
                                       &params, &hooks) == NULL);
    CHECK(hooks.text.values == NULL && at_vae_open(ctx));
    CHECK(hooks.cache_result == H3_CACHE_MISS);
    const char *other_path = path_in("wire_other.png");
    uint8_t *other = picture(64, 64, 5);
    write_png(other_path, other, 64, 64);
    free(other);
    h3_image_reference other_reference = {other_path};
    params.references = &other_reference;
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.text.values == NULL && at_vae_open(ctx));
    CHECK(hooks.cache_result == H3_CACHE_MISS);
    params.references = &reference;

    char *kept = ctx->cache_dir;
    ctx->cache_dir = NULL;
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.text.values == NULL && at_vae_open(ctx));
    CHECK(hooks.cache_result == H3_CACHE_NOT_USED);
    ctx->cache_dir = kept;

    const uint8_t *vae_hook[1] = {decoded};
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    hooks.vae_pixels = vae_hook;
    CHECK(h3_generate_image_with_hooks(ctx, prompt, &params, &hooks) == NULL);
    CHECK(hooks.text.values == NULL && at_vae_open(ctx));
    CHECK(hooks.cache_result == H3_CACHE_NOT_USED);

    /* Text-to-image shares the cache with no references. */
    h3_image_params text_params = H3_IMAGE_PARAMS_DEFAULT;
    text_params.min_generation_area = 1024;
    text_params.width = text_params.height = 64;
    text_params.seed = 11;
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    h3_result *result = h3_generate_image_with_hooks(ctx, "a red fox",
                                                     &text_params, &hooks);
    CHECK(result != NULL);
    CHECK(same_text(&plain, &hooks.text) && hooks.cache_result == H3_CACHE_HIT);
    h3_text_embedding_free(&hooks.text);
    h3_result_free(result);
    memset(&hooks, 0, sizeof(hooks));
    hooks.stop_after_lookup = 1;
    CHECK(h3_generate_image_with_hooks(ctx, "a blue fox", &text_params,
                                       &hooks) == NULL);
    CHECK(hooks.text.values == NULL && hooks.cache_result == H3_CACHE_MISS);

    free(decoded);
    free(edit_key);
    free(text_key);
    h3_text_embedding_free(&stored);
    h3_text_embedding_free(&plain);
    free(ctx->cache_dir);
    free(ctx->model_dir);
    free(ctx);
}

int main(int argc, char **argv) {
    if (argc > 2) {
        fprintf(stderr, "usage: %s [TOKENIZER_JSON]\n", argv[0]);
        return 2;
    }
    make_directory();
    test_key();
    test_round_trip();
    test_eviction();
    test_store_hygiene();
    test_limit();
    test_set_directory();
    if (argc > 1) test_wiring(argv[1]);
    else printf("skip: the wiring test needs TOKENIZER_JSON\n");
    remove_directory();
    printf("ok: %d image cache checks\n", tests_run);
    return 0;
}
