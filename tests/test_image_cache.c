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

static void set_age(const char *path, int seconds_ago) {
    struct timeval times[2];
    gettimeofday(&times[0], NULL);
    times[0].tv_sec -= seconds_ago;
    times[1] = times[0];
    CHECK(utimes(path, times) == 0);
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

int main(void) {
    make_directory();
    test_key();
    test_round_trip();
    test_eviction();
    test_limit();
    remove_directory();
    printf("ok: %d image cache checks\n", tests_run);
    return 0;
}
