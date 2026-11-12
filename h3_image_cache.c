#include "h3_image_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define CACHE_MAGIC "H3EC"
#define CACHE_FORMAT 1u
#define CACHE_HEADER_BYTES 40u
#define CACHE_SUFFIX ".h3ec"
#define CACHE_MAX_KEY_BYTES ((uint64_t)1 << 20)
#define CACHE_MAX_TOKENS ((uint64_t)1 << 20)

static void hex_digest(const uint8_t digest[CC_SHA256_DIGEST_LENGTH],
                       char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    static const char digits[] = "0123456789abcdef";
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) {
        out[2 * index] = digits[digest[index] >> 4];
        out[2 * index + 1] = digits[digest[index] & 15];
    }
    out[2 * CC_SHA256_DIGEST_LENGTH] = '\0';
}

static void sha_update(CC_SHA256_CTX *context, const void *bytes,
                       size_t count) {
    const uint8_t *cursor = bytes;
    while (count) {
        size_t piece = count < ((size_t)1 << 30) ? count : ((size_t)1 << 30);
        CC_SHA256_Update(context, cursor, (CC_LONG)piece);
        cursor += piece;
        count -= piece;
    }
}

static void sha_hex(const void *bytes, size_t count,
                    char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    CC_SHA256_CTX context;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Init(&context);
    sha_update(&context, bytes, count);
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
}

/* "none" when the file cannot be read. */
static void file_sha_hex(const char *path,
                         char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        strcpy(out, "none");
        return;
    }
    CC_SHA256_CTX context;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    uint8_t buffer[1 << 16];
    size_t count;
    int failed = 0;
    CC_SHA256_Init(&context);
    while ((count = fread(buffer, 1, sizeof(buffer), file)) > 0)
        CC_SHA256_Update(&context, buffer, (CC_LONG)count);
    failed = ferror(file);
    fclose(file);
    if (failed) {
        strcpy(out, "none");
        return;
    }
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
}

char *h3_image_cache_key(const h3_ctx *ctx, const char *prompt, size_t count,
                         const int *widths, const int *heights,
                         const uint8_t *const *sources,
                         const int *vision_widths, const int *vision_heights) {
    char *text = NULL;
    size_t length = 0;
    FILE *key = open_memstream(&text, &length);
    if (!key) return NULL;
    char *real = realpath(ctx->model_dir, NULL);
    char *tokenizer_path = malloc(strlen(ctx->model_dir) + 32);
    char tokenizer[2 * CC_SHA256_DIGEST_LENGTH + 1];
    char shaders[2 * CC_SHA256_DIGEST_LENGTH + 1];
    if (!tokenizer_path) {
        fclose(key);
        free(text);
        free(real);
        return NULL;
    }
    sprintf(tokenizer_path, "%s/tokenizer/tokenizer.json", ctx->model_dir);
    file_sha_hex(tokenizer_path, tokenizer);
    file_sha_hex("h3_shaders.metal", shaders);
    const h3_component_info *encoder = &ctx->model.text_encoder;
    fprintf(key, "edit-cache=%d|model=%s|text-encoder=%llu:%zu:%zu"
            "|tokenizer=%s|shaders=%s|prompt=%zu:", H3_EDIT_CACHE_VERSION,
            real ? real : ctx->model_dir, (unsigned long long)encoder->bytes,
            encoder->files, encoder->tensors, tokenizer, shaders,
            strlen(prompt));
    fwrite(prompt, 1, strlen(prompt), key);
    fprintf(key, "|refs=%zu", count);
    for (size_t index = 0; index < count; index++) {
        char pixels[2 * CC_SHA256_DIGEST_LENGTH + 1];
        sha_hex(sources[index],
                (size_t)3 * (size_t)widths[index] * (size_t)heights[index],
                pixels);
        fprintf(key, "|ref%zu=%dx%d:%s:%dx%d", index, widths[index],
                heights[index], pixels, vision_widths[index],
                vision_heights[index]);
    }
    free(tokenizer_path);
    free(real);
    if (fclose(key) != 0) {
        free(text);
        return NULL;
    }
    return text;
}

int h3_image_cache_path(const char *dir, const char *key, char *path,
                        size_t path_size) {
    char name[2 * CC_SHA256_DIGEST_LENGTH + 1];
    sha_hex(key, strlen(key), name);
    int written = snprintf(path, path_size, "%s/%s" CACHE_SUFFIX, dir, name);
    return written > 0 && (size_t)written < path_size;
}

static void put_u32(uint8_t *at, uint32_t value) {
    for (size_t index = 0; index < 4; index++)
        at[index] = (uint8_t)(value >> (8 * index));
}

static void put_u64(uint8_t *at, uint64_t value) {
    for (size_t index = 0; index < 8; index++)
        at[index] = (uint8_t)(value >> (8 * index));
}

static uint32_t get_u32(const uint8_t *at) {
    uint32_t value = 0;
    for (size_t index = 0; index < 4; index++)
        value |= (uint32_t)at[index] << (8 * index);
    return value;
}

static uint64_t get_u64(const uint8_t *at) {
    uint64_t value = 0;
    for (size_t index = 0; index < 8; index++)
        value |= (uint64_t)at[index] << (8 * index);
    return value;
}

static void warn(const char *path, const char *reason) {
    fprintf(stderr, "h3: warning: edit text cache: ignoring %s: %s\n", path,
            reason);
}

static int read_exact(FILE *file, void *bytes, size_t count,
                      CC_SHA256_CTX *context) {
    if (fread(bytes, 1, count, file) != count) return 0;
    if (context) sha_update(context, bytes, count);
    return 1;
}

int h3_image_cache_load(const char *dir, const char *key,
                        h3_text_embedding *text, char *path,
                        size_t path_size) {
    memset(text, 0, sizeof(*text));
    if (!h3_image_cache_path(dir, key, path, path_size)) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) {
        if (errno != ENOENT) warn(path, strerror(errno));
        return 0;
    }
    struct stat status;
    uint8_t header[CACHE_HEADER_BYTES];
    uint8_t stored[CC_SHA256_DIGEST_LENGTH], computed[CC_SHA256_DIGEST_LENGTH];
    char *stored_key = NULL;
    const char *reason = "truncated";
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    h3_text_embedding loaded;
    memset(&loaded, 0, sizeof(loaded));
    if (fstat(fileno(file), &status) != 0 ||
        !read_exact(file, header, sizeof(header), &context)) goto bad;
    uint64_t key_length = get_u64(header + 8);
    uint64_t tokens = get_u64(header + 16);
    uint64_t width = get_u64(header + 24);
    uint64_t has_tags = get_u64(header + 32);
    reason = "not an H3 cache file";
    if (memcmp(header, CACHE_MAGIC, 4)) goto bad;
    reason = "another file format version";
    if (get_u32(header + 4) != CACHE_FORMAT) goto bad;
    reason = "corrupt header";
    if (key_length > CACHE_MAX_KEY_BYTES || tokens == 0 ||
        tokens > CACHE_MAX_TOKENS || width != H3_TEXT_HIDDEN_SIZE ||
        has_tags > 1) goto bad;
    uint64_t expected = CACHE_HEADER_BYTES + key_length +
                        tokens * width * sizeof(uint16_t) + has_tags * tokens +
                        CC_SHA256_DIGEST_LENGTH;
    reason = "truncated";
    if ((uint64_t)status.st_size != expected) goto bad;
    stored_key = malloc((size_t)key_length + 1);
    if (!stored_key || !read_exact(file, stored_key, (size_t)key_length,
                                   &context)) goto bad;
    reason = "holds another key";
    if (key_length != strlen(key) || memcmp(stored_key, key, (size_t)key_length))
        goto bad;
    reason = "cannot read";
    loaded.tokens = (size_t)tokens;
    loaded.width = (size_t)width;
    loaded.values = malloc(loaded.tokens * loaded.width *
                           sizeof(*loaded.values));
    if (has_tags) loaded.tags = malloc(loaded.tokens);
    if (!loaded.values || (has_tags && !loaded.tags) ||
        !read_exact(file, loaded.values, loaded.tokens * loaded.width *
                    sizeof(*loaded.values), &context) ||
        (has_tags && !read_exact(file, loaded.tags, loaded.tokens, &context)))
        goto bad;
    CC_SHA256_Final(computed, &context);
    if (!read_exact(file, stored, sizeof(stored), NULL)) goto bad;
    reason = "checksum mismatch";
    if (memcmp(stored, computed, sizeof(stored))) goto bad;
    fclose(file);
    free(stored_key);
    utimes(path, NULL);
    *text = loaded;
    return 1;
bad:
    warn(path, reason);
    fclose(file);
    free(stored_key);
    h3_text_embedding_free(&loaded);
    return 0;
}

static int make_directories(const char *dir) {
    char *copy = strdup(dir);
    if (!copy) return 0;
    for (char *at = copy + 1; ; at++) {
        if (*at == '/' || *at == '\0') {
            char saved = *at;
            *at = '\0';
            if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
                free(copy);
                return 0;
            }
            *at = saved;
            if (!saved) break;
        }
    }
    free(copy);
    struct stat status;
    return stat(dir, &status) == 0 && S_ISDIR(status.st_mode);
}

static int write_entry(FILE *file, const char *key,
                       const h3_text_embedding *text) {
    uint8_t header[CACHE_HEADER_BYTES];
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX context;
    size_t key_length = strlen(key);
    size_t value_bytes = text->tokens * text->width * sizeof(*text->values);
    memcpy(header, CACHE_MAGIC, 4);
    put_u32(header + 4, CACHE_FORMAT);
    put_u64(header + 8, key_length);
    put_u64(header + 16, text->tokens);
    put_u64(header + 24, text->width);
    put_u64(header + 32, text->tags != NULL);
    CC_SHA256_Init(&context);
    sha_update(&context, header, sizeof(header));
    sha_update(&context, key, key_length);
    sha_update(&context, text->values, value_bytes);
    if (text->tags) sha_update(&context, text->tags, text->tokens);
    CC_SHA256_Final(digest, &context);
    return fwrite(header, 1, sizeof(header), file) == sizeof(header) &&
           fwrite(key, 1, key_length, file) == key_length &&
           fwrite(text->values, 1, value_bytes, file) == value_bytes &&
           (!text->tags ||
            fwrite(text->tags, 1, text->tokens, file) == text->tokens) &&
           fwrite(digest, 1, sizeof(digest), file) == sizeof(digest);
}

int h3_image_cache_store(const char *dir, const char *key,
                         const h3_text_embedding *text, uint64_t limit,
                         char *path, size_t path_size) {
    if (!h3_image_cache_path(dir, key, path, path_size)) return 0;
    if (!text->tokens || text->width != H3_TEXT_HIDDEN_SIZE ||
        !make_directories(dir)) {
        warn(dir, "cannot create the cache directory");
        return 0;
    }
    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path,
                 (int)getpid()) >= (int)sizeof(temporary)) return 0;
    FILE *file = fopen(temporary, "wb");
    int ok = file != NULL;
    if (ok) ok = write_entry(file, key, text);
    if (file && fclose(file) != 0) ok = 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) {
        warn(path, "cannot write the entry");
        unlink(temporary);
        return 0;
    }
    (void)limit;
    return 1;
}

uint64_t h3_image_cache_limit(void) {
    const char *text = getenv("H3_CACHE_BYTES");
    if (!text || !*text) return H3_EDIT_CACHE_DEFAULT_BYTES;
    char *end;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || value == 0 || text[0] == '-') {
        fprintf(stderr, "h3: warning: H3_CACHE_BYTES=%s is not a positive "
                "number of bytes; using %llu\n", text,
                (unsigned long long)H3_EDIT_CACHE_DEFAULT_BYTES);
        return H3_EDIT_CACHE_DEFAULT_BYTES;
    }
    return value;
}
