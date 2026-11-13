#include "h3_image_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <crt_externs.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
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

static char build_id_override[2 * CC_SHA256_DIGEST_LENGTH + 1];

void h3_image_cache_set_build_id(const char *id) {
    if (id) snprintf(build_id_override, sizeof(build_id_override), "%s", id);
    else build_id_override[0] = '\0';
}

static const char *build_identity(const h3_image_cache_inputs *inputs) {
    if (build_id_override[0]) return build_id_override;
    return inputs->build_id && *inputs->build_id ? inputs->build_id : "none";
}

static int by_name(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Every H3_ variable can change kernel routing or numerics, except these. */
static int ignored_variable(const char *name, size_t length) {
    static const char *const ignored[] = {
        "H3_CACHE_DIR", "H3_CACHE_BYTES", "H3_PROFILE", "H3_DEBUG_GPU_MEMORY",
        "H3_DEBUG_VISION", "H3_DEBUG_VISION_STEPS", "H3_FFMPEG", "H3_FFPROBE",
        "H3_AUDIO_ENCODER_TRACE", "H3_QWEN_PREFETCH", "H3_QWEN_PREFETCH_DEPTH"};
    for (size_t index = 0; index < sizeof(ignored) / sizeof(ignored[0]);
         index++)
        if (strlen(ignored[index]) == length &&
            !strncmp(ignored[index], name, length)) return 1;
    return 0;
}

static void environment_hash(char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    char **environment = *_NSGetEnviron();
    size_t count = 0, capacity = 0;
    char **picked = NULL;
    for (size_t index = 0; environment && environment[index]; index++) {
        const char *entry = environment[index];
        const char *equals = strchr(entry, '=');
        size_t length = equals ? (size_t)(equals - entry) : strlen(entry);
        if (strncmp(entry, "H3_", 3) || ignored_variable(entry, length))
            continue;
        if (count == capacity) {
            char **bigger = realloc(picked, (capacity ? capacity * 2 : 16) *
                                    sizeof(*picked));
            if (!bigger) break;
            picked = bigger;
            capacity = capacity ? capacity * 2 : 16;
        }
        picked[count++] = (char *)entry;
    }
    qsort(picked, count, sizeof(*picked), by_name);
    CC_SHA256_CTX context;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Init(&context);
    for (size_t index = 0; index < count; index++) {
        sha_update(&context, picked[index], strlen(picked[index]) + 1);
    }
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
    free(picked);
}

/* Name, size, modification time and inode of every file of a directory. */
static void directory_hash(const char *path,
                           char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    DIR *directory = opendir(path);
    if (!directory) {
        strcpy(out, "none");
        return;
    }
    char **names = NULL;
    size_t count = 0, capacity = 0;
    struct dirent *item;
    while ((item = readdir(directory))) {
        if (item->d_name[0] == '.') continue;
        if (count == capacity) {
            char **bigger = realloc(names, (capacity ? capacity * 2 : 32) *
                                    sizeof(*names));
            if (!bigger) break;
            names = bigger;
            capacity = capacity ? capacity * 2 : 32;
        }
        names[count] = strdup(item->d_name);
        if (names[count]) count++;
    }
    closedir(directory);
    qsort(names, count, sizeof(*names), by_name);
    CC_SHA256_CTX context;
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Init(&context);
    for (size_t index = 0; index < count; index++) {
        char file[PATH_MAX], line[PATH_MAX + 128];
        struct stat status;
        snprintf(file, sizeof(file), "%s/%s", path, names[index]);
        if (stat(file, &status) == 0) {
            int length = snprintf(line, sizeof(line), "%s:%lld:%lld.%09ld:%llu\n",
                                  names[index], (long long)status.st_size,
                                  (long long)status.st_mtimespec.tv_sec,
                                  status.st_mtimespec.tv_nsec,
                                  (unsigned long long)status.st_ino);
            if (length > 0 && (size_t)length < sizeof(line))
                sha_update(&context, line, (size_t)length);
        }
        free(names[index]);
    }
    free(names);
    CC_SHA256_Final(digest, &context);
    hex_digest(digest, out);
}

char *h3_image_cache_key(const h3_image_cache_inputs *inputs,
                         const char *prompt, size_t count,
                         const int *widths, const int *heights,
                         const uint8_t *const *sources,
                         const int *vision_widths, const int *vision_heights) {
    char *text = NULL;
    size_t length = 0;
    FILE *key = open_memstream(&text, &length);
    if (!key) return NULL;
    char *real = realpath(inputs->model_dir, NULL);
    char *tokenizer_path = malloc(strlen(inputs->model_dir) + 32);
    char tokenizer[2 * CC_SHA256_DIGEST_LENGTH + 1];
    char shaders[2 * CC_SHA256_DIGEST_LENGTH + 1];
    char weights[2 * CC_SHA256_DIGEST_LENGTH + 1];
    char environment[2 * CC_SHA256_DIGEST_LENGTH + 1];
    if (!tokenizer_path) {
        fclose(key);
        free(text);
        free(real);
        return NULL;
    }
    sprintf(tokenizer_path, "%s/tokenizer/tokenizer.json", inputs->model_dir);
    file_sha_hex(tokenizer_path, tokenizer);
    file_sha_hex("h3_shaders.metal", shaders);
    sprintf(tokenizer_path, "%s/text_encoder", inputs->model_dir);
    directory_hash(tokenizer_path, weights);
    environment_hash(environment);
    fprintf(key, "edit-cache=%d|build=%s|device=%s|env=%s|model=%s"
            "|text-encoder=%llu:%zu:%zu|weights=%s|tokenizer=%s|shaders=%s"
            "|prompt=%zu:", H3_EDIT_CACHE_VERSION, build_identity(inputs),
            inputs->device_name, environment, real ? real : inputs->model_dir,
            inputs->text_encoder_bytes, inputs->text_encoder_files,
            inputs->text_encoder_tensors, weights, tokenizer, shaders, strlen(prompt));
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
    fprintf(stderr, "h3: warning: text cache: ignoring %s: %s\n", path,
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
    if (!*dir) {
        errno = EINVAL;
        return 0;
    }
    char *copy = strdup(dir);
    if (!copy) return 0;
    for (char *at = copy + 1; ; at++) {
        if (*at == '/' || *at == '\0') {
            char saved = *at;
            *at = '\0';
            if (mkdir(copy, 0700) != 0 && errno != EEXIST) {
                free(copy);
                return 0;
            }
            *at = saved;
            if (!saved) break;
        }
    }
    free(copy);
    struct stat status;
    if (stat(dir, &status) != 0) return 0;
    if (!S_ISDIR(status.st_mode)) errno = ENOTDIR;
    return S_ISDIR(status.st_mode);
}

typedef struct {
    char name[NAME_MAX + 1];
    struct timespec modified;
    uint64_t bytes;
} cache_entry;

static int older_first(const void *a, const void *b) {
    const cache_entry *x = a, *y = b;
    if (x->modified.tv_sec != y->modified.tv_sec)
        return x->modified.tv_sec < y->modified.tv_sec ? -1 : 1;
    if (x->modified.tv_nsec != y->modified.tv_nsec)
        return x->modified.tv_nsec < y->modified.tv_nsec ? -1 : 1;
    return strcmp(x->name, y->name);
}

static void evict(const char *dir, uint64_t limit) {
    DIR *directory = opendir(dir);
    if (!directory) return;
    cache_entry *entries = NULL;
    size_t count = 0, capacity = 0;
    uint64_t total = 0;
    struct dirent *item;
    size_t suffix = strlen(CACHE_SUFFIX);
    time_t now = time(NULL);
    while ((item = readdir(directory))) {
        size_t length = strlen(item->d_name);
        const char *orphan = strstr(item->d_name, CACHE_SUFFIX ".tmp.");
        if (orphan) {
            char path[PATH_MAX];
            struct stat status;
            if (snprintf(path, sizeof(path), "%s/%s", dir, item->d_name) <
                    (int)sizeof(path) &&
                stat(path, &status) == 0 &&
                now - status.st_mtimespec.tv_sec > 3600) unlink(path);
            continue;
        }
        if (length <= suffix ||
            strcmp(item->d_name + length - suffix, CACHE_SUFFIX)) continue;
        char path[PATH_MAX];
        struct stat status;
        if (snprintf(path, sizeof(path), "%s/%s", dir, item->d_name) >=
                (int)sizeof(path) ||
            stat(path, &status) != 0 || !S_ISREG(status.st_mode)) continue;
        if (count == capacity) {
            size_t grown = capacity ? capacity * 2 : 16;
            cache_entry *bigger = realloc(entries, grown * sizeof(*entries));
            if (!bigger) break;
            entries = bigger;
            capacity = grown;
        }
        snprintf(entries[count].name, sizeof(entries[count].name), "%s",
                 item->d_name);
        entries[count].modified = status.st_mtimespec;
        entries[count].bytes = (uint64_t)status.st_size;
        total += entries[count].bytes;
        count++;
    }
    closedir(directory);
    qsort(entries, count, sizeof(*entries), older_first);
    for (size_t index = 0; index < count && total > limit; index++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", dir, entries[index].name);
        if (unlink(path) == 0) total -= entries[index].bytes;
    }
    free(entries);
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
    if (!text->tokens || text->width != H3_TEXT_HIDDEN_SIZE) {
        warn(path, "not a text embedding");
        return 0;
    }
    if (!make_directories(dir)) {
        warn(dir, strerror(errno));
        return 0;
    }
    char temporary[PATH_MAX];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path,
                 (int)getpid()) >= (int)sizeof(temporary)) return 0;
    errno = 0;
    int descriptor = open(temporary, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *file = descriptor >= 0 ? fdopen(descriptor, "wb") : NULL;
    if (descriptor >= 0 && !file) close(descriptor);
    int ok = file != NULL;
    if (ok) ok = write_entry(file, key, text);
    if (file && fclose(file) != 0) ok = 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) {
        warn(path, errno ? strerror(errno) : "cannot write the entry");
        unlink(temporary);
        return 0;
    }
    evict(dir, limit);
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
