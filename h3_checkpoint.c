#include "h3_checkpoint.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    H3_CHECKPOINT_HEADER_SIZE = 128,
    H3_CHECKPOINT_ENDIAN = 0x01020304,
    H3_CHECKPOINT_HASH_OFFSET = 80
};

static const unsigned char h3_checkpoint_magic[8] = {
    'H', '3', 'C', 'K', 'P', 'T', '1', '\n'
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t size) {
    const unsigned char *bytes = data;
    for (size_t index = 0; index < size; index++) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t h3_checkpoint_signature(const char *text) {
    if (!text) return 0;
    return hash_bytes(UINT64_C(14695981039346656037), text, strlen(text));
}

static void put_u32(unsigned char *destination, uint32_t value) {
    for (unsigned index = 0; index < 4; index++)
        destination[index] = (unsigned char)(value >> (8 * index));
}

static void put_u64(unsigned char *destination, uint64_t value) {
    for (unsigned index = 0; index < 8; index++)
        destination[index] = (unsigned char)(value >> (8 * index));
}

static uint32_t get_u32(const unsigned char *source) {
    uint32_t value = 0;
    for (unsigned index = 0; index < 4; index++)
        value |= (uint32_t)source[index] << (8 * index);
    return value;
}

static uint64_t get_u64(const unsigned char *source) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++)
        value |= (uint64_t)source[index] << (8 * index);
    return value;
}

static uint64_t double_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double bits_double(uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int valid_info(const h3_checkpoint_info *info) {
    return info && info->signature && info->total_steps >= 2 &&
        info->next_step > 0 && info->next_step < info->total_steps &&
        info->render_width > 0 && info->render_height > 0 &&
        info->frame_count >= 22 && info->video_elements > 0 &&
        info->audio_elements > 0 && isfinite(info->denoise_seconds) &&
        info->denoise_seconds >= 0.0 &&
        info->video_elements <= SIZE_MAX / sizeof(float) &&
        info->audio_elements <= SIZE_MAX / sizeof(float);
}

static void encode_header(unsigned char header[H3_CHECKPOINT_HEADER_SIZE],
                          const h3_checkpoint_info *info, uint64_t hash) {
    memset(header, 0, H3_CHECKPOINT_HEADER_SIZE);
    memcpy(header, h3_checkpoint_magic, sizeof(h3_checkpoint_magic));
    put_u32(header + 8, H3_CHECKPOINT_HEADER_SIZE);
    uint32_t byte_order = H3_CHECKPOINT_ENDIAN;
    memcpy(header + 12, &byte_order, sizeof(byte_order));
    put_u64(header + 16, info->signature);
    put_u64(header + 24, info->seed);
    put_u32(header + 32, info->total_steps);
    put_u32(header + 36, info->next_step);
    put_u32(header + 40, info->render_width);
    put_u32(header + 44, info->render_height);
    put_u32(header + 48, info->frame_count);
    put_u64(header + 56, info->video_elements);
    put_u64(header + 64, info->audio_elements);
    put_u64(header + 72, double_bits(info->denoise_seconds));
    put_u64(header + H3_CHECKPOINT_HASH_OFFSET, hash);
}

static int decode_header(const unsigned char header[H3_CHECKPOINT_HEADER_SIZE],
                         h3_checkpoint_info *info, uint64_t *hash,
                         char *error, size_t error_size) {
    if (memcmp(header, h3_checkpoint_magic, sizeof(h3_checkpoint_magic)) ||
        get_u32(header + 8) != H3_CHECKPOINT_HEADER_SIZE) {
        fail(error, error_size, "unsupported H3 checkpoint format");
        return 0;
    }
    uint32_t byte_order;
    memcpy(&byte_order, header + 12, sizeof(byte_order));
    if (byte_order != H3_CHECKPOINT_ENDIAN) {
        fail(error, error_size, "checkpoint byte order is unsupported");
        return 0;
    }
    memset(info, 0, sizeof(*info));
    info->signature = get_u64(header + 16);
    info->seed = get_u64(header + 24);
    info->total_steps = get_u32(header + 32);
    info->next_step = get_u32(header + 36);
    info->render_width = get_u32(header + 40);
    info->render_height = get_u32(header + 44);
    info->frame_count = get_u32(header + 48);
    info->video_elements = get_u64(header + 56);
    info->audio_elements = get_u64(header + 64);
    info->denoise_seconds = bits_double(get_u64(header + 72));
    *hash = get_u64(header + H3_CHECKPOINT_HASH_OFFSET);
    if (!valid_info(info)) {
        fail(error, error_size, "checkpoint metadata is invalid");
        return 0;
    }
    return 1;
}

static uint64_t checkpoint_hash(
        const unsigned char header[H3_CHECKPOINT_HEADER_SIZE],
        const h3_checkpoint_info *info,
        const float *video, const float *audio) {
    unsigned char canonical[H3_CHECKPOINT_HEADER_SIZE];
    memcpy(canonical, header, sizeof(canonical));
    memset(canonical + H3_CHECKPOINT_HASH_OFFSET, 0, sizeof(uint64_t));
    uint64_t hash = hash_bytes(UINT64_C(14695981039346656037),
                               canonical, sizeof(canonical));
    hash = hash_bytes(hash, video,
                      (size_t)info->video_elements * sizeof(*video));
    return hash_bytes(hash, audio,
                      (size_t)info->audio_elements * sizeof(*audio));
}

static int write_all(int descriptor, const void *data, size_t size) {
    const unsigned char *bytes = data;
    while (size) {
        ssize_t written = write(descriptor, bytes, size);
        if (written < 0 && errno == EINTR) continue;
        if (written < 0) return 0;
        if (written == 0) {
            errno = EIO;
            return 0;
        }
        bytes += (size_t)written;
        size -= (size_t)written;
    }
    return 1;
}

static int read_all(int descriptor, void *data, size_t size) {
    unsigned char *bytes = data;
    while (size) {
        ssize_t count = read(descriptor, bytes, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        bytes += (size_t)count;
        size -= (size_t)count;
    }
    return 1;
}

int h3_checkpoint_save(const char *path, const h3_checkpoint_info *info,
                       const float *video, const float *audio,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !video || !audio || !valid_info(info)) {
        fail(error, error_size, "invalid checkpoint save arguments");
        return 0;
    }
    size_t path_length = strlen(path);
    if (path_length > SIZE_MAX - sizeof(".tmp.XXXXXX")) {
        fail(error, error_size, "checkpoint path is too long");
        return 0;
    }
    char *temporary = malloc(path_length + sizeof(".tmp.XXXXXX"));
    if (!temporary) {
        fail(error, error_size, "out of memory constructing checkpoint path");
        return 0;
    }
    snprintf(temporary, path_length + sizeof(".tmp.XXXXXX"),
             "%s.tmp.XXXXXX", path);
    int descriptor = mkstemp(temporary);
    if (descriptor < 0) {
        fail(error, error_size, "cannot create checkpoint %s: %s",
             temporary, strerror(errno));
        free(temporary);
        return 0;
    }
    if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) {
        int saved_errno = errno;
        close(descriptor);
        unlink(temporary);
        fail(error, error_size, "cannot secure checkpoint %s: %s",
             temporary, strerror(saved_errno));
        free(temporary);
        return 0;
    }
    unsigned char header[H3_CHECKPOINT_HEADER_SIZE];
    encode_header(header, info, 0);
    put_u64(header + H3_CHECKPOINT_HASH_OFFSET,
            checkpoint_hash(header, info, video, audio));
    size_t video_bytes = (size_t)info->video_elements * sizeof(*video);
    size_t audio_bytes = (size_t)info->audio_elements * sizeof(*audio);
    int ok = write_all(descriptor, header, sizeof(header)) &&
             write_all(descriptor, video, video_bytes) &&
             write_all(descriptor, audio, audio_bytes) &&
             fsync(descriptor) == 0;
    int saved_errno = errno;
    if (close(descriptor) != 0 && ok) {
        ok = 0;
        saved_errno = errno;
    }
    /* The file fsync prevents a torn replacement. The rename is atomic, but
     * this does not promise directory-entry durability across power loss. */
    if (ok && rename(temporary, path) != 0) {
        ok = 0;
        saved_errno = errno;
    }
    if (!ok) {
        unlink(temporary);
        fail(error, error_size, "cannot write checkpoint %s: %s",
             path, strerror(saved_errno));
    }
    free(temporary);
    return ok;
}

static int compatible(const h3_checkpoint_info *expected,
                      const h3_checkpoint_info *actual) {
    return expected->signature == actual->signature &&
        expected->seed == actual->seed &&
        expected->total_steps == actual->total_steps &&
        expected->render_width == actual->render_width &&
        expected->render_height == actual->render_height &&
        expected->frame_count == actual->frame_count &&
        expected->video_elements == actual->video_elements &&
        expected->audio_elements == actual->audio_elements;
}

int h3_checkpoint_load(const char *path,
                       const h3_checkpoint_info *expected,
                       h3_checkpoint_info *actual,
                       float **video, float **audio,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (video) *video = NULL;
    if (audio) *audio = NULL;
    if (!path || !*path || !expected || !actual || !video || !audio) {
        fail(error, error_size, "invalid checkpoint load arguments");
        return 0;
    }
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    int descriptor = open(path, flags);
    if (descriptor < 0) {
        fail(error, error_size, "cannot open checkpoint %s: %s",
             path, strerror(errno));
        return 0;
    }
    struct stat status;
    unsigned char header[H3_CHECKPOINT_HEADER_SIZE];
    uint64_t stored_hash = 0;
    int ok = fstat(descriptor, &status) == 0 && S_ISREG(status.st_mode) &&
             read_all(descriptor, header, sizeof(header)) &&
             decode_header(header, actual, &stored_hash, error, error_size);
    if (!ok && error && error_size && !*error)
        fail(error, error_size, "cannot read checkpoint %s", path);
    if (ok && !compatible(expected, actual)) {
        fail(error, error_size,
             "checkpoint does not match prompt, model, seed, shape, or schedule");
        ok = 0;
    }
    uint64_t video_bytes = ok ? actual->video_elements * sizeof(float) : 0;
    uint64_t audio_bytes = ok ? actual->audio_elements * sizeof(float) : 0;
    uint64_t expected_bytes = H3_CHECKPOINT_HEADER_SIZE;
    if (ok && (video_bytes > UINT64_MAX - expected_bytes ||
               audio_bytes > UINT64_MAX - expected_bytes - video_bytes)) {
        fail(error, error_size, "checkpoint size overflows");
        ok = 0;
    }
    if (ok) expected_bytes += video_bytes + audio_bytes;
    if (ok && (status.st_size < 0 ||
               (uint64_t)status.st_size != expected_bytes)) {
        fail(error, error_size, "checkpoint file size is inconsistent");
        ok = 0;
    }
    if (ok) {
        *video = malloc((size_t)video_bytes);
        *audio = malloc((size_t)audio_bytes);
        if (!*video || !*audio) {
            fail(error, error_size, "out of memory loading checkpoint latents");
            ok = 0;
        }
    }
    if (ok && (!read_all(descriptor, *video, (size_t)video_bytes) ||
               !read_all(descriptor, *audio, (size_t)audio_bytes))) {
        fail(error, error_size, "checkpoint payload is truncated");
        ok = 0;
    }
    if (ok && checkpoint_hash(header, actual, *video, *audio) != stored_hash) {
        fail(error, error_size, "checkpoint checksum does not match");
        ok = 0;
    }
    if (close(descriptor) != 0 && ok) {
        fail(error, error_size, "cannot close checkpoint %s: %s",
             path, strerror(errno));
        ok = 0;
    }
    if (!ok) {
        free(*video);
        free(*audio);
        *video = NULL;
        *audio = NULL;
    }
    return ok;
}
