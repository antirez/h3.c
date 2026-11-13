#ifndef H3_IMAGE_CACHE_H
#define H3_IMAGE_CACHE_H

#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

/* Bump when the presentation builder, the sizing rules or the tokenization
 * change what an embedding means. */
#define H3_EDIT_CACHE_VERSION 1
#define H3_EDIT_CACHE_DEFAULT_BYTES ((uint64_t)2 << 30)

/* What names the computation besides the request. The caller gathers these
 * from its own state; the module reads no context. The key also covers the
 * tokenizer file, the shader file and the H3_ environment, which it reads
 * itself. */
typedef struct {
    const char *model_dir;
    const char *device_name;
    /* The text encoder inventory of the checkpoint. */
    unsigned long long text_encoder_bytes;
    size_t text_encoder_files, text_encoder_tensors;
    /* The identity of the code that computes an embedding: a digest of its
     * sources, fixed at build time. NULL is "none". */
    const char *build_id;
} h3_image_cache_inputs;

/* The cache key of a text embedding, malloc'd (the caller frees it). count is
 * the number of references, 0 for text-to-image; sources are the decoded
 * planar bytes [3][h][w] of each reference, vision_* the Qwen vision size of
 * its geometry. The key names the content and the computation: the pixels, the
 * prompt, the model files (name, size, time and inode), the tokenizer, the
 * build identity, the shader file, the GPU and the H3_ environment. It does
 * not name a reference's file name, the seed or the canvas. NULL when out of
 * memory. */
char *h3_image_cache_key(const h3_image_cache_inputs *inputs,
                         const char *prompt, size_t count,
                         const int *widths, const int *heights,
                         const uint8_t *const *sources,
                         const int *vision_widths, const int *vision_heights);

/* Test seam: a build identity that replaces inputs->build_id in the key; NULL
 * removes it. */
void h3_image_cache_set_build_id(const char *id);

/* The entry file of a key inside dir. */
int h3_image_cache_path(const char *dir, const char *key, char *path,
                        size_t path_size);

/* 1 and a malloc'd embedding on a hit, which also touches the file. A missing
 * file is a plain miss; an unreadable, corrupt, truncated or foreign file is a
 * miss with one warning line. path receives the entry file either way. */
int h3_image_cache_load(const char *dir, const char *key,
                        h3_text_embedding *text, char *path,
                        size_t path_size);

/* Write the entry atomically, then delete the oldest entries until the
 * directory holds at most limit bytes. 0 with one warning line on failure. */
int h3_image_cache_store(const char *dir, const char *key,
                         const h3_text_embedding *text, uint64_t limit,
                         char *path, size_t path_size);

/* H3_CACHE_BYTES, or the default when unset or not a positive number. */
uint64_t h3_image_cache_limit(void);

#endif
