#ifndef H3_IMAGE_CACHE_H
#define H3_IMAGE_CACHE_H

#include "h3_internal.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

/* Bump when the presentation builder, the sizing rules or the tokenization
 * change what an embedding means. */
#define H3_EDIT_CACHE_VERSION 1
#define H3_EDIT_CACHE_DEFAULT_BYTES ((uint64_t)2 << 30)

/* The cache key of a text embedding, malloc'd (the caller frees it). count is
 * the number of references, 0 for text-to-image; sources are the decoded
 * planar bytes [3][h][w] of each reference, vision_* the Qwen vision size of
 * its geometry. The key names the content and the computation: the pixels, the
 * prompt, the model files (name, size, time and inode), the tokenizer, the
 * running executable (which holds the host code and the shader source it was
 * built with), the shader file, the GPU and the H3_ environment. It does not
 * name a reference's file name, the seed or the canvas. NULL when out of memory. */
char *h3_image_cache_key(const h3_ctx *ctx, const char *prompt, size_t count,
                         const int *widths, const int *heights,
                         const uint8_t *const *sources,
                         const int *vision_widths, const int *vision_heights);

/* Test seam: the build identity the key uses instead of the SHA-256 of the
 * running executable; NULL goes back to the executable. */
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
