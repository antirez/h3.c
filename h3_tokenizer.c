/* h3_tokenizer.c - C stub for the CUDA/Linux build (feat/cuda).
 *
 * The Metal build uses h3_tokenizer.m (Objective-C/Foundation). On Linux there
 * is no Foundation, so the CUDA build compiles this C file against the same
 * h3_tokenizer.h API. It is a scaffold stub for I1: it satisfies the link so
 * the host binary builds and `--info` runs. Real tokenizer port is a later
 * iteration (I20). Generation paths that need tokenization return an error.
 */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_tokenizer {
    int unused;
};

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size) {
    (void)tokenizer_json;
    if (error && error_size) {
        snprintf(error, error_size,
                 "CUDA build: tokenizer not yet ported (feat/cuda)");
    }
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    (void)tokenizer;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    (void)tokenizer; (void)utf8; (void)pad_empty;
    if (ids) *ids = NULL;
    if (count) *count = 0;
    if (error && error_size) {
        snprintf(error, error_size,
                 "CUDA build: tokenizer not yet ported (feat/cuda)");
    }
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    (void)tokenizer; (void)ids; (void)count;
    if (error && error_size) {
        snprintf(error, error_size,
                 "CUDA build: tokenizer not yet ported (feat/cuda)");
    }
    return NULL;
}
