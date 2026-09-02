#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;
typedef struct h3_lora_set h3_lora_set;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);
size_t h3_weight_store_tensor_count(const h3_weight_store *store);
const h3_st_tensor *h3_weight_store_tensor_at(
    const h3_weight_store *store, size_t index);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* Load an exact BF16 matrix, applying ordered LoRA deltas in bounded row
 * tiles. An untouched tensor retains the ordinary zero-copy load path. */
h3_gpu_tensor *h3_weight_load_bf16_with_loras(
    const h3_weight_store *store, h3_gpu *gpu, const char *name,
    uint64_t rows, uint64_t columns, const h3_lora_set *loras,
    size_t *peak_scratch, char *error, size_t error_size);

#endif
