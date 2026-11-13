#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* Source and destination of one head_dim-row slab of the grouped
 * [head][q,k,v][head_dim] QKV matrix: the projection (0 = q, 1 = k, 2 = v) it
 * comes from and the element offsets in that projection's matrix and in the
 * grouped matrix. Returns 0 for a slab outside 3 * heads. */
int h3_weight_qkv_slab(size_t slab, unsigned heads, unsigned head_dim,
                       uint64_t columns, unsigned *projection,
                       uint64_t *source_offset, uint64_t *destination_offset);

/* Load three [heads * head_dim, columns] BF16 projections into one grouped
 * [head][q,k,v][head_dim] matrix with 3 * heads ranged reads straight from the
 * shard files. */
h3_gpu_tensor *h3_weight_load_bf16_qkv(const h3_weight_store *store,
                                       h3_gpu *gpu, const char *q_name,
                                       const char *k_name, const char *v_name,
                                       unsigned heads, unsigned head_dim,
                                       uint64_t columns,
                                       char *error, size_t error_size);

/* Load a [rows, columns] BF16 matrix whose two row halves are exchanged when
 * swap_halves is set (two half-tensor ranged reads). */
h3_gpu_tensor *h3_weight_load_bf16_halves(const h3_weight_store *store,
                                          h3_gpu *gpu, const char *name,
                                          uint64_t rows, uint64_t columns,
                                          int swap_halves,
                                          char *error, size_t error_size);

#endif
