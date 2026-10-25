#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct h3_weight_store h3_weight_store;

/* Round-to-nearest-even F32 -> BF16 bit pattern. This matches the packing
 * used by the released BF16 checkpoints and by the native Metal kernels, so
 * host-side conversions and checkpoint payloads agree bit for bit. */
static inline uint16_t h3_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static inline float h3_bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

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

/* Validate an exact shape and upload the tensor as BF16. Payloads stored as
 * F32 (as the released video VAE checkpoint does) are converted on the host
 * with round-to-nearest-even before the upload; payloads that are already
 * BF16 are streamed straight into the Metal buffer. */
h3_gpu_tensor *h3_weight_load_f32_as_bf16(const h3_weight_store *store,
                                           h3_gpu *gpu, const char *name,
                                           int ndim, const uint64_t *shape,
                                           char *error, size_t error_size);

#endif
