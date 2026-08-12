/* h3_cuda.cu - CUDA backend for h3.c (feat/cuda).
 * Metal backend (h3_gpu.m/h3_shaders.metal) is preserved untouched;
 * this file implements the same h3_gpu.h C API against CUDA/cuBLAS.
 * I1 scaffold: probe/create/free + I2 tensor layer real; compute ops are stubs.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include "h3_gpu.h"
#include "h3.h"
#include "h3_cuda.h"

#define H3_CUDA_ERR "CUDA backend: op not yet implemented (feat/cuda)"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

struct h3_gpu { void *dev_ctx; h3_gpu_stats stats; char error[512]; };
struct h3_gpu_tensor { void *device_ptr; h3_gpu_dtype dtype; size_t elements; size_t bytes; h3_gpu *owner; };

static size_t h3_gpu_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
    case H3_GPU_F32: return sizeof(float);
    case H3_GPU_BF16: return sizeof(uint16_t);
    case H3_GPU_I8: return sizeof(int8_t);
    case H3_GPU_U32: return sizeof(uint32_t);
    default: return 0;
    }
}

/* Allocate device memory (optionally filled from host `values`). */
static h3_gpu_tensor *h3_gpu_tensor_alloc(h3_gpu *gpu, size_t elements, h3_gpu_dtype dtype, const void *values) {
    size_t bytes = elements * h3_gpu_dtype_size(dtype);
    if (bytes == 0) return NULL;
    void *dptr = NULL;
    if (cudaMalloc(&dptr, bytes) != cudaSuccess) return NULL;
    if (values) {
        if (cudaMemcpy(dptr, values, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dptr); return NULL;
        }
    }
    h3_gpu_tensor *t = (h3_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) { cudaFree(dptr); return NULL; }
    t->device_ptr = dptr; t->dtype = dtype; t->elements = elements; t->bytes = bytes; t->owner = gpu;
    if (gpu) {
        gpu->stats.allocated_bytes += bytes;
        gpu->stats.live_bytes += bytes;
        if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
            gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
        gpu->stats.tensor_allocations++;
    }
    return t;
}

/* Stream a file range into a device tensor via a small host staging buffer. */
static int h3_gpu_tensor_file_load(h3_gpu *gpu, h3_gpu_tensor *t, const char *path, uint64_t file_offset, char *error, size_t error_size) {
    if (!t || !path || !*path || file_offset > (uint64_t)INT64_MAX) return 0;
    size_t bytes = t->bytes;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (error && error_size) snprintf(error, error_size, "cannot open %s: %s", path, strerror(errno)); return 0; }
    size_t chunk = MIN(bytes, (size_t)(1 << 20));
    void *host = malloc(chunk ? chunk : 1);
    if (!host) { close(fd); return 0; }
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(chunk, bytes - completed);
        size_t got = 0;
        while (got < request) {
            ssize_t count = pread(fd, (char *)host + got, request - got, (off_t)(file_offset + completed + got));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                if (error && error_size) snprintf(error, error_size, "cannot read %s payload: %s", path, count < 0 ? strerror(errno) : "unexpected end of file");
                free(host); close(fd); return 0;
            }
            got += (size_t)count;
        }
        if (cudaMemcpy((char *)t->device_ptr + completed, host, request, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size) snprintf(error, error_size, "cudaMemcpy failed");
            free(host); close(fd); return 0;
        }
        completed += request;
    }
    free(host); close(fd);
    (void)gpu;
    return 1;
}

static int h3_gpu_tensor_read_file_bf16_mode(h3_gpu_tensor *tensor, const char *path, uint64_t file_offset, size_t elements, int uncached, char *error, size_t error_size) {
    (void)uncached;
    if (error && error_size) error[0] = '\0';
    if (!tensor || !path || !*path || tensor->dtype != H3_GPU_BF16 || elements != tensor->elements || file_offset > (uint64_t)INT64_MAX) {
        if (error && error_size) snprintf(error, error_size, "invalid BF16 file read request");
        return 0;
    }
    size_t bytes = elements * sizeof(uint16_t);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { if (error && error_size) snprintf(error, error_size, "cannot open %s: %s", path, strerror(errno)); return 0; }
    size_t chunk = MIN(bytes, (size_t)(1 << 20));
    void *host = malloc(chunk ? chunk : 1);
    if (!host) { close(fd); return 0; }
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(chunk, bytes - completed);
        size_t got = 0;
        while (got < request) {
            ssize_t count = pread(fd, (char *)host + got, request - got, (off_t)(file_offset + completed + got));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                if (error && error_size) snprintf(error, error_size, "cannot read BF16 payload from %s: %s", path, count < 0 ? strerror(errno) : "unexpected end of file");
                free(host); close(fd); return 0;
            }
            got += (size_t)count;
        }
        if (cudaMemcpy((char *)tensor->device_ptr + completed, host, request, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (error && error_size) snprintf(error, error_size, "cudaMemcpy failed");
            free(host); close(fd); return 0;
        }
        completed += request;
    }
    free(host); close(fd);
    return 1;
}

int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size) {
    int count = 0;
    cudaError_t ce = cudaGetDeviceCount(&count);
    if (ce != cudaSuccess || count < 1) {
        if (error && error_size)
            snprintf(error, error_size, "no CUDA device available: %s", cudaGetErrorString(ce));
        return 0;
    }
    if (info) {
        memset(info, 0, sizeof(*info));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        snprintf(info->name, sizeof(info->name), "%s", prop.name);
        snprintf(info->architecture, sizeof(info->architecture), "sm_%d", prop.major * 100 + prop.minor * 10);
        info->physical_memory = (uint64_t)prop.totalGlobalMem;
        info->unified_memory = 1;
    }
    return 1;
}
h3_gpu *h3_gpu_create(const char *shader_source_path, char *error, size_t error_size) {
    (void)shader_source_path;
    h3_gpu *g = (h3_gpu *)calloc(1, sizeof(*g));
    if (!g) { if (error && error_size) snprintf(error, error_size, "oom"); return NULL; }
    return g;
}
void h3_gpu_free(h3_gpu *gpu) { if (gpu) free(gpu); }
const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu && gpu->error[0] ? gpu->error : "no error";
}
static void h3_cuda_seterr(const h3_gpu *gpu) {
    if (gpu) snprintf(((h3_gpu *)gpu)->error, sizeof(((h3_gpu *)gpu)->error), "%s", H3_CUDA_ERR);
}

int h3_gpu_is_m5(const h3_gpu *gpu) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_has_nax_mlp(const h3_gpu *gpu) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_has_int8_mlp(const h3_gpu *gpu) { h3_cuda_seterr(gpu); return (int)0; }
h3_gpu_tensor * h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_I8, NULL);
}
h3_gpu_tensor * h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, values);
}
h3_gpu_tensor * h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, values);
}
h3_gpu_tensor * h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_alloc(gpu, elements, H3_GPU_U32, values);
}
h3_gpu_tensor * h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_alloc(gpu, elements, H3_GPU_BF16, NULL);
    if (!t) return NULL;
    if (!h3_gpu_tensor_file_load(gpu, t, path, file_offset, NULL, 0)) { h3_gpu_tensor_free(t); return NULL; }
    return t;
}
h3_gpu_tensor * h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_alloc(gpu, elements, H3_GPU_F32, NULL);
    if (!t) return NULL;
    if (!h3_gpu_tensor_file_load(gpu, t, path, file_offset, NULL, 0)) { h3_gpu_tensor_free(t); return NULL; }
    return t;
}
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(tensor, path, file_offset, elements, 0, error, error_size);
}
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(tensor, path, file_offset, elements, 1, error, error_size);
}
void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->device_ptr) cudaFree(tensor->device_ptr);
    if (tensor->owner) {
        tensor->owner->stats.live_bytes = tensor->owner->stats.live_bytes >= tensor->bytes ? tensor->owner->stats.live_bytes - tensor->bytes : 0;
    }
    free(tensor);
}
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_F32 || source_offset > tensor->elements || elements > tensor->elements - source_offset) return 0;
    size_t bytes = elements * sizeof(float);
    return (cudaMemcpy(values, (char *)tensor->device_ptr + source_offset * sizeof(float), bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_BF16 || elements > tensor->elements) return 0;
    size_t bytes = elements * sizeof(uint16_t);
    return (cudaMemcpy(values, tensor->device_ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_F32 || destination_offset > tensor->elements || elements > tensor->elements - destination_offset) return 0;
    size_t bytes = elements * sizeof(float);
    return (cudaMemcpy((char *)tensor->device_ptr + destination_offset * sizeof(float), values, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!tensor || !values || tensor->dtype != H3_GPU_BF16 || destination_offset > tensor->elements || elements > tensor->elements - destination_offset) return 0;
    size_t bytes = elements * sizeof(uint16_t);
    return (cudaMemcpy((char *)tensor->device_ptr + destination_offset * sizeof(uint16_t), values, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 1 : 0;
}
int h3_gpu_begin(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_continue(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_submit(h3_gpu *gpu) {
    (void)gpu;
    return 1;  /* CUDA has no explicit command buffer */
}
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->stats;
    return 1;
}
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    (void)gpu; (void)label;
    /* no-op on CUDA */
}
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    (void)gpu; (void)phase;
    /* no-op on CUDA */
}
int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) { h3_cuda_seterr(gpu); return (int)0; }
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) { h3_cuda_seterr(gpu); return (int)0; }
