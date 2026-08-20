#include "h3_gpu.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>

#ifdef H3_USE_CUDNN
#include <cudnn_frontend.h>
#include <memory>
#include <unordered_map>
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <type_traits>
#include <unistd.h>

#ifdef H3_USE_CUDNN
namespace h3_fe = cudnn_frontend;

struct h3_cudnn_sdpa {
    uint32_t sequence;
    uint32_t heads;
    float scale;
    int head_major_output;
    int ready;
    cudnnHandle_t handle;
    std::shared_ptr<h3_fe::graph::Graph> graph;
    void *workspace;
    h3_cudnn_sdpa *next;
};
#endif

struct h3_gpu {
    cudaStream_t stream;
    cublasLtHandle_t blas;
    cudaEvent_t begin_event;
    cudaEvent_t end_event;
    cudaEvent_t continue_event;
    h3_gpu_stats stats;
    char error[512];
    char profile_label[128];
    double profile_mark_time;
    double encode_start_time;
    int recording;
#ifdef H3_USE_CUDNN
    h3_cudnn_sdpa *cudnn_sdpa;
#endif
};

struct h3_gpu_tensor {
    h3_gpu *gpu;
    void *data;
    size_t elements = 0;
    size_t bytes;
    h3_gpu_dtype dtype;
};

static double h3_wall_time(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int h3_set_error(h3_gpu *gpu, const char *format, ...) {
    if (gpu) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(gpu->error, sizeof(gpu->error), format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int h3_cuda_ok(h3_gpu *gpu, cudaError_t status,
                      const char *operation) {
    if (status == cudaSuccess) return 1;
    return h3_set_error(gpu, "%s: %s", operation, cudaGetErrorString(status));
}

#ifdef H3_USE_CUDNN
static void h3_cudnn_sdpa_free(h3_cudnn_sdpa *entry) {
    while (entry) {
        h3_cudnn_sdpa *next = entry->next;
        if (entry->workspace) (void)cudaFree(entry->workspace);
        entry->graph.reset();
        if (entry->handle) (void)cudnnDestroy(entry->handle);
        delete entry;
        entry = next;
    }
}

static h3_cudnn_sdpa *h3_cudnn_sdpa_get(
        h3_gpu *gpu, uint32_t sequence, uint32_t heads, float scale,
        int head_major_output, char *reason, size_t reason_size) {
    for (h3_cudnn_sdpa *entry = gpu->cudnn_sdpa; entry; entry = entry->next)
        if (entry->sequence == sequence && entry->heads == heads &&
            entry->scale == scale &&
            entry->head_major_output == head_major_output)
            return entry;

    h3_cudnn_sdpa *entry = new (std::nothrow) h3_cudnn_sdpa{};
    if (!entry) {
        snprintf(reason, reason_size, "out of memory creating cuDNN SDPA cache");
        return NULL;
    }
    entry->sequence = sequence;
    entry->heads = heads;
    entry->scale = scale;
    entry->head_major_output = head_major_output;
    entry->next = gpu->cudnn_sdpa;
    gpu->cudnn_sdpa = entry;

    try {
        cudnnStatus_t cudnn_status = cudnnCreate(&entry->handle);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            snprintf(reason, reason_size, "cudnnCreate: %s",
                     cudnnGetErrorString(cudnn_status));
            return entry;
        }
        cudnn_status = cudnnSetStream(entry->handle, gpu->stream);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            snprintf(reason, reason_size, "cudnnSetStream: %s",
                     cudnnGetErrorString(cudnn_status));
            return entry;
        }

        enum { Q_UID = 1, K_UID = 2, V_UID = 3, O_UID = 4 };
        int64_t b = 1, h = heads, s = sequence, d = 128;
        entry->graph = std::make_shared<h3_fe::graph::Graph>();
        entry->graph->set_io_data_type(h3_fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(h3_fe::DataType_t::FLOAT)
            .set_compute_data_type(h3_fe::DataType_t::FLOAT);
        auto q = entry->graph->tensor(
            h3_fe::graph::Tensor_attributes()
                .set_name("Q").set_uid(Q_UID)
                .set_dim({b, h, s, d})
                .set_stride({h * s * d, s * d, d, 1}));
        auto k = entry->graph->tensor(
            h3_fe::graph::Tensor_attributes()
                .set_name("K").set_uid(K_UID)
                .set_dim({b, h, s, d})
                .set_stride({h * s * d, s * d, d, 1}));
        auto v = entry->graph->tensor(
            h3_fe::graph::Tensor_attributes()
                .set_name("V").set_uid(V_UID)
                .set_dim({b, h, s, d})
                .set_stride({h * s * d, s * d, d, 1}));
        auto options = h3_fe::graph::SDPA_attributes()
            .set_name("h3_sdpa")
            .set_generate_stats(false)
            .set_attn_scale(scale);
        auto result = entry->graph->sdpa(q, k, v, options);
        auto output = result[0];
        output->set_output(true).set_uid(O_UID).set_dim({b, h, s, d});
        if (head_major_output)
            output->set_stride({h * s * d, s * d, d, 1});
        else
            output->set_stride({h * s * d, d, h * d, 1});

        auto status = entry->graph->build(entry->handle,
                                           {h3_fe::HeurMode_t::A});
        if (!status.is_good()) {
            snprintf(reason, reason_size, "cuDNN graph build: %s",
                     status.get_message().c_str());
            return entry;
        }
        int64_t workspace_size = 0;
        auto workspace_status =
            entry->graph->get_workspace_size(workspace_size);
        if (!workspace_status.is_good() || workspace_size < 0) {
            snprintf(reason, reason_size, "cuDNN workspace query: %s",
                     workspace_status.get_message().c_str());
            return entry;
        }
        if (workspace_size > 0) {
            cudaError_t cuda_status = cudaMalloc(&entry->workspace,
                                                  (size_t)workspace_size);
            if (cuda_status != cudaSuccess) {
                snprintf(reason, reason_size, "cuDNN workspace: %s",
                         cudaGetErrorString(cuda_status));
                return entry;
            }
        }
        entry->ready = 1;
        return entry;
    } catch (const std::exception &error) {
        snprintf(reason, reason_size, "cuDNN frontend: %s", error.what());
        return entry;
    }
}

static int h3_cudnn_sdpa_execute(
        h3_gpu *gpu, h3_cudnn_sdpa *entry, void *output,
        const void *query, const void *key, const void *value,
        char *reason, size_t reason_size) {
    enum { Q_UID = 1, K_UID = 2, V_UID = 3, O_UID = 4 };
    if (!entry || !entry->ready) return 0;
    std::unordered_map<int64_t, void *> pointers = {
        {Q_UID, const_cast<void *>(query)},
        {K_UID, const_cast<void *>(key)},
        {V_UID, const_cast<void *>(value)},
        {O_UID, output},
    };
    cudnnStatus_t cudnn_status = cudnnSetStream(entry->handle, gpu->stream);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        snprintf(reason, reason_size, "cudnnSetStream: %s",
                 cudnnGetErrorString(cudnn_status));
        return 0;
    }
    auto status = entry->graph->execute(entry->handle, pointers,
                                         entry->workspace);
    if (!status.is_good()) {
        snprintf(reason, reason_size, "cuDNN SDPA execute: %s",
                 status.get_message().c_str());
        return 0;
    }
    return 1;
}
#endif

static size_t h3_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_F32: return sizeof(float);
        case H3_GPU_BF16: return sizeof(uint16_t);
        case H3_GPU_I8: return sizeof(int8_t);
        case H3_GPU_U32: return sizeof(uint32_t);
    }
    return 0;
}

static h3_gpu_tensor *h3_tensor_new(h3_gpu *gpu, size_t elements,
                                    h3_gpu_dtype dtype) {
    if (!gpu) return NULL;
    size_t item_size = h3_dtype_size(dtype);
    if (!item_size || elements > SIZE_MAX / item_size) {
        h3_set_error(gpu, "invalid or overflowing tensor size");
        return NULL;
    }
    h3_gpu_tensor *tensor = (h3_gpu_tensor *)calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_set_error(gpu, "out of memory allocating tensor metadata");
        return NULL;
    }
    tensor->gpu = gpu;
    tensor->elements = elements;
    tensor->bytes = elements * item_size;
    tensor->dtype = dtype;
    if (tensor->bytes && !h3_cuda_ok(gpu, cudaMalloc(&tensor->data, tensor->bytes),
                                     "cudaMalloc")) {
        free(tensor);
        return NULL;
    }
    gpu->stats.allocated_bytes += tensor->bytes;
    gpu->stats.live_bytes += tensor->bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    gpu->stats.tensor_allocations++;
    return tensor;
}

static int h3_require_tensor(const h3_gpu_tensor *tensor,
                             h3_gpu_dtype dtype, size_t elements) {
    return tensor && tensor->dtype == dtype && tensor->elements >= elements;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    (void)shader_source_path;
    h3_gpu *gpu = (h3_gpu *)calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return NULL;
    }
    cudaError_t status = cudaStreamCreateWithFlags(&gpu->stream,
                                                    cudaStreamNonBlocking);
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
    if (status == cudaSuccess) blas_status = cublasLtCreate(&gpu->blas);
    if (status == cudaSuccess && blas_status != CUBLAS_STATUS_SUCCESS)
        status = cudaErrorInitializationError;
    if (status == cudaSuccess) status = cudaEventCreate(&gpu->begin_event);
    if (status == cudaSuccess) status = cudaEventCreate(&gpu->end_event);
    if (status == cudaSuccess) status = cudaEventCreate(&gpu->continue_event);
    if (status != cudaSuccess) {
        if (error && error_size)
            snprintf(error, error_size, "CUDA initialization: %s",
                     cudaGetErrorString(status));
        if (gpu->continue_event) cudaEventDestroy(gpu->continue_event);
        if (gpu->end_event) cudaEventDestroy(gpu->end_event);
        if (gpu->begin_event) cudaEventDestroy(gpu->begin_event);
        if (gpu->blas) cublasLtDestroy(gpu->blas);
        if (gpu->stream) cudaStreamDestroy(gpu->stream);
        free(gpu);
        return NULL;
    }
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "CUDA context");
    gpu->profile_mark_time = h3_wall_time();
    if (error && error_size) error[0] = '\0';
    return gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    cudaError_t status = cudaStreamSynchronize(gpu->stream);
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "%s: %.6fs GPU, %.6fs encode, %.6fs wait, "
                "%llu bytes peak, %llu submissions%s%s\n",
                gpu->profile_label, gpu->stats.gpu_seconds,
                gpu->stats.command_encode_seconds,
                gpu->stats.command_wait_seconds,
                (unsigned long long)gpu->stats.peak_live_bytes,
                (unsigned long long)gpu->stats.submissions,
                status == cudaSuccess ? "" : ", teardown error: ",
                status == cudaSuccess ? "" : cudaGetErrorString(status));
    }
    (void)cudaEventDestroy(gpu->continue_event);
    (void)cudaEventDestroy(gpu->end_event);
    (void)cudaEventDestroy(gpu->begin_event);
#ifdef H3_USE_CUDNN
    h3_cudnn_sdpa_free(gpu->cudnn_sdpa);
#endif
    (void)cublasLtDestroy(gpu->blas);
    (void)cudaStreamDestroy(gpu->stream);
    free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_nax_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_int8_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_tensor_new(gpu, elements, H3_GPU_F32);
}
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_tensor_new(gpu, elements, H3_GPU_BF16);
}
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_tensor_new(gpu, elements, H3_GPU_I8);
}

static h3_gpu_tensor *h3_tensor_from(h3_gpu *gpu, const void *values,
                                     size_t elements, h3_gpu_dtype dtype) {
    if (elements && !values) {
        h3_set_error(gpu, "tensor source is required");
        return NULL;
    }
    h3_gpu_tensor *tensor = h3_tensor_new(gpu, elements, dtype);
    if (!tensor) return NULL;
    if (tensor->bytes && !h3_cuda_ok(gpu, cudaMemcpy(tensor->data, values,
            tensor->bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_tensor_from(gpu, values, elements, H3_GPU_F32);
}
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_tensor_from(gpu, values, elements, H3_GPU_BF16);
}
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                       size_t elements) {
    return h3_tensor_from(gpu, values, elements, H3_GPU_U32);
}

static int h3_read_file(h3_gpu_tensor *tensor, const char *path,
                        uint64_t file_offset, size_t elements, int streaming,
                        char *error, size_t error_size) {
    if (!tensor || !path || tensor->dtype != H3_GPU_BF16 ||
        elements > tensor->elements || file_offset > (uint64_t)INT64_MAX) {
        if (error && error_size) snprintf(error, error_size, "invalid BF16 file read");
        return 0;
    }
    size_t bytes = elements * sizeof(uint16_t);
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        if (error && error_size) snprintf(error, error_size, "BF16 file range overflows off_t");
        return 0;
    }
    void *staging = bytes ? malloc(bytes) : NULL;
    if (bytes && !staging) {
        if (error && error_size) snprintf(error, error_size, "cannot allocate pinned staging buffer");
        return 0;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (error && error_size) snprintf(error, error_size, "cannot open %s: %s", path, strerror(errno));
        free(staging);
        return 0;
    }
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, (char *)staging + done, bytes - done,
                            (off_t)(file_offset + done));
        if (got <= 0) {
            if (error && error_size) snprintf(error, error_size, "short read from %s", path);
            close(fd);
            free(staging);
            return 0;
        }
        done += (size_t)got;
    }
    if (streaming && bytes)
        (void)posix_fadvise(fd, (off_t)file_offset, (off_t)bytes,
                           POSIX_FADV_DONTNEED);
    close(fd);
    cudaError_t status = bytes ? cudaMemcpy(tensor->data, staging, bytes,
        cudaMemcpyHostToDevice) : cudaSuccess;
    free(staging);
    if (status != cudaSuccess) {
        if (error && error_size) snprintf(error, error_size, "CUDA file upload: %s", cudaGetErrorString(status));
        return 0;
    }
    if (error && error_size) error[0] = '\0';
    return 1;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    char error[256];
    if (tensor && !h3_read_file(tensor, path, file_offset, elements, 0,
                                error, sizeof(error))) {
        h3_set_error(gpu, "%s", error);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    if (!gpu || !path || elements > SIZE_MAX / sizeof(float) ||
        file_offset > (uint64_t)INT64_MAX) {
        h3_set_error(gpu, "invalid F32 file read");
        return NULL;
    }
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, elements);
    if (!tensor) return NULL;
    size_t bytes = elements * sizeof(float);
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        h3_set_error(gpu, "F32 file range overflows off_t");
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    void *staging = bytes ? malloc(bytes) : NULL;
    if (bytes && !staging) {
        h3_set_error(gpu, "cannot allocate pinned staging buffer");
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        h3_set_error(gpu, "cannot open %s: %s", path, strerror(errno));
        free(staging);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, (char *)staging + done, bytes - done,
                            (off_t)(file_offset + done));
        if (got <= 0) break;
        done += (size_t)got;
    }
    close(fd);
    cudaError_t status = cudaSuccess;
    if (done != bytes) status = cudaErrorInvalidValue;
    else if (bytes) status = cudaMemcpy(tensor->data, staging, bytes,
                                        cudaMemcpyHostToDevice);
    free(staging);
    if (done != bytes || status != cudaSuccess) {
        h3_set_error(gpu, "cannot load F32 tensor from %s", path);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_read_file(tensor, path, file_offset, elements, 0, error, error_size);
}
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_read_file(tensor, path, file_offset, elements, 1, error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->data) cudaFree(tensor->data);
    if (tensor->gpu && tensor->gpu->stats.live_bytes >= tensor->bytes)
        tensor->gpu->stats.live_bytes -= tensor->bytes;
    free(tensor);
}
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

__global__ static void h3_bf16_to_f32_kernel(float *output,
                                              const __nv_bfloat16 *input,
                                              size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __bfloat162float(input[index]);
}
__global__ static void h3_f32_to_bf16_kernel(__nv_bfloat16 *output,
                                              const float *input,
                                              size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2bfloat16(input[index]);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    if (!tensor || !values || source_offset > tensor->elements ||
        elements > tensor->elements - source_offset)
        return h3_set_error(tensor ? tensor->gpu : NULL, "invalid F32 tensor read range");
    if (tensor->dtype == H3_GPU_F32) {
        return h3_cuda_ok(tensor->gpu, cudaMemcpy(values,
            (const float *)tensor->data + source_offset,
            elements * sizeof(float), cudaMemcpyDeviceToHost), "read F32 tensor");
    }
    if (tensor->dtype != H3_GPU_BF16)
        return h3_set_error(tensor->gpu, "tensor is neither F32 nor BF16");
    float *temporary = NULL;
    if (elements && cudaMalloc(&temporary, elements * sizeof(float)) != cudaSuccess)
        return h3_set_error(tensor->gpu, "cannot allocate BF16 conversion buffer");
    if (elements) h3_bf16_to_f32_kernel<<<(elements + 255) / 256, 256, 0,
        tensor->gpu->stream>>>(temporary,
        (const __nv_bfloat16 *)tensor->data + source_offset, elements);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && elements) status = cudaMemcpyAsync(values, temporary,
        elements * sizeof(float), cudaMemcpyDeviceToHost, tensor->gpu->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(tensor->gpu->stream);
    if (temporary) cudaFree(temporary);
    return h3_cuda_ok(tensor->gpu, status, "read BF16 tensor as F32");
}
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    if (!h3_require_tensor(tensor, H3_GPU_BF16, elements) || !values)
        return h3_set_error(tensor ? tensor->gpu : NULL, "invalid BF16 tensor read");
    return h3_cuda_ok(tensor->gpu, cudaMemcpy(values, tensor->data,
        elements * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read BF16 tensor");
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    if (!tensor || !values || destination_offset > tensor->elements ||
        elements > tensor->elements - destination_offset)
        return h3_set_error(tensor ? tensor->gpu : NULL, "invalid F32 tensor write range");
    if (tensor->dtype == H3_GPU_F32) {
        return h3_cuda_ok(tensor->gpu, cudaMemcpy(
            (float *)tensor->data + destination_offset, values,
            elements * sizeof(float), cudaMemcpyHostToDevice), "write F32 tensor");
    }
    if (tensor->dtype != H3_GPU_BF16)
        return h3_set_error(tensor->gpu, "tensor is neither F32 nor BF16");
    float *temporary = NULL;
    if (elements && cudaMalloc(&temporary, elements * sizeof(float)) != cudaSuccess)
        return h3_set_error(tensor->gpu, "cannot allocate BF16 conversion buffer");
    cudaError_t status = elements ? cudaMemcpyAsync(temporary, values,
        elements * sizeof(float), cudaMemcpyHostToDevice, tensor->gpu->stream) : cudaSuccess;
    if (status == cudaSuccess && elements) h3_f32_to_bf16_kernel<<<
        (elements + 255) / 256, 256, 0, tensor->gpu->stream>>>(
        (__nv_bfloat16 *)tensor->data + destination_offset, temporary, elements);
    if (status == cudaSuccess) status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaStreamSynchronize(tensor->gpu->stream);
    if (temporary) cudaFree(temporary);
    return h3_cuda_ok(tensor->gpu, status, "write F32 as BF16");
}
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    if (!tensor || tensor->dtype != H3_GPU_BF16 || !values ||
        destination_offset > tensor->elements ||
        elements > tensor->elements - destination_offset)
        return h3_set_error(tensor ? tensor->gpu : NULL, "invalid BF16 tensor write range");
    return h3_cuda_ok(tensor->gpu, cudaMemcpy(
        (uint16_t *)tensor->data + destination_offset, values,
        elements * sizeof(uint16_t), cudaMemcpyHostToDevice), "write BF16 tensor");
}
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu || gpu->recording) return h3_set_error(gpu, "command stream is already active");
    gpu->error[0] = '\0';
    cudaError_t status = cudaEventRecord(gpu->begin_event, gpu->stream);
    if (status != cudaSuccess)
        return h3_cuda_ok(gpu, status, "record begin event");
    gpu->recording = 1;
    gpu->encode_start_time = h3_wall_time();
    return 1;
}
int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu || !gpu->recording) return h3_set_error(gpu, "no active command stream");
    cudaError_t status = cudaEventRecord(gpu->continue_event, gpu->stream);
    if (status == cudaSuccess)
        status = cudaStreamWaitEvent(gpu->stream, gpu->continue_event, 0);
    if (status == cudaSuccess) gpu->stats.submissions++;
    return h3_cuda_ok(gpu, status, "record CUDA continuation boundary");
}
int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu || !gpu->recording) return h3_set_error(gpu, "no active command stream");
    double wait_start = h3_wall_time();
    gpu->stats.command_encode_seconds += wait_start - gpu->encode_start_time;
    cudaError_t status = cudaEventRecord(gpu->end_event, gpu->stream);
    if (status == cudaSuccess) status = cudaEventSynchronize(gpu->end_event);
    gpu->stats.command_wait_seconds += h3_wall_time() - wait_start;
    if (status == cudaSuccess) {
        float milliseconds = 0.0f;
        status = cudaEventElapsedTime(&milliseconds, gpu->begin_event,
                                      gpu->end_event);
        gpu->stats.gpu_seconds += (double)milliseconds / 1000.0;
    }
    gpu->recording = 0;
    if (status == cudaSuccess) gpu->stats.submissions++;
    return h3_cuda_ok(gpu, status, "submit CUDA stream");
}
const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "CUDA context is null";
}
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->stats;
    return 1;
}
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu) return;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%.127s",
             label ? label : "CUDA context");
}
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    if (!gpu || !getenv("H3_PROFILE")) return;
    cudaError_t status = cudaStreamSynchronize(gpu->stream);
    if (status != cudaSuccess) {
        h3_cuda_ok(gpu, status, "profile stream synchronization");
        return;
    }
    double now = h3_wall_time();
    fprintf(stderr, "%s: %s %.6fs\n", gpu->profile_label,
            phase ? phase : "mark", now - gpu->profile_mark_time);
    gpu->profile_mark_time = now;
}

static int h3_copy(h3_gpu *gpu, h3_gpu_tensor *destination,
                   size_t destination_offset, const h3_gpu_tensor *source,
                   size_t source_offset, size_t elements,
                   h3_gpu_dtype dtype) {
    if (!gpu || !destination || !source || destination->gpu != gpu ||
        source->gpu != gpu || destination->dtype != dtype ||
        source->dtype != dtype || destination_offset > destination->elements ||
        source_offset > source->elements ||
        elements > destination->elements - destination_offset ||
        elements > source->elements - source_offset)
        return h3_set_error(gpu, "invalid tensor copy");
    size_t item_size = h3_dtype_size(dtype);
    cudaError_t status = cudaMemcpyAsync(
        (char *)destination->data + destination_offset * item_size,
        (const char *)source->data + source_offset * item_size,
        elements * item_size, cudaMemcpyDeviceToDevice, gpu->stream);
    if (status == cudaSuccess) gpu->stats.blit_copies++;
    return h3_cuda_ok(gpu, status, "CUDA tensor copy");
}
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    return h3_copy(gpu, destination, destination_offset, source, source_offset,
                   elements, H3_GPU_BF16);
}
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    return h3_copy(gpu, destination, destination_offset, source, source_offset,
                   elements, H3_GPU_F32);
}

static int h3_launch_ok(h3_gpu *gpu, const char *operation) {
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, status, operation);
}

static int h3_tensor_is(const h3_gpu_tensor *tensor, const h3_gpu *gpu,
                        h3_gpu_dtype dtype, size_t elements) {
    return tensor && tensor->gpu == gpu && tensor->dtype == dtype &&
           tensor->elements >= elements;
}

static int h3_mul_size(size_t left, size_t right, size_t *result) {
    if (left && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int h3_blas_ok(h3_gpu *gpu, cublasStatus_t status,
                      const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    return h3_set_error(gpu, "%s: cuBLASLt status %d", operation,
                        (int)status);
}

static int h3_linear_lt(h3_gpu *gpu, void *output, const void *input,
                        const void *weight, uint32_t rows,
                        uint32_t input_dim, uint32_t output_dim,
                        cudaDataType_t input_type, cudaDataType_t weight_type,
                        cudaDataType_t output_type) {
    cublasLtMatmulDesc_t operation = NULL;
    cublasLtMatrixLayout_t input_layout = NULL;
    cublasLtMatrixLayout_t weight_layout = NULL;
    cublasLtMatrixLayout_t output_layout = NULL;
    cublasOperation_t transpose = CUBLAS_OP_T;
    cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
            sizeof(transpose));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &input_layout, input_type, rows, input_dim, input_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &weight_layout, weight_type, output_dim, input_dim, input_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutCreate(
            &output_layout, output_type, rows, output_dim, output_dim);
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatrixLayoutSetAttribute(
            output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
            sizeof(row_major));
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmul(gpu->blas, operation, &alpha, input,
            input_layout, weight, weight_layout, &beta, output, output_layout,
            output, output_layout, NULL, NULL, 0, gpu->stream);
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (status == CUBLAS_STATUS_SUCCESS) gpu->stats.mps_linear_dispatches++;
    return h3_blas_ok(gpu, status, "cuBLASLt linear");
}

template <typename T>
__global__ static void h3_linear_bias_kernel(T *output, const T *bias,
                                              size_t elements,
                                              uint32_t columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = (float)output[index] + (float)bias[index % columns];
    output[index] = (T)value;
}

static int h3_linear_bias(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *bias, size_t elements,
                          uint32_t columns) {
    if (!bias) return 1;
    unsigned blocks = (unsigned)((elements + 255) / 256);
    if (output->dtype == H3_GPU_F32)
        h3_linear_bias_kernel<<<blocks, 256, 0, gpu->stream>>>(
            (float *)output->data, (const float *)bias->data, elements,
            columns);
    else
        h3_linear_bias_kernel<<<blocks, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)bias->data, elements, columns);
    return h3_launch_ok(gpu, "linear bias");
}

__global__ static void h3_patch_convert_kernel(__nv_bfloat16 *output,
                                                const float *input,
                                                const float *bias,
                                                size_t elements,
                                                uint32_t columns,
                                                int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = input[index];
    if (has_bias) value += bias[index % columns];
    output[index] = __float2bfloat16(value);
}

static int h3_patch_convert(h3_gpu *gpu, __nv_bfloat16 *output,
                            const float *input,
                            const h3_gpu_tensor *bias, size_t elements,
                            uint32_t columns) {
    unsigned blocks = (unsigned)((elements + 255) / 256);
    h3_patch_convert_kernel<<<blocks, 256, 0, gpu->stream>>>(
        output, input, bias ? (const float *)bias->data : input, elements,
        columns, bias != NULL);
    return h3_launch_ok(gpu, "patch linear conversion");
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim) {
    size_t inputs, weights, outputs;
    if (!rows || !input_dim || !output_dim ||
        !h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(rows, output_dim, &outputs) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, inputs) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weights) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, outputs) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_dim)))
        return h3_set_error(gpu, "invalid F32 linear tensors or shape");
    return h3_linear_lt(gpu, output->data, input->data, weight->data, rows,
                        input_dim, output_dim, CUDA_R_32F, CUDA_R_32F,
                        CUDA_R_32F) &&
           h3_linear_bias(gpu, output, bias, outputs, output_dim);
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim) {
    size_t inputs, weights, outputs;
    if (!rows || !input_dim || !output_dim ||
        !h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(rows, output_dim, &outputs) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, inputs) ||
        !h3_tensor_is(weight, gpu, H3_GPU_BF16, weights) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, outputs) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_BF16, output_dim)))
        return h3_set_error(gpu, "invalid BF16 linear tensors or shape");
    return h3_linear_lt(gpu, output->data, input->data, weight->data, rows,
                        input_dim, output_dim, CUDA_R_16BF, CUDA_R_16BF,
                        CUDA_R_16BF) &&
           h3_linear_bias(gpu, output, bias, outputs, output_dim);
}

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
        size_t output_offset, const h3_gpu_tensor *input,
        size_t input_offset, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim) {
    size_t inputs, weights, outputs;
    if (!rows || output_dim != 5376 ||
        (input_dim != 32 && input_dim != 96) ||
        !h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(rows, output_dim, &outputs) ||
        input_offset > SIZE_MAX - inputs ||
        output_offset > SIZE_MAX - outputs ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, input_offset + inputs) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weights) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, output_offset + outputs) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_dim)))
        return h3_set_error(gpu, "invalid patch linear tensors or shape");
    const float *input_data = (const float *)input->data + input_offset;
    __nv_bfloat16 *output_data =
        (__nv_bfloat16 *)output->data + output_offset;
    float *temporary = NULL;
    cudaError_t status = cudaMallocAsync((void **)&temporary,
        outputs * sizeof(*temporary), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "patch linear temporary allocation"))
        return 0;
    int ok = h3_linear_lt(gpu, temporary, input_data, weight->data, rows,
                          input_dim, output_dim, CUDA_R_32F, CUDA_R_32F,
                          CUDA_R_32F) &&
             h3_patch_convert(gpu, output_data, temporary, bias, outputs,
                              output_dim);
    status = cudaFreeAsync(temporary, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "patch linear temporary free");
    return ok;
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                            bias, rows, input_dim,
                                            output_dim);
}

__global__ static void h3_patch_scatter_kernel(
        __nv_bfloat16 *output, const float *input, const float *bias,
        const uint32_t *row_map, uint32_t output_rows, uint32_t rows,
        uint32_t columns, int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)rows * columns;
    if (index >= elements) return;
    uint32_t source_row = (uint32_t)(index / columns);
    uint32_t destination_row = row_map[source_row];
    if (destination_row < output_rows)
        output[(size_t)destination_row * columns + index % columns] =
            __float2bfloat16(input[index] +
                (has_bias ? bias[index % columns] : 0.0f));
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, const h3_gpu_tensor *row_map,
        uint32_t output_rows, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim) {
    size_t inputs, weights, outputs, mapped_outputs;
    if (!rows || !output_rows || output_dim != 5376 ||
        (input_dim != 32 && input_dim != 96) ||
        !h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(output_rows, output_dim, &outputs) ||
        !h3_mul_size(rows, output_dim, &mapped_outputs) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, inputs) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weights) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, outputs) ||
        !h3_tensor_is(row_map, gpu, H3_GPU_U32, rows) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_dim)))
        return h3_set_error(gpu, "invalid mapped patch linear tensors or shape");
    float *temporary = NULL;
    cudaError_t status = cudaMallocAsync((void **)&temporary,
        mapped_outputs * sizeof(*temporary), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "mapped patch temporary allocation"))
        return 0;
    int ok = h3_linear_lt(gpu, temporary, input->data, weight->data, rows,
                          input_dim, output_dim, CUDA_R_32F, CUDA_R_32F,
                          CUDA_R_32F);
    if (ok) {
        unsigned blocks = (unsigned)((mapped_outputs + 255) / 256);
        h3_patch_scatter_kernel<<<blocks, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data, temporary,
            bias ? (const float *)bias->data : temporary,
            (const uint32_t *)row_map->data, output_rows, rows, output_dim,
            bias != NULL);
        ok = h3_launch_ok(gpu, "mapped patch scatter");
    }
    status = cudaFreeAsync(temporary, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "mapped patch temporary free");
    return ok;
}

__global__ static void h3_quantize_rows_kernel(
        int8_t *output, float *scales, const __nv_bfloat16 *input,
        uint32_t rows, uint32_t columns) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t base = (size_t)row * columns;
    float maximum = 0.0f;
    for (uint32_t column = 0; column < columns; column++)
        maximum = fmaxf(maximum, fabsf(__bfloat162float(input[base + column])));
    float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f / 127.0f;
    float inverse = 1.0f / scale;
    scales[row] = scale;
    for (uint32_t column = 0; column < columns; column++) {
        int value = (int)nearbyintf(
            __bfloat162float(input[base + column]) * inverse);
        output[base + column] = (int8_t)max(-127, min(127, value));
    }
}

static int h3_quantize_rows(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *scales,
                            const h3_gpu_tensor *input, uint32_t rows,
                            uint32_t columns) {
    size_t elements;
    if (!rows || !columns || !h3_mul_size(rows, columns, &elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_I8, elements) ||
        !h3_tensor_is(scales, gpu, H3_GPU_F32, rows))
        return h3_set_error(gpu, "invalid INT8 quantization tensors or shape");
    h3_quantize_rows_kernel<<<(rows + 127) / 128, 128, 0, gpu->stream>>>(
        (int8_t *)output->data, (float *)scales->data,
        (const __nv_bfloat16 *)input->data, rows, columns);
    return h3_launch_ok(gpu, "BF16 row quantization");
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
        uint32_t columns) {
    return h3_quantize_rows(gpu, output, scales, input, rows, columns);
}

__global__ static void h3_linear_int8_kernel(
        __nv_bfloat16 *output, const int8_t *input, const int8_t *weight,
        const float *input_scales, const float *weight_scales,
        uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (column >= output_dim || row >= rows) return;
    int32_t sum = 0;
    size_t input_base = (size_t)row * input_dim;
    size_t weight_base = (size_t)column * input_dim;
    for (uint32_t inner = 0; inner < input_dim; inner++)
        sum += (int32_t)input[input_base + inner] *
               (int32_t)weight[weight_base + inner];
    output[(size_t)row * output_dim + column] = __float2bfloat16(
        (float)sum * input_scales[row] * weight_scales[column]);
}

static int h3_linear_int8_quantized(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *quantized_input,
        const h3_gpu_tensor *input_scales, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *weight_scales, uint32_t rows,
        uint32_t input_dim, uint32_t output_dim) {
    size_t inputs, weights, outputs;
    if (!h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(rows, output_dim, &outputs) ||
        !h3_tensor_is(quantized_input, gpu, H3_GPU_I8, inputs) ||
        !h3_tensor_is(input_scales, gpu, H3_GPU_F32, rows) ||
        !h3_tensor_is(weight, gpu, H3_GPU_I8, weights) ||
        !h3_tensor_is(weight_scales, gpu, H3_GPU_F32, output_dim) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, outputs))
        return h3_set_error(gpu, "invalid quantized INT8 linear tensors");
    dim3 grid((output_dim + 127) / 128, rows);
    h3_linear_int8_kernel<<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data, (const int8_t *)quantized_input->data,
        (const int8_t *)weight->data, (const float *)input_scales->data,
        (const float *)weight_scales->data, rows, input_dim, output_dim);
    return h3_launch_ok(gpu, "INT8 linear");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *weight_scales, uint32_t rows,
        uint32_t input_dim, uint32_t output_dim,
        int use_slower_uncached_int8_scales) {
    (void)use_slower_uncached_int8_scales;
    size_t inputs, weights, outputs;
    if (!rows || !input_dim || !output_dim ||
        !h3_mul_size(rows, input_dim, &inputs) ||
        !h3_mul_size(output_dim, input_dim, &weights) ||
        !h3_mul_size(rows, output_dim, &outputs) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, inputs) ||
        !h3_tensor_is(quantized_input, gpu, H3_GPU_I8, inputs) ||
        !h3_tensor_is(input_scales, gpu, H3_GPU_F32, rows) ||
        !h3_tensor_is(weight, gpu, H3_GPU_I8, weights) ||
        !h3_tensor_is(weight_scales, gpu, H3_GPU_F32, output_dim) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, outputs))
        return h3_set_error(gpu, "invalid INT8 linear tensors or shape");
    if (!h3_quantize_rows(gpu, quantized_input, input_scales, input, rows,
                          input_dim)) return 0;
    return h3_linear_int8_quantized(gpu, output, quantized_input,
        input_scales, weight, weight_scales, rows, input_dim, output_dim);
}

__global__ static void h3_quantize_head_major_kernel(
        int8_t *output, float *scales, const __nv_bfloat16 *input,
        uint32_t rows, uint32_t heads, uint32_t head_dim) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t columns = heads * head_dim;
    float maximum = 0.0f;
    for (uint32_t head = 0; head < heads; head++)
        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
            size_t source = ((size_t)head * rows + row) * head_dim + dimension;
            maximum = fmaxf(maximum, fabsf(__bfloat162float(input[source])));
        }
    float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f / 127.0f;
    float inverse = 1.0f / scale;
    scales[row] = scale;
    for (uint32_t head = 0; head < heads; head++)
        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
            size_t source = ((size_t)head * rows + row) * head_dim + dimension;
            int value = (int)nearbyintf(
                __bfloat162float(input[source]) * inverse);
            output[(size_t)row * columns + (size_t)head * head_dim + dimension] =
                (int8_t)max(-127, min(127, value));
        }
}

int h3_gpu_linear_int8_head_major_bf16(h3_gpu *gpu,
        h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
        h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
        uint32_t rows, uint32_t heads, uint32_t head_dim,
        uint32_t output_dim) {
    size_t columns, inputs;
    if (!rows || !heads || !head_dim || !output_dim ||
        !h3_mul_size(heads, head_dim, &columns) || columns > UINT32_MAX ||
        !h3_mul_size(rows, columns, &inputs) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, inputs) ||
        !h3_tensor_is(quantized_input, gpu, H3_GPU_I8, inputs) ||
        !h3_tensor_is(input_scales, gpu, H3_GPU_F32, rows))
        return h3_set_error(gpu, "invalid head-major INT8 linear tensors");
    h3_quantize_head_major_kernel<<<(rows + 127) / 128, 128, 0,
        gpu->stream>>>((int8_t *)quantized_input->data,
        (float *)input_scales->data, (const __nv_bfloat16 *)input->data,
        rows, heads, head_dim);
    if (!h3_launch_ok(gpu, "head-major INT8 quantization")) return 0;
    return h3_linear_int8_quantized(gpu, output, quantized_input,
        input_scales, weight, weight_scales, rows, (uint32_t)columns,
        output_dim);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *activated, h3_gpu_tensor *quantized_activation,
        h3_gpu_tensor *activation_scales, const h3_gpu_tensor *input,
        const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc1_scales,
        const h3_gpu_tensor *fc2_weight, const h3_gpu_tensor *fc2_scales,
        const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16,
        uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
        uint32_t output_dim, int use_slower_grouped_quantizer,
        int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
        int input_is_quantized) {
    (void)fc1_bf16;
    (void)fc2_bf16;
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    (void)use_int8_row_fc2;
    size_t fused_elements, activation_elements, input_elements;
    if (!rows || !input_dim || !hidden_dim || !output_dim ||
        hidden_dim > UINT32_MAX / 2 ||
        !h3_mul_size(rows, (size_t)hidden_dim * 2, &fused_elements) ||
        !h3_mul_size(rows, hidden_dim, &activation_elements) ||
        !h3_mul_size(rows, input_dim, &input_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, input_elements) ||
        !h3_tensor_is(activated, gpu, H3_GPU_BF16, activation_elements) ||
        !h3_tensor_is(quantized_activation, gpu, H3_GPU_I8,
                      input_elements > activation_elements ?
                      input_elements : activation_elements) ||
        !h3_tensor_is(activation_scales, gpu, H3_GPU_F32, rows))
        return h3_set_error(gpu, "invalid INT8 MLP activation tensors");
    __nv_bfloat16 *fused_data = NULL;
    cudaError_t status = cudaMallocAsync((void **)&fused_data,
        fused_elements * sizeof(*fused_data), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "INT8 MLP temporary allocation")) return 0;
    h3_gpu_tensor fused = {gpu, fused_data, fused_elements,
                           fused_elements * sizeof(*fused_data), H3_GPU_BF16};
    int ok;
    if (input_is_quantized)
        ok = h3_linear_int8_quantized(gpu, &fused, quantized_activation,
            activation_scales, fc1_weight, fc1_scales, rows, input_dim,
            hidden_dim * 2);
    else
        ok = h3_gpu_linear_int8_bf16(gpu, &fused, quantized_activation,
            activation_scales, input, fc1_weight, fc1_scales, rows, input_dim,
            hidden_dim * 2, 0);
    if (ok)
        ok = h3_gpu_swiglu_bf16(gpu, activated, &fused, rows, hidden_dim);
    if (ok)
        ok = h3_gpu_linear_int8_bf16(gpu, output, quantized_activation,
            activation_scales, activated, fc2_weight, fc2_scales, rows,
            hidden_dim, output_dim, 0);
    status = cudaFreeAsync(fused_data, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "INT8 MLP temporary free");
    return ok;
}

__global__ static void h3_rms_inverse_kernel(
        float *inverse, const __nv_bfloat16 *input, size_t input_offset,
        uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    size_t base = input_offset + (size_t)row * width;
    for (uint32_t column = 0; column < width; column++) {
        float value = __bfloat162float(input[base + column]);
        sum = fmaf(value, value, sum);
    }
    inverse[row] = rsqrtf(sum / (float)width + epsilon);
}

int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *inverse, const h3_gpu_tensor *input,
        size_t input_offset, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t rows, uint32_t width, uint32_t output_dim, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    size_t elements;
    if (!rows || !width || !output_dim || epsilon < 0.0f ||
        !h3_mul_size(rows, width, &elements) ||
        !h3_tensor_is(inverse, gpu, H3_GPU_F32, rows))
        return h3_set_error(gpu, "invalid AdaLN linear inverse or shape");
    __nv_bfloat16 *normalized_data = NULL;
    cudaError_t status = cudaMallocAsync((void **)&normalized_data,
        elements * sizeof(*normalized_data), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "AdaLN linear temporary allocation"))
        return 0;
    h3_gpu_tensor normalized = {gpu, normalized_data, elements,
        elements * sizeof(*normalized_data), H3_GPU_BF16};
    h3_rms_inverse_kernel<<<(rows + 127) / 128, 128, 0, gpu->stream>>>(
        (float *)inverse->data, (const __nv_bfloat16 *)input->data,
        input_offset, rows, width, epsilon);
    int ok = h3_launch_ok(gpu, "AdaLN inverse RMS") &&
        h3_gpu_adaln_bf16_offset(gpu, &normalized, input, input_offset,
            norm_weight, modulation, row_map, rows, width, slots, shift_slot,
            scale_slot, epsilon) &&
        h3_gpu_linear_bf16(gpu, output, &normalized, weight, bias, rows,
                            width, output_dim);
    status = cudaFreeAsync(normalized_data, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "AdaLN linear temporary free");
    return ok;
}

__global__ static void h3_fill_scales_kernel(float *scales, uint32_t begin,
                                              uint32_t end) {
    uint32_t row = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < end) scales[row] = 1.0f;
}

int h3_gpu_gate_adaln_quantize_int8(h3_gpu *gpu,
        h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output,
        h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual,
        const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *gate_modulation,
        const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots,
        uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot,
        float epsilon) {
    size_t elements, padded_elements;
    if (!rows || padded_rows < rows || !width ||
        !h3_mul_size(rows, width, &elements) ||
        !h3_mul_size(padded_rows, width, &padded_elements) ||
        !h3_tensor_is(quantized_output, gpu, H3_GPU_I8, padded_elements) ||
        !h3_tensor_is(quantized_scales, gpu, H3_GPU_F32, padded_rows))
        return h3_set_error(gpu, "invalid fused gate/AdaLN INT8 tensors");
    __nv_bfloat16 *normalized_data = NULL;
    cudaError_t status = cudaMallocAsync((void **)&normalized_data,
        elements * sizeof(*normalized_data), gpu->stream);
    if (status == cudaSuccess && padded_rows > rows)
        status = cudaMemsetAsync((int8_t *)quantized_output->data + elements,
            0, (padded_elements - elements) * sizeof(int8_t), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "fused gate/AdaLN temporary setup")) {
        if (normalized_data) (void)cudaFreeAsync(normalized_data, gpu->stream);
        return 0;
    }
    h3_gpu_tensor normalized = {gpu, normalized_data, elements,
        elements * sizeof(*normalized_data), H3_GPU_BF16};
    int ok = h3_gpu_gate_adaln_bf16(gpu, gated_residual, &normalized,
        residual, branch, norm_weight, gate_modulation, norm_modulation,
        row_map, rows, width, slots, gate_slot, shift_slot, scale_slot,
        epsilon) &&
        h3_quantize_rows(gpu, quantized_output, quantized_scales, &normalized,
                         rows, width);
    if (ok && padded_rows > rows) {
        h3_fill_scales_kernel<<<(padded_rows - rows + 127) / 128, 128, 0,
            gpu->stream>>>((float *)quantized_scales->data, rows, padded_rows);
        ok = h3_launch_ok(gpu, "INT8 padding scales");
    }
    status = cudaFreeAsync(normalized_data, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "fused gate/AdaLN temporary free");
    return ok;
}

int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu *gpu,
        h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
        h3_gpu_tensor *qkv, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm,
        const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
        const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
        float epsilon) {
    size_t inner;
    if (!h3_mul_size(heads, head_dim, &inner) || inner > UINT32_MAX / 3)
        return h3_set_error(gpu, "grouped QKV projection shape overflows");
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                              (uint32_t)inner * 3) &&
        h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
            k_norm, rope_cos, rope_sin, rows, heads, head_dim, rope_half,
            epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_int8(h3_gpu *gpu,
        h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
        h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm,
        const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
        const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon,
        int input_is_quantized, int use_slower_unfused_qkv_rope,
        int use_slower_scalar_qkv_rms,
        int use_slower_uncached_int8_scales) {
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    size_t inner, qkv_elements;
    if (!h3_mul_size(heads, head_dim, &inner) || inner > UINT32_MAX / 3 ||
        !h3_mul_size(rows, inner * 3, &qkv_elements))
        return h3_set_error(gpu, "INT8 grouped QKV shape overflows");
    __nv_bfloat16 *qkv_data = NULL;
    cudaError_t status = cudaMallocAsync((void **)&qkv_data,
        qkv_elements * sizeof(*qkv_data), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "INT8 QKV temporary allocation")) return 0;
    h3_gpu_tensor qkv = {gpu, qkv_data, qkv_elements,
                         qkv_elements * sizeof(*qkv_data), H3_GPU_BF16};
    int ok;
    if (input_is_quantized)
        ok = h3_linear_int8_quantized(gpu, &qkv, quantized_input,
            input_scales, weight, weight_scales, rows, input_dim,
            (uint32_t)inner * 3);
    else
        ok = h3_gpu_linear_int8_bf16(gpu, &qkv, quantized_input,
            input_scales, input, weight, weight_scales, rows, input_dim,
            (uint32_t)inner * 3, use_slower_uncached_int8_scales);
    if (ok)
        ok = h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, &qkv,
            q_norm, k_norm, rope_cos, rope_sin, rows, heads, head_dim,
            rope_half, epsilon);
    status = cudaFreeAsync(qkv_data, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "INT8 QKV temporary free");
    return ok;
}

template <typename T>
__global__ static void h3_attention_kernel(
        T *output, const T *query, const T *key, const T *value,
        uint32_t batch, uint32_t sequence, uint32_t query_heads,
        uint32_t kv_heads, uint32_t head_dim, float scale, int causal,
        int head_major_output, int head_major_input) {
    __shared__ float partial[1024];
    __shared__ float rescale;
    __shared__ float weight;
    __shared__ float denominator;
    uint32_t query_head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t batch_index = blockIdx.z;
    uint32_t kv_head = query_head / (query_heads / kv_heads);
    uint32_t keys = causal ? row + 1 : sequence;
    size_t query_base = head_major_input ?
        (((size_t)batch_index * query_heads + query_head) * sequence + row) *
            head_dim :
        (((size_t)batch_index * sequence + row) * query_heads + query_head) *
            head_dim;
    uint32_t dimension = threadIdx.x;
    float result = 0.0f;
    float maximum = -INFINITY;
    if (threadIdx.x == 0) denominator = 0.0f;
    __syncthreads();
    for (uint32_t key_row = 0; key_row < keys; key_row++) {
        size_t key_base = head_major_input ?
            (((size_t)batch_index * kv_heads + kv_head) * sequence + key_row) *
                head_dim :
            (((size_t)batch_index * sequence + key_row) * kv_heads + kv_head) *
                head_dim;
        float product = 0.0f;
        if (dimension < head_dim) {
            if constexpr (std::is_same<T, float>::value)
                product = query[query_base + dimension] *
                          key[key_base + dimension];
            else
                product = __bfloat162float(query[query_base + dimension]) *
                          __bfloat162float(key[key_base + dimension]);
        }
        partial[threadIdx.x] = product;
        __syncthreads();
        for (uint32_t offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset)
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = partial[0] * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        __syncthreads();
        if (dimension < head_dim) {
            size_t value_index = key_base + dimension;
            float value_element;
            if constexpr (std::is_same<T, float>::value)
                value_element = value[value_index];
            else
                value_element = __bfloat162float(value[value_index]);
            result = result * rescale + weight * value_element;
        }
        __syncthreads();
    }
    if (dimension < head_dim) {
        result /= denominator;
        size_t output_index = head_major_output ?
            (((size_t)batch_index * query_heads + query_head) * sequence + row) *
                head_dim + dimension :
            (((size_t)batch_index * sequence + row) * query_heads + query_head) *
                head_dim + dimension;
        if constexpr (std::is_same<T, float>::value)
            output[output_index] = result;
        else
            output[output_index] = __float2bfloat16(result);
    }
}

__global__ static void h3_attention_tiled_bf16_kernel(
        __nv_bfloat16 *output, const __nv_bfloat16 *query,
        const __nv_bfloat16 *key, const __nv_bfloat16 *value,
        uint32_t sequence, uint32_t heads, float scale,
        int head_major_output) {
    enum { HEAD_DIM = 128, QUERIES = 8 };
    __shared__ float shared_key[HEAD_DIM];
    __shared__ float shared_value[HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y * QUERIES + warp;
    int active = row < sequence;
    size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        if (threadIdx.x < HEAD_DIM) {
            size_t index = ((size_t)head * sequence + key_row) * HEAD_DIM +
                           threadIdx.x;
            shared_key[threadIdx.x] = __bfloat162float(key[index]);
            shared_value[threadIdx.x] = __bfloat162float(value[index]);
        }
        __syncthreads();
        float product = 0.0f;
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < 4; item++) {
                uint32_t dimension = lane + item * 32;
                product += __bfloat162float(query[query_base + dimension]) *
                           shared_key[dimension];
            }
#pragma unroll
            for (uint32_t offset = 16; offset; offset >>= 1)
                product += __shfl_down_sync(0xffffffffu, product, offset);
        }
        float rescale = 1.0f;
        float weight = 0.0f;
        if (active && lane == 0) {
            float score = product * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        rescale = __shfl_sync(0xffffffffu, rescale, 0);
        weight = __shfl_sync(0xffffffffu, weight, 0);
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < 4; item++) {
                uint32_t dimension = lane + item * 32;
                result[item] = result[item] * rescale +
                               weight * shared_value[dimension];
            }
        }
        __syncthreads();
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);
    if (!active) return;
#pragma unroll
    for (uint32_t item = 0; item < 4; item++) {
        uint32_t dimension = lane + item * 32;
        size_t index = head_major_output ?
            ((size_t)head * sequence + row) * HEAD_DIM + dimension :
            ((size_t)row * heads + head) * HEAD_DIM + dimension;
        output[index] = __float2bfloat16(result[item] / denominator);
    }
}

template <int HEAD_DIM>
__global__ static void h3_attention_tiled_f32_kernel(
        float *output, const float *query, const float *key,
        const float *value, uint32_t sequence, uint32_t heads, float scale,
        int head_major_output) {
    enum { QUERIES = 8, ITEMS = HEAD_DIM / 32 };
    __shared__ float shared_key[HEAD_DIM];
    __shared__ float shared_value[HEAD_DIM];
    uint32_t warp = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y * QUERIES + warp;
    int active = row < sequence;
    size_t query_base = ((size_t)head * sequence + row) * HEAD_DIM;
    float result[ITEMS];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) result[item] = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        if (threadIdx.x < HEAD_DIM) {
            size_t index = ((size_t)head * sequence + key_row) * HEAD_DIM +
                           threadIdx.x;
            shared_key[threadIdx.x] = key[index];
            shared_value[threadIdx.x] = value[index];
        }
        __syncthreads();
        float product = 0.0f;
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < ITEMS; item++) {
                uint32_t dimension = lane + item * 32;
                product += query[query_base + dimension] *
                           shared_key[dimension];
            }
#pragma unroll
            for (uint32_t offset = 16; offset; offset >>= 1)
                product += __shfl_down_sync(0xffffffffu, product, offset);
        }
        float rescale = 1.0f;
        float weight = 0.0f;
        if (active && lane == 0) {
            float score = product * scale;
            float next_maximum = fmaxf(maximum, score);
            rescale = expf(maximum - next_maximum);
            weight = expf(score - next_maximum);
            denominator = denominator * rescale + weight;
            maximum = next_maximum;
        }
        rescale = __shfl_sync(0xffffffffu, rescale, 0);
        weight = __shfl_sync(0xffffffffu, weight, 0);
        if (active) {
#pragma unroll
            for (uint32_t item = 0; item < ITEMS; item++) {
                uint32_t dimension = lane + item * 32;
                result[item] = result[item] * rescale +
                               weight * shared_value[dimension];
            }
        }
        __syncthreads();
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);
    if (!active) return;
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        uint32_t dimension = lane + item * 32;
        size_t index = head_major_output ?
            ((size_t)head * sequence + row) * HEAD_DIM + dimension :
            ((size_t)row * heads + head) * HEAD_DIM + dimension;
        output[index] = result[item] / denominator;
    }
}

static int h3_attention_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float scale, h3_gpu_dtype dtype, int causal, int head_major_output) {
    int head_major_input = causal != 2;
    if (causal == 2) causal = 1;
    size_t query_elements, kv_elements;
    size_t batch_sequence, query_rows, kv_rows;
    if (!batch || !sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 1024 ||
        !h3_mul_size(batch, sequence, &batch_sequence) ||
        !h3_mul_size(batch_sequence, query_heads, &query_rows) ||
        !h3_mul_size(batch_sequence, kv_heads, &kv_rows) ||
        !h3_mul_size(query_rows, head_dim, &query_elements) ||
        !h3_mul_size(kv_rows, head_dim, &kv_elements) ||
        !h3_tensor_is(query, gpu, dtype, query_elements) ||
        !h3_tensor_is(key, gpu, dtype, kv_elements) ||
        !h3_tensor_is(value, gpu, dtype, kv_elements) ||
        !h3_tensor_is(output, gpu, dtype, query_elements))
        return h3_set_error(gpu, "invalid attention tensors or shape");
    uint32_t threads = 128;
    while (threads < head_dim) threads <<= 1;
    dim3 grid(query_heads, sequence, batch);
    int used_cudnn = 0;
#ifdef H3_USE_CUDNN
    if (!getenv("H3_DISABLE_CUDNN_ATTENTION") && dtype == H3_GPU_BF16 &&
        batch == 1 && !causal && head_dim == 128 &&
        query_heads == kv_heads && head_major_input) {
        char reason[512] = {0};
        h3_cudnn_sdpa *entry = h3_cudnn_sdpa_get(
            gpu, sequence, query_heads, scale, head_major_output,
            reason, sizeof(reason));
        if (entry && entry->ready) {
            if (!h3_cudnn_sdpa_execute(
                    gpu, entry, output->data, query->data, key->data,
                    value->data, reason, sizeof(reason)))
                return h3_set_error(gpu, "%s", reason);
            used_cudnn = 1;
        } else if (getenv("H3_REQUIRE_CUDNN_ATTENTION")) {
            return h3_set_error(gpu, "%s", reason[0] ? reason :
                                "cuDNN SDPA is unavailable");
        }
    }
#else
    if (getenv("H3_REQUIRE_CUDNN_ATTENTION"))
        return h3_set_error(gpu, "cuDNN SDPA was not enabled at build time");
#endif
    if (used_cudnn) {
        /* The graph is enqueued on the same nonblocking stream. */
    } else if (!getenv("H3_DISABLE_TILED_ATTENTION") &&
        dtype == H3_GPU_BF16 &&
        batch == 1 && !causal && head_dim == 128 &&
        query_heads == kv_heads && head_major_input) {
        dim3 tiled_grid(query_heads, (sequence + 7) / 8, batch);
        h3_attention_tiled_bf16_kernel<<<tiled_grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)query->data,
            (const __nv_bfloat16 *)key->data,
            (const __nv_bfloat16 *)value->data, sequence, query_heads, scale,
            head_major_output);
    } else if (!getenv("H3_DISABLE_TILED_ATTENTION") &&
        dtype == H3_GPU_F32 &&
        batch == 1 && !causal && head_dim == 64 &&
        query_heads == kv_heads && head_major_input) {
        dim3 tiled_grid(query_heads, (sequence + 7) / 8, batch);
        h3_attention_tiled_f32_kernel<64><<<tiled_grid, 256, 0, gpu->stream>>>(
            (float *)output->data, (const float *)query->data,
            (const float *)key->data, (const float *)value->data, sequence,
            query_heads, scale, head_major_output);
    } else if (dtype == H3_GPU_F32)
        h3_attention_kernel<float><<<grid, threads, 0, gpu->stream>>>(
            (float *)output->data, (const float *)query->data,
            (const float *)key->data, (const float *)value->data, batch,
            sequence, query_heads, kv_heads, head_dim, scale, causal,
            head_major_output, head_major_input);
    else
        h3_attention_kernel<__nv_bfloat16><<<grid, threads, 0,
            gpu->stream>>>((__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)query->data,
            (const __nv_bfloat16 *)key->data,
            (const __nv_bfloat16 *)value->data, batch, sequence, query_heads,
            kv_heads, head_dim, scale, causal, head_major_output,
            head_major_input);
    int ok = h3_launch_ok(gpu, "scaled dot-product attention");
    if (ok) gpu->stats.mps_sdpa_dispatches++;
    return ok;
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float scale) {
    return h3_attention_dispatch(gpu, output, query, key, value, 1, sequence,
        heads, heads, head_dim, scale, H3_GPU_F32, 0, 0);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, float scale) {
    return h3_attention_dispatch(gpu, output, query, key, value, batch,
        sequence, heads, heads, head_dim, scale, H3_GPU_F32, 1, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float scale) {
    return h3_attention_dispatch(gpu, output, query, key, value, 1, sequence,
        heads, heads, head_dim, scale, H3_GPU_BF16, 0, 0);
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float scale) {
    return h3_attention_dispatch(gpu, output, query, key, value, 1, sequence,
        heads, heads, head_dim, scale, H3_GPU_BF16, 0, 1);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *query, const h3_gpu_tensor *key,
        const h3_gpu_tensor *value, uint32_t sequence, uint32_t query_heads,
        uint32_t kv_heads, uint32_t head_dim, float scale) {
    return h3_attention_dispatch(gpu, output, query, key, value, 1, sequence,
        query_heads, kv_heads, head_dim, scale, H3_GPU_BF16, 2, 0);
}

__global__ static void h3_conv1d_kernel(float *output, const float *input,
        const float *weight, const float *bias, uint32_t batch,
        uint32_t length, uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel, uint32_t stride, uint32_t padding,
        uint32_t dilation, uint32_t output_length, size_t output_elements,
        int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t output_time = (uint32_t)(row % output_length);
    uint32_t batch_index = (uint32_t)(row / output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t input_time = (int64_t)output_time * stride - padding +
                                 (int64_t)tap * dilation;
            if (input_time < 0 || input_time >= length) continue;
            size_t input_index = ((size_t)batch_index * length +
                                  (size_t)input_time) * input_channels +
                                 input_channel;
            size_t weight_index = ((size_t)output_channel * input_channels +
                                   input_channel) * kernel + tap;
            sum = fmaf(input[input_index], weight[weight_index], sum);
        }
    output[((size_t)batch_index * output_length + output_time) *
           output_channels + output_channel] = sum;
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t batch, uint32_t length,
        uint32_t input_channels, uint32_t output_channels, uint32_t kernel,
        uint32_t stride, uint32_t padding, uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel ? kernel - 1 : 0) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2ull * padding < effective)
        return h3_set_error(gpu, "invalid Conv1d shape");
    uint64_t output_length64 = ((uint64_t)length + 2ull * padding - effective) /
                               stride + 1;
    if (output_length64 > UINT32_MAX)
        return h3_set_error(gpu, "Conv1d output length overflows");
    uint32_t output_length = (uint32_t)output_length64;
    size_t input_elements, weight_elements, output_elements;
    if (!h3_mul_size((size_t)batch * length, input_channels, &input_elements) ||
        !h3_mul_size((size_t)output_channels * input_channels, kernel,
                     &weight_elements) ||
        !h3_mul_size((size_t)batch * output_length, output_channels,
                     &output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, input_elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weight_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, output_elements) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_channels)))
        return h3_set_error(gpu, "invalid Conv1d tensors");
    h3_conv1d_kernel<<<(output_elements + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length, output_elements, bias != NULL);
    int ok = h3_launch_ok(gpu, "Conv1d");
    if (ok) gpu->stats.mps_conv_dispatches++;
    return ok;
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t batch, uint32_t length,
        uint32_t input_channels, uint32_t output_channels, uint32_t kernel,
        uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
        length, input_channels, output_channels, kernel, 1, padding, dilation);
}

__global__ static void h3_conv_transpose1d_kernel(float *output,
        const float *input, const float *weight, const float *bias,
        uint32_t batch, uint32_t length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, uint32_t output_length, size_t output_elements,
        int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t output_time = (uint32_t)(row % output_length);
    uint32_t batch_index = (uint32_t)(row / output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t numerator = (int64_t)output_time + padding - tap;
            if (numerator < 0 || numerator % stride) continue;
            uint64_t input_time = (uint64_t)numerator / stride;
            if (input_time >= length) continue;
            size_t input_index = ((size_t)batch_index * length + input_time) *
                                 input_channels + input_channel;
            size_t weight_index = ((size_t)input_channel * output_channels +
                                   output_channel) * kernel + tap;
            sum = fmaf(input[input_index], weight[weight_index], sum);
        }
    output[((size_t)batch_index * output_length + output_time) *
           output_channels + output_channel] = sum;
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t batch, uint32_t length,
        uint32_t input_channels, uint32_t output_channels, uint32_t kernel,
        uint32_t stride, uint32_t padding) {
    uint64_t full = length ? (uint64_t)(length - 1) * stride + kernel : 0;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || full < 2ull * padding || full - 2ull * padding > UINT32_MAX)
        return h3_set_error(gpu, "invalid ConvTranspose1d shape");
    uint32_t output_length = (uint32_t)(full - 2ull * padding);
    size_t input_elements, weight_elements, output_elements;
    if (!h3_mul_size((size_t)batch * length, input_channels, &input_elements) ||
        !h3_mul_size((size_t)input_channels * output_channels, kernel,
                     &weight_elements) ||
        !h3_mul_size((size_t)batch * output_length, output_channels,
                     &output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, input_elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weight_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, output_elements) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_channels)))
        return h3_set_error(gpu, "invalid ConvTranspose1d tensors");
    h3_conv_transpose1d_kernel<<<(output_elements + 255) / 256, 256, 0,
                                  gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data,
        batch, length, input_channels, output_channels, kernel, stride,
        padding, output_length, output_elements, bias != NULL);
    int ok = h3_launch_ok(gpu, "ConvTranspose1d");
    if (ok) gpu->stats.mps_conv_dispatches++;
    return ok;
}

__global__ static void h3_audio_qkv_split_kernel(float *query, float *key,
        float *value, const float *qkv, const float *q_bias,
        const float *k_bias, const float *v_bias, size_t count,
        uint32_t length, uint32_t heads, uint32_t head_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t width = heads * head_dim;
    uint32_t column = (uint32_t)(index % width);
    size_t row = index / width;
    uint32_t time = (uint32_t)(row % length);
    uint32_t batch_index = (uint32_t)(row / length);
    uint32_t head = column / head_dim;
    uint32_t dimension = column % head_dim;
    size_t output = (((size_t)batch_index * heads + head) * length + time) *
                    head_dim + dimension;
    size_t base = row * width * 3;
    query[output] = qkv[base + column] + q_bias[column];
    key[output] = qkv[base + width + column] + k_bias[column];
    value[output] = qkv[base + width * 2 + column] + v_bias[column];
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias,
        const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length,
        uint32_t heads, uint32_t head_dim) {
    size_t width, count;
    if (!batch || !length || !h3_mul_size(heads, head_dim, &width) || !width ||
        !h3_mul_size((size_t)batch * length, width, &count) ||
        count > SIZE_MAX / 3 || !h3_tensor_is(qkv, gpu, H3_GPU_F32, count * 3) ||
        !h3_tensor_is(q_bias, gpu, H3_GPU_F32, width) ||
        !h3_tensor_is(k_bias, gpu, H3_GPU_F32, width) ||
        !h3_tensor_is(v_bias, gpu, H3_GPU_F32, width) ||
        !h3_tensor_is(query, gpu, H3_GPU_F32, count) ||
        !h3_tensor_is(key, gpu, H3_GPU_F32, count) ||
        !h3_tensor_is(value, gpu, H3_GPU_F32, count))
        return h3_set_error(gpu, "invalid audio QKV tensors or shape");
    h3_audio_qkv_split_kernel<<<(count + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, (const float *)q_bias->data,
        (const float *)k_bias->data, (const float *)v_bias->data, count,
        length, heads, head_dim);
    return h3_launch_ok(gpu, "audio QKV split");
}

__global__ static void h3_audio_pool_kernel(float *output,
        const float *attended, size_t count, uint32_t heads,
        uint32_t head_dim, uint32_t output_dim) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t column = (uint32_t)(index % output_dim);
    size_t row = index / output_dim;
    uint32_t pool = head_dim / output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < heads; head++) {
        size_t base = (row * heads + head) * head_dim + column * pool;
        for (uint32_t item = 0; item < pool; item++) sum += attended[base + item];
    }
    output[index] = sum / (float)(heads * pool);
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *attended, uint32_t batch, uint32_t length,
        uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    size_t input_elements, output_elements;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim ||
        !h3_mul_size((size_t)batch * length * heads, head_dim,
                     &input_elements) ||
        !h3_mul_size((size_t)batch * length, output_dim, &output_elements) ||
        !h3_tensor_is(attended, gpu, H3_GPU_F32, input_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, output_elements))
        return h3_set_error(gpu, "invalid audio attention pool tensors");
    h3_audio_pool_kernel<<<(output_elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data, (const float *)attended->data,
        output_elements, heads, head_dim, output_dim);
    return h3_launch_ok(gpu, "audio attention pool");
}

__global__ static void h3_alias_free_snake_kernel(float *output,
        const float *input, const float *alpha_log, const float *beta_log,
        const float *upsample_filter, const float *downsample_filter,
        uint32_t batch, uint32_t length, uint32_t channels, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = (uint32_t)(index % channels);
    size_t row = index / channels;
    uint32_t time = (uint32_t)(row % length);
    uint32_t batch_index = (uint32_t)(row / length);
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_tap = 0; down_tap < 12; down_tap++) {
        int up_time = max(0, min((int)length * 2 - 1,
                                 (int)time * 2 + down_tap - 5));
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_tap = 0; up_tap < 12; up_tap++) {
            int numerator = raw_time - up_tap;
            if (numerator < 0 || (numerator & 1)) continue;
            int source_time = max(0, min((int)length - 1,
                                         numerator / 2 - 5));
            size_t source = ((size_t)batch_index * length + source_time) *
                            channels + channel;
            upsampled = fmaf(input[source], 2.0f * upsample_filter[up_tap],
                              upsampled);
        }
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_tap], result);
    }
    output[((size_t)batch_index * length + time) * channels + channel] = result;
}

int h3_gpu_alias_free_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *alpha_log,
        const h3_gpu_tensor *beta_log,
        const h3_gpu_tensor *upsample_filter,
        const h3_gpu_tensor *downsample_filter, uint32_t batch,
        uint32_t length, uint32_t channels) {
    size_t elements;
    if (!batch || !length || !channels ||
        !h3_mul_size((size_t)batch * length, channels, &elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(alpha_log, gpu, H3_GPU_F32, channels) ||
        !h3_tensor_is(beta_log, gpu, H3_GPU_F32, channels) ||
        !h3_tensor_is(upsample_filter, gpu, H3_GPU_F32, 12) ||
        !h3_tensor_is(downsample_filter, gpu, H3_GPU_F32, 12))
        return h3_set_error(gpu, "invalid alias-free Snake tensors");
    h3_alias_free_snake_kernel<<<(elements + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)alpha_log->data, (const float *)beta_log->data,
        (const float *)upsample_filter->data,
        (const float *)downsample_filter->data, batch, length, channels,
        elements);
    return h3_launch_ok(gpu, "alias-free Snake");
}

__device__ static int h3_reflect(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

__global__ static void h3_vae_pad_kernel(float *output, const float *input,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t depth_front, uint32_t height_before,
        uint32_t height_after, uint32_t width_before, uint32_t width_after) {
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t output_x = blockIdx.y;
    uint32_t plane = blockIdx.z;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    if (channel >= channels || output_x >= output_width ||
        plane >= batch * output_depth * output_height) return;
    uint32_t output_y = plane % output_height;
    uint32_t temporal = plane / output_height;
    uint32_t output_t = temporal % output_depth;
    uint32_t batch_index = temporal / output_depth;
    size_t destination = ((((size_t)batch_index * output_depth + output_t) *
        output_height + output_y) * output_width + output_x) * channels +
        channel;
    if (output_t < depth_front) { output[destination] = 0.0f; return; }
    int source_y = h3_reflect((int)output_y - (int)height_before, (int)height);
    int source_x = h3_reflect((int)output_x - (int)width_before, (int)width);
    uint32_t source_t = output_t - depth_front;
    size_t source = ((((size_t)batch_index * depth + source_t) * height +
        (uint32_t)source_y) * width + (uint32_t)source_x) * channels + channel;
    output[destination] = input[source];
}

int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, uint32_t batch, uint32_t depth,
        uint32_t height, uint32_t width, uint32_t channels,
        uint32_t depth_front, uint32_t height_before, uint32_t height_after,
        uint32_t width_before, uint32_t width_after) {
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width)
        return h3_set_error(gpu, "invalid VAE padding shape");
    size_t input_elements, output_elements;
    uint64_t output_depth = (uint64_t)depth + depth_front;
    uint64_t output_height = (uint64_t)height + height_before + height_after;
    uint64_t output_width = (uint64_t)width + width_before + width_after;
    if (output_depth > UINT32_MAX || output_height > UINT32_MAX ||
        output_width > UINT32_MAX ||
        !h3_mul_size((size_t)batch * depth * height * width, channels,
                     &input_elements) ||
        !h3_mul_size((size_t)batch * (size_t)output_depth *
                     (size_t)output_height * (size_t)output_width, channels,
                     &output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, input_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, output_elements))
        return h3_set_error(gpu, "invalid VAE padding tensors");
    dim3 grid((channels + 127) / 128, (uint32_t)output_width,
              (uint32_t)((uint64_t)batch * output_depth * output_height));
    h3_vae_pad_kernel<<<grid, 128, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data, batch, depth,
        height, width, channels, depth_front, height_before, height_after,
        width_before, width_after);
    return h3_launch_ok(gpu, "VAE encoder padding");
}

__global__ static void h3_conv3d_kernel(float *output, const float *input,
        const float *weight, const float *bias, uint32_t depth,
        uint32_t height, uint32_t width, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel_depth,
        uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth,
        uint32_t stride_height, uint32_t stride_width, uint32_t output_depth,
        uint32_t output_height, uint32_t output_width, size_t output_elements,
        int has_bias) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = (uint32_t)(index % output_channels);
    size_t row = index / output_channels;
    uint32_t spatial_count = output_depth * output_height * output_width;
    uint32_t spatial = (uint32_t)(row % spatial_count);
    uint32_t batch_index = (uint32_t)(row / spatial_count);
    uint32_t output_x = spatial % output_width;
    uint32_t output_y = (spatial / output_width) % output_height;
    uint32_t output_t = spatial / (output_width * output_height);
    if (output_t >= output_depth) return;
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t kt = 0; kt < kernel_depth; kt++)
            for (uint32_t ky = 0; ky < kernel_height; ky++)
                for (uint32_t kx = 0; kx < kernel_width; kx++) {
                    uint32_t it = output_t * stride_depth + kt;
                    uint32_t iy = output_y * stride_height + ky;
                    uint32_t ix = output_x * stride_width + kx;
                    size_t input_index = ((((size_t)batch_index * depth + it) *
                        height + iy) * width + ix) * input_channels + input_channel;
                    size_t weight_index = (((((size_t)output_channel *
                        input_channels + input_channel) * kernel_depth + kt) *
                        kernel_height + ky) * kernel_width + kx);
                    sum = fmaf(input[input_index], weight[weight_index], sum);
                }
    size_t destination = ((((size_t)batch_index * output_depth + output_t) *
        output_height + output_y) * output_width + output_x) * output_channels +
        output_channel;
    output[destination] = sum;
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth,
        uint32_t height, uint32_t width, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel_depth,
        uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth,
        uint32_t stride_height, uint32_t stride_width) {
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width || depth < kernel_depth ||
        height < kernel_height || width < kernel_width)
        return h3_set_error(gpu, "invalid Conv3d shape");
    uint32_t od = (depth - kernel_depth) / stride_depth + 1;
    uint32_t oh = (height - kernel_height) / stride_height + 1;
    uint32_t ow = (width - kernel_width) / stride_width + 1;
    size_t input_elements, weight_elements, output_elements;
    if (!h3_mul_size((size_t)batch * depth * height * width, input_channels,
                     &input_elements) ||
        !h3_mul_size((size_t)output_channels * input_channels * kernel_depth *
                     kernel_height, kernel_width, &weight_elements) ||
        !h3_mul_size((size_t)batch * od * oh * ow, output_channels,
                     &output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, input_elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, weight_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, output_elements) ||
        (bias && !h3_tensor_is(bias, gpu, H3_GPU_F32, output_channels)))
        return h3_set_error(gpu, "invalid Conv3d tensors");
    h3_conv3d_kernel<<<(output_elements + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : (const float *)input->data,
        depth, height, width, input_channels, output_channels, kernel_depth,
        kernel_height, kernel_width, stride_depth, stride_height, stride_width,
        od, oh, ow, output_elements, bias != NULL);
    int ok = h3_launch_ok(gpu, "Conv3d");
    if (ok) gpu->stats.mps_conv_dispatches++;
    return ok;
}

__global__ static void h3_group_norm_silu_kernel(float *output,
        const float *input, const float *weight, const float *bias,
        uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t groups, uint32_t rows, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t channels_per_group = channels / groups;
    uint32_t group = row % groups;
    uint32_t temporal = row / groups;
    uint32_t elements = height * width * channels_per_group;
    float mean = 0.0f;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        mean += input[((size_t)temporal * height * width + spatial) * channels +
                      channel];
    }
    mean /= (float)elements;
    float variance = 0.0f;
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        float centered = input[((size_t)temporal * height * width + spatial) *
                               channels + channel] - mean;
        variance = fmaf(centered, centered, variance);
    }
    float inverse = rsqrtf(variance / (float)elements + epsilon);
    for (uint32_t index = 0; index < elements; index++) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group * channels_per_group + index % channels_per_group;
        size_t destination = ((size_t)temporal * height * width + spatial) *
                             channels + channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + expf(-value));
    }
    (void)depth;
}

int h3_gpu_vae_encoder_group_norm_silu_f32(h3_gpu *gpu,
        h3_gpu_tensor *output, const h3_gpu_tensor *input,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
        uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t groups, float epsilon) {
    size_t elements;
    uint64_t rows64 = (uint64_t)batch * depth * groups;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) || rows64 > UINT32_MAX ||
        !h3_mul_size((size_t)batch * depth * height * width, channels,
                     &elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, channels) ||
        !h3_tensor_is(bias, gpu, H3_GPU_F32, channels))
        return h3_set_error(gpu, "invalid VAE group norm tensors or shape");
    uint32_t rows = (uint32_t)rows64;
    h3_group_norm_silu_kernel<<<(rows + 127) / 128, 128, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, (const float *)bias->data, depth, height,
        width, channels, groups, rows, epsilon);
    return h3_launch_ok(gpu, "VAE group norm SiLU");
}

static int h3_mlp_bf16_impl(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *activated, const h3_gpu_tensor *input,
        const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight,
        uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
        uint32_t output_dim) {
    size_t fused_elements, activated_elements, output_elements;
    if (!rows || !input_dim || !hidden_dim || !output_dim ||
        hidden_dim > UINT32_MAX / 2 ||
        !h3_mul_size(rows, (size_t)hidden_dim * 2, &fused_elements) ||
        !h3_mul_size(rows, hidden_dim, &activated_elements) ||
        !h3_mul_size(rows, output_dim, &output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16,
                      (size_t)rows * input_dim) ||
        !h3_tensor_is(fc1_weight, gpu, H3_GPU_BF16,
                      (size_t)hidden_dim * 2 * input_dim) ||
        !h3_tensor_is(fc2_weight, gpu, H3_GPU_BF16,
                      (size_t)output_dim * hidden_dim) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, output_elements) ||
        (activated && !h3_tensor_is(activated, gpu, H3_GPU_BF16,
                                    activated_elements)))
        return h3_set_error(gpu, "invalid BF16 MLP tensors or shape");
    __nv_bfloat16 *fused_data = NULL;
    __nv_bfloat16 *activated_data = activated ?
        (__nv_bfloat16 *)activated->data : NULL;
    cudaError_t status = cudaMallocAsync((void **)&fused_data,
        fused_elements * sizeof(*fused_data), gpu->stream);
    if (status == cudaSuccess && !activated_data)
        status = cudaMallocAsync((void **)&activated_data,
            activated_elements * sizeof(*activated_data), gpu->stream);
    if (!h3_cuda_ok(gpu, status, "BF16 MLP temporary allocation")) {
        if (fused_data) (void)cudaFreeAsync(fused_data, gpu->stream);
        return 0;
    }
    h3_gpu_tensor fused = {gpu, fused_data, fused_elements,
                           fused_elements * sizeof(*fused_data), H3_GPU_BF16};
    h3_gpu_tensor activation = {gpu, activated_data, activated_elements,
        activated_elements * sizeof(*activated_data), H3_GPU_BF16};
    int ok = h3_gpu_linear_bf16(gpu, &fused, input, fc1_weight, NULL, rows,
                                 input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, &activation, &fused, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, &activation, fc2_weight, NULL,
                                 rows, hidden_dim, output_dim);
    status = cudaFreeAsync(fused_data, gpu->stream);
    if (ok && !activated)
        status = cudaFreeAsync(activated_data, gpu->stream);
    if (ok) ok = h3_cuda_ok(gpu, status, "BF16 MLP temporary free");
    return ok;
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight,
        const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim,
        uint32_t hidden_dim, uint32_t output_dim) {
    return h3_mlp_bf16_impl(gpu, output, NULL, input, fc1_weight, fc2_weight,
                            rows, input_dim, hidden_dim, output_dim);
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        h3_gpu_tensor *activated, const h3_gpu_tensor *input,
        const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight,
        uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
        uint32_t output_dim) {
    return h3_mlp_bf16_impl(gpu, output, activated, input, fc1_weight,
                            fc2_weight, rows, input_dim, hidden_dim,
                            output_dim);
}

enum h3_unary_kind { H3_SILU, H3_GELU_EXACT, H3_GELU_APPROX, H3_CLIP };

__global__ static void h3_unary_f32_kernel(float *output, const float *input,
                                            size_t count, int kind,
                                            float minimum, float maximum) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = input[index];
    if (kind == H3_SILU) value /= 1.0f + expf(-value);
    else if (kind == H3_CLIP) value = fminf(maximum, fmaxf(minimum, value));
    output[index] = value;
}

__global__ static void h3_unary_bf16_kernel(__nv_bfloat16 *output,
                                             const __nv_bfloat16 *input,
                                             size_t count, int kind) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = __bfloat162float(input[index]);
    if (kind == H3_SILU) value /= 1.0f + expf(-value);
    else if (kind == H3_GELU_APPROX) {
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        value = inner <= -10.0f ? 0.0f : inner >= 10.0f ? value :
                0.5f * value * (1.0f + tanhf(inner));
    } else {
        value = value <= -10.0f ? 0.0f : value >= 10.0f ? value :
                0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    }
    output[index] = __float2bfloat16(value);
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements))
        return h3_set_error(gpu, "invalid F32 SiLU tensors");
    if (elements) h3_unary_f32_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data, (const float *)input->data,
                       elements, H3_SILU, 0.0f, 0.0f);
    return h3_launch_ok(gpu, "F32 SiLU");
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements))
        return h3_set_error(gpu, "invalid F32-to-BF16 tensors");
    if (elements) h3_f32_to_bf16_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((__nv_bfloat16 *)output->data,
                       (const float *)input->data, elements);
    return h3_launch_ok(gpu, "F32-to-BF16 cast");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements))
        return h3_set_error(gpu, "invalid BF16-to-F32 tensors");
    if (elements) h3_bf16_to_f32_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data,
                       (const __nv_bfloat16 *)input->data, elements);
    return h3_launch_ok(gpu, "BF16-to-F32 cast");
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements) || minimum > maximum)
        return h3_set_error(gpu, "invalid F32 clip arguments");
    if (elements) h3_unary_f32_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data, (const float *)input->data,
                       elements, H3_CLIP, minimum, maximum);
    return h3_launch_ok(gpu, "F32 clip");
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements))
        return h3_set_error(gpu, "invalid BF16 SiLU tensors");
    if (elements) h3_unary_bf16_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)input->data, elements, H3_SILU);
    return h3_launch_ok(gpu, "BF16 SiLU");
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements))
        return h3_set_error(gpu, "invalid BF16 GELU tensors");
    if (elements) h3_unary_bf16_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)input->data, elements,
        approximate ? H3_GELU_APPROX : H3_GELU_EXACT);
    return h3_launch_ok(gpu, "BF16 GELU");
}

enum h3_binary_kind { H3_ADD, H3_SUB, H3_SILU_MUL };

__global__ static void h3_binary_bf16_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *left, const __nv_bfloat16 *right,
        size_t count, int kind) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = __bfloat162float(left[index]);
    float b = __bfloat162float(right[index]);
    float value = kind == H3_ADD ? a + b : kind == H3_SUB ? a - b :
                  a / (1.0f + expf(-a)) * b;
    output[index] = __float2bfloat16(value);
}

static int h3_binary_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *left, const h3_gpu_tensor *right,
        uint32_t elements, int kind, const char *label) {
    if (!h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(left, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(right, gpu, H3_GPU_BF16, elements))
        return h3_set_error(gpu, "invalid %s tensors", label);
    if (elements) h3_binary_bf16_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)left->data,
        (const __nv_bfloat16 *)right->data, elements, kind);
    return h3_launch_ok(gpu, label);
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    return h3_binary_bf16(gpu, output, left, right, elements, H3_ADD, "BF16 add");
}
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    return h3_binary_bf16(gpu, output, left, right, elements, H3_SUB, "BF16 subtract");
}
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    return h3_binary_bf16(gpu, output, gate, up, elements, H3_SILU_MUL, "BF16 SiLU multiply");
}

__global__ static void h3_add_scaled_kernel(float *output, const float *left,
        const float *right, size_t count, float left_scale,
        float right_scale) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = left[index] * left_scale + right[index] * right_scale;
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(left, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(right, gpu, H3_GPU_F32, elements))
        return h3_set_error(gpu, "invalid F32 scaled-add tensors");
    if (elements) h3_add_scaled_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data, (const float *)left->data,
        (const float *)right->data, elements, left_scale, right_scale);
    return h3_launch_ok(gpu, "F32 scaled add");
}

__global__ static void h3_euler_kernel(float *sample, size_t sample_offset,
        const __nv_bfloat16 *last, const __nv_bfloat16 *previous,
        size_t count, float delta, float ratio) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float last_value = __bfloat162float(last[index]);
    float velocity = fmaf(ratio,
        last_value - __bfloat162float(previous[index]), last_value);
    sample[sample_offset + index] =
        fmaf(delta, velocity, sample[sample_offset + index]);
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!h3_tensor_is(sample, gpu, H3_GPU_F32, sample_offset + elements) ||
        !h3_tensor_is(last, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(previous, gpu, H3_GPU_BF16, elements))
        return h3_set_error(gpu, "invalid Euler tensors");
    if (elements) h3_euler_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)sample->data, sample_offset,
        (const __nv_bfloat16 *)last->data,
        (const __nv_bfloat16 *)previous->data, elements, delta, ratio);
    return h3_launch_ok(gpu, "BF16 Euler update");
}

__global__ static void h3_rms_norm_f32_kernel(float *output,
        const float *input, const float *weight, uint32_t rows,
        uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = input[(size_t)row * width + column];
        sum = fmaf(value, value, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        output[index] = input[index] * inverse * weight[column];
    }
}

__global__ static void h3_rms_norm_bf16_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
        uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float value = __bfloat162float(input[(size_t)row * width + column]);
        sum = fmaf(value, value, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        float value = __bfloat162float(input[index]) * inverse *
                      __bfloat162float(weight[column]);
        output[index] = __float2bfloat16(value);
    }
}

static int h3_matrix_elements(h3_gpu *gpu, uint32_t rows, uint32_t width,
                              size_t *elements) {
    if (!rows || !width || (size_t)rows > SIZE_MAX / width)
        return h3_set_error(gpu, "invalid matrix shape");
    *elements = (size_t)rows * width;
    return 1;
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) || epsilon < 0.0f ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, width))
        return h3_set_error(gpu, "invalid F32 RMS norm arguments");
    h3_rms_norm_f32_kernel<<<rows, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, rows, width, epsilon);
    return h3_launch_ok(gpu, "F32 RMS norm");
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) || epsilon < 0.0f ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_BF16, width))
        return h3_set_error(gpu, "invalid BF16 RMS norm arguments");
    h3_rms_norm_bf16_kernel<<<rows, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data, rows, width, epsilon);
    return h3_launch_ok(gpu, "BF16 RMS norm");
}

__global__ static void h3_layer_norm_f32_kernel(float *output,
        const float *input, const float *weight, const float *bias,
        uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        sum += input[(size_t)row * width + column];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = sums[0] / (float)width;
    sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered = input[(size_t)row * width + column] - mean;
        sum = fmaf(centered, centered, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        output[index] = (input[index] - mean) * inverse * weight[column] + bias[column];
    }
}

__global__ static void h3_layer_norm_bf16_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
        const __nv_bfloat16 *bias, uint32_t rows, uint32_t width,
        float epsilon) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x)
        sum += __bfloat162float(input[(size_t)row * width + column]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = sums[0] / (float)width;
    sum = 0.0f;
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        float centered = __bfloat162float(input[(size_t)row * width + column]) - mean;
        sum = fmaf(centered, centered, sum);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0] / (float)width + epsilon);
    for (uint32_t column = threadIdx.x; column < width; column += blockDim.x) {
        size_t index = (size_t)row * width + column;
        float value = (__bfloat162float(input[index]) - mean) * inverse *
                      __bfloat162float(weight[column]) +
                      __bfloat162float(bias[column]);
        output[index] = __float2bfloat16(value);
    }
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
        float epsilon) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) || epsilon < 0.0f ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_F32, width) ||
        !h3_tensor_is(bias, gpu, H3_GPU_F32, width))
        return h3_set_error(gpu, "invalid F32 layer norm arguments");
    h3_layer_norm_f32_kernel<<<rows, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)weight->data, (const float *)bias->data,
        rows, width, epsilon);
    return h3_launch_ok(gpu, "F32 layer norm");
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
        const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
        float epsilon) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) || epsilon < 0.0f ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_BF16, width) ||
        !h3_tensor_is(bias, gpu, H3_GPU_BF16, width))
        return h3_set_error(gpu, "invalid BF16 layer norm arguments");
    h3_layer_norm_bf16_kernel<<<rows, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)weight->data,
        (const __nv_bfloat16 *)bias->data, rows, width, epsilon);
    return h3_launch_ok(gpu, "BF16 layer norm");
}

template <typename T>
__device__ static float h3_value(T value);
template <>
__device__ float h3_value<float>(float value) { return value; }
template <>
__device__ float h3_value<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template <typename T>
__device__ static T h3_store(float value);
template <>
__device__ float h3_store<float>(float value) { return value; }
template <>
__device__ __nv_bfloat16 h3_store<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
__global__ static void h3_adaln_kernel(T *output, const T *input,
        const T *weight, const T *modulation, const uint32_t *row_map,
        uint32_t rows, uint32_t width, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon,
        size_t input_offset) {
    __shared__ float inverse;
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const T *source = input + input_offset + (size_t)row * width;
    if (threadIdx.x == 0) {
        float square_sum = 0.0f;
        for (uint32_t k = 0; k < width; ++k) {
            float value = h3_value(source[k]);
            square_sum = fmaf(value, value, square_sum);
        }
        inverse = rsqrtf(square_sum / (float)width + epsilon);
    }
    __syncthreads();
    size_t base = (size_t)row_map[row] * slots * width;
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float normalized = h3_value(source[column]) * inverse *
                           h3_value(weight[column]);
        float shift = h3_value(
            modulation[base + (size_t)shift_slot * width + column]);
        float scale = h3_value(
            modulation[base + (size_t)scale_slot * width + column]);
        output[(size_t)row * width + column] =
            h3_store<T>(normalized * (1.0f + scale) + shift);
    }
}

template <typename T>
__global__ static void h3_gate_kernel(T *output, const T *residual,
        const T *branch, const T *modulation, const uint32_t *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    size_t base = (size_t)row_map[row] * slots * width;
    float gate = h3_value(modulation[base + (size_t)gate_slot * width + column]);
    output[index] = h3_store<T>(h3_value(residual[index]) +
                                h3_value(branch[index]) * gate);
}

static int h3_adaln_validate(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, size_t input_offset,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *modulation,
        const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width,
        uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
        h3_gpu_dtype dtype, size_t *elements) {
    if (!h3_matrix_elements(gpu, rows, width, elements) || !slots ||
        shift_slot >= slots || scale_slot >= slots ||
        input_offset > SIZE_MAX - *elements ||
        !h3_tensor_is(output, gpu, dtype, *elements) ||
        !h3_tensor_is(input, gpu, dtype, input_offset + *elements) ||
        !h3_tensor_is(weight, gpu, dtype, width) ||
        !modulation || modulation->gpu != gpu || modulation->dtype != dtype ||
        !h3_tensor_is(row_map, gpu, H3_GPU_U32, rows))
        return h3_set_error(gpu, "invalid AdaLN arguments");
    return 1;
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    size_t elements = 0;
    if (epsilon < 0.0f || !h3_adaln_validate(gpu, output, input, 0,
        norm_weight, modulation, row_map, rows, width, slots, shift_slot,
        scale_slot, H3_GPU_F32, &elements)) return 0;
    h3_adaln_kernel<float><<<rows, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)norm_weight->data, (const float *)modulation->data,
        (const uint32_t *)row_map->data, rows, width, slots, shift_slot,
        scale_slot, epsilon, 0);
    return h3_launch_ok(gpu, "F32 AdaLN");
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, size_t input_offset,
        const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
        const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width,
        uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
        float epsilon) {
    size_t elements = 0;
    if (epsilon < 0.0f || !h3_adaln_validate(gpu, output, input, input_offset,
        norm_weight, modulation, row_map, rows, width, slots, shift_slot,
        scale_slot, H3_GPU_BF16, &elements)) return 0;
    h3_adaln_kernel<__nv_bfloat16><<<rows, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)input->data,
        (const __nv_bfloat16 *)norm_weight->data,
        (const __nv_bfloat16 *)modulation->data,
        (const uint32_t *)row_map->data, rows, width, slots, shift_slot,
        scale_slot, epsilon, input_offset);
    return h3_launch_ok(gpu, "BF16 AdaLN");
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(gpu, output, input, 0, norm_weight,
        modulation, row_map, rows, width, slots, shift_slot, scale_slot,
        epsilon);
}

static int h3_gate_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot,
        h3_gpu_dtype dtype) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) || !slots ||
        gate_slot >= slots || !h3_tensor_is(output, gpu, dtype, elements) ||
        !h3_tensor_is(residual, gpu, dtype, elements) ||
        !h3_tensor_is(branch, gpu, dtype, elements) ||
        !modulation || modulation->gpu != gpu || modulation->dtype != dtype ||
        !h3_tensor_is(row_map, gpu, H3_GPU_U32, rows))
        return h3_set_error(gpu, "invalid gate arguments");
    dim3 grid((width + 255) / 256, rows);
    if (dtype == H3_GPU_F32)
        h3_gate_kernel<float><<<grid, 256, 0, gpu->stream>>>(
            (float *)output->data, (const float *)residual->data,
            (const float *)branch->data, (const float *)modulation->data,
            (const uint32_t *)row_map->data, rows, width, slots, gate_slot);
    else
        h3_gate_kernel<__nv_bfloat16><<<grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)residual->data,
            (const __nv_bfloat16 *)branch->data,
            (const __nv_bfloat16 *)modulation->data,
            (const uint32_t *)row_map->data, rows, width, slots, gate_slot);
    return h3_launch_ok(gpu, dtype == H3_GPU_F32 ? "F32 gate" : "BF16 gate");
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot) {
    return h3_gate_dispatch(gpu, output, residual, branch, modulation, row_map,
                            rows, width, slots, gate_slot, H3_GPU_F32);
}
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot) {
    return h3_gate_dispatch(gpu, output, residual, branch, modulation, row_map,
                            rows, width, slots, gate_slot, H3_GPU_BF16);
}

__global__ static void h3_embedding_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *weight, const uint32_t *token_ids,
        uint32_t tokens, uint32_t vocab_size, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t token = blockIdx.y;
    if (token >= tokens || column >= width) return;
    uint32_t id = token_ids[token];
    output[(size_t)token * width + column] = id < vocab_size ?
        weight[(size_t)id * width + column] : __float2bfloat16(0.0f);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *weight, const h3_gpu_tensor *token_ids,
        uint32_t tokens, uint32_t vocab_size, uint32_t width) {
    size_t output_elements = 0;
    size_t weight_elements = 0;
    if (!h3_matrix_elements(gpu, tokens, width, &output_elements) ||
        !h3_matrix_elements(gpu, vocab_size, width, &weight_elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, output_elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_BF16, weight_elements) ||
        !h3_tensor_is(token_ids, gpu, H3_GPU_U32, tokens))
        return h3_set_error(gpu, "invalid embedding arguments");
    dim3 grid((width + 255) / 256, tokens);
    h3_embedding_kernel<<<grid, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)weight->data,
        (const uint32_t *)token_ids->data, tokens, vocab_size, width);
    return h3_launch_ok(gpu, "BF16 embedding");
}

template <typename T>
__global__ static void h3_swiglu_kernel(T *output, const T *fused,
                                        uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t base = (size_t)row * width * 2;
    float gate = h3_value(fused[base + column]);
    float up = h3_value(fused[base + width + column]);
    output[(size_t)row * width + column] =
        h3_store<T>(gate / (1.0f + expf(-gate)) * up);
}

static int h3_swiglu_dispatch(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *fused, uint32_t rows, uint32_t width,
        h3_gpu_dtype dtype) {
    size_t output_elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &output_elements) ||
        output_elements > SIZE_MAX / 2 ||
        !h3_tensor_is(output, gpu, dtype, output_elements) ||
        !h3_tensor_is(fused, gpu, dtype, output_elements * 2))
        return h3_set_error(gpu, "invalid SwiGLU arguments");
    dim3 grid((width + 255) / 256, rows);
    if (dtype == H3_GPU_F32)
        h3_swiglu_kernel<float><<<grid, 256, 0, gpu->stream>>>(
            (float *)output->data, (const float *)fused->data, rows, width);
    else
        h3_swiglu_kernel<__nv_bfloat16><<<grid, 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->data,
            (const __nv_bfloat16 *)fused->data, rows, width);
    return h3_launch_ok(gpu, dtype == H3_GPU_F32 ? "F32 SwiGLU" : "BF16 SwiGLU");
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    return h3_swiglu_dispatch(gpu, output, fused, rows, width, H3_GPU_F32);
}
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    return h3_swiglu_dispatch(gpu, output, fused, rows, width, H3_GPU_BF16);
}

__global__ static void h3_scale_add_kernel(float *output,
        const float *residual, const float *branch, const float *scale,
        uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    size_t index = (size_t)row * width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
        const h3_gpu_tensor *scale, uint32_t rows, uint32_t width) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(residual, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(branch, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(scale, gpu, H3_GPU_F32, width))
        return h3_set_error(gpu, "invalid scale-add arguments");
    dim3 grid((width + 255) / 256, rows);
    h3_scale_add_kernel<<<grid, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)residual->data,
        (const float *)branch->data, (const float *)scale->data, rows, width);
    return h3_launch_ok(gpu, "F32 scale add");
}

__global__ static void h3_geglu_kernel(float *output, const float *gate,
        const float *linear, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = gate[index];
    float gelu = 0.5f * value * (1.0f + tanhf(0.7978845608028654f *
        (value + 0.044715f * value * value * value)));
    output[index] = gelu * linear[index];
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *gate, const h3_gpu_tensor *linear,
        uint32_t elements) {
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(gate, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(linear, gpu, H3_GPU_F32, elements))
        return h3_set_error(gpu, "invalid GEGLU arguments");
    if (elements) h3_geglu_kernel<<<(elements + 255) / 256, 256, 0,
        gpu->stream>>>((float *)output->data, (const float *)gate->data,
                       (const float *)linear->data, elements);
    return h3_launch_ok(gpu, "F32 GEGLU");
}

__global__ static void h3_snake_kernel(float *output, const float *input,
        const float *alpha, size_t count, uint32_t channels) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float a = alpha[index % channels];
    float value = input[index];
    float wave = sinf(a * value);
    output[index] = value + wave * wave / (a + 1e-9f);
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, const h3_gpu_tensor *alpha,
        uint32_t batch, uint32_t length, uint32_t channels) {
    size_t count = (size_t)batch * length;
    if (!batch || !length || !channels || count > SIZE_MAX / channels)
        return h3_set_error(gpu, "invalid Snake shape");
    count *= channels;
    if (!h3_tensor_is(output, gpu, H3_GPU_F32, count) ||
        !h3_tensor_is(input, gpu, H3_GPU_F32, count) ||
        !h3_tensor_is(alpha, gpu, H3_GPU_F32, channels))
        return h3_set_error(gpu, "invalid Snake tensors");
    h3_snake_kernel<<<(count + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)input->data,
        (const float *)alpha->data, count, channels);
    return h3_launch_ok(gpu, "F32 Snake1d");
}

__global__ static void h3_weight_norm_kernel(float *output,
        const float *vector, const float *magnitude, uint32_t outer,
        uint32_t inner) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    size_t base = (size_t)row * inner;
    float square_sum = 0.0f;
    for (uint32_t column = 0; column < inner; column++)
        square_sum = fmaf(vector[base + column], vector[base + column], square_sum);
    float scale = magnitude[row] * rsqrtf(square_sum);
    for (uint32_t column = 0; column < inner; column++)
        output[base + column] = vector[base + column] * scale;
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *vector, const h3_gpu_tensor *magnitude,
        uint32_t outer, uint32_t inner) {
    size_t elements = 0;
    if (!h3_matrix_elements(gpu, outer, inner, &elements) ||
        !h3_tensor_is(output, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(vector, gpu, H3_GPU_F32, elements) ||
        !h3_tensor_is(magnitude, gpu, H3_GPU_F32, outer))
        return h3_set_error(gpu, "invalid weight norm arguments");
    h3_weight_norm_kernel<<<(outer + 255) / 256, 256, 0, gpu->stream>>>(
        (float *)output->data, (const float *)vector->data,
        (const float *)magnitude->data, outer, inner);
    return h3_launch_ok(gpu, "F32 weight norm");
}

__global__ static void h3_head_rms_kernel(__nv_bfloat16 *tensor,
        const __nv_bfloat16 *weight, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float epsilon) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence || head >= heads) return;
    size_t base = ((size_t)row * heads + head) * head_dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = __bfloat162float(tensor[base + d]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum / (float)head_dim + epsilon);
    for (uint32_t d = 0; d < head_dim; d++)
        tensor[base + d] = __float2bfloat16(
            __bfloat162float(tensor[base + d]) * inverse *
            __bfloat162float(weight[d]));
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
        const h3_gpu_tensor *weight, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float epsilon) {
    size_t elements = (size_t)sequence * heads;
    if (!sequence || !heads || !head_dim || elements > SIZE_MAX / head_dim)
        return h3_set_error(gpu, "invalid head RMS shape");
    elements *= head_dim;
    if (!h3_tensor_is(tensor, gpu, H3_GPU_BF16, elements) ||
        !h3_tensor_is(weight, gpu, H3_GPU_BF16, head_dim) || epsilon < 0.0f)
        return h3_set_error(gpu, "invalid head RMS arguments");
    dim3 grid((sequence + 127) / 128, heads);
    h3_head_rms_kernel<<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)tensor->data,
        (const __nv_bfloat16 *)weight->data, sequence, heads, head_dim, epsilon);
    return h3_launch_ok(gpu, "BF16 head RMS norm");
}

__global__ static void h3_rope_text_kernel(__nv_bfloat16 *query,
        __nv_bfloat16 *key, const float *rope_cos, const float *rope_sin,
        uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
        uint32_t head_dim) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    if (row >= sequence) return;
    uint32_t half = head_dim / 2;
    for (uint32_t d = 0; d < half; d++) {
        float c = rope_cos[(size_t)row * half + d];
        float s = rope_sin[(size_t)row * half + d];
        if (head < query_heads) {
            size_t base = ((size_t)row * query_heads + head) * head_dim;
            float first = __bfloat162float(query[base + d]);
            float second = __bfloat162float(query[base + half + d]);
            query[base + d] = __float2bfloat16(first * c - second * s);
            query[base + half + d] = __float2bfloat16(second * c + first * s);
        }
        if (head < kv_heads) {
            size_t base = ((size_t)row * kv_heads + head) * head_dim;
            float first = __bfloat162float(key[base + d]);
            float second = __bfloat162float(key[base + half + d]);
            key[base + d] = __float2bfloat16(first * c - second * s);
            key[base + half + d] = __float2bfloat16(second * c + first * s);
        }
    }
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, const h3_gpu_tensor *rope_cos_f32,
        const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim) {
    if (!sequence || !query_heads || !kv_heads || !head_dim || head_dim % 2)
        return h3_set_error(gpu, "invalid text RoPE shape");
    size_t query_elements = (size_t)sequence * query_heads * head_dim;
    size_t key_elements = (size_t)sequence * kv_heads * head_dim;
    size_t rope_elements = (size_t)sequence * (head_dim / 2);
    if (!h3_tensor_is(query, gpu, H3_GPU_BF16, query_elements) ||
        !h3_tensor_is(key, gpu, H3_GPU_BF16, key_elements) ||
        !h3_tensor_is(rope_cos_f32, gpu, H3_GPU_F32, rope_elements) ||
        !h3_tensor_is(rope_sin_f32, gpu, H3_GPU_F32, rope_elements))
        return h3_set_error(gpu, "invalid text RoPE tensors");
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    dim3 grid((sequence + 127) / 128, maximum_heads);
    h3_rope_text_kernel<<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (const float *)rope_cos_f32->data, (const float *)rope_sin_f32->data,
        sequence, query_heads, kv_heads, head_dim);
    return h3_launch_ok(gpu, "BF16 text RoPE");
}

template <typename T>
__global__ static void h3_qkv_rope_kernel(T *query, T *key, T *value,
        const T *qkv, const T *q_weight, const T *k_weight,
        const T *rope_cos, const T *rope_sin, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
        float epsilon, int grouped, int normalize, int weighted) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (row >= sequence || head >= heads || dimension >= head_dim) return;
    size_t inner = (size_t)heads * head_dim;
    size_t row_base = (size_t)row * inner * 3;
    size_t q_base = row_base + (size_t)head * head_dim;
    size_t k_base = q_base + inner;
    size_t v_base = q_base + inner * 2;
    if (grouped) {
        q_base = row_base + (size_t)head * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    }
    float q_inverse = 1.0f;
    float k_inverse = 1.0f;
    if (normalize) {
        float q_sum = 0.0f, k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float q = h3_value(qkv[q_base + d]);
            float k = h3_value(qkv[k_base + d]);
            q_sum = fmaf(q, q, q_sum);
            k_sum = fmaf(k, k, k_sum);
        }
        q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
        k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);
    }
    float qw = weighted ? h3_value(q_weight[dimension]) : 1.0f;
    float kw = weighted ? h3_value(k_weight[dimension]) : 1.0f;
    float q0 = h3_value(qkv[q_base + dimension]) * q_inverse * qw;
    float k0 = h3_value(qkv[k_base + dimension]) * k_inverse * kw;
    if (dimension < rope_half * 2) {
        uint32_t rope_index = dimension % rope_half;
        uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                 dimension - rope_half;
        float q1 = h3_value(qkv[q_base + pair]) * q_inverse *
                   (weighted ? h3_value(q_weight[pair]) : 1.0f);
        float k1 = h3_value(qkv[k_base + pair]) * k_inverse *
                   (weighted ? h3_value(k_weight[pair]) : 1.0f);
        float c = h3_value(rope_cos[(size_t)row * rope_half + rope_index]);
        float s = h3_value(rope_sin[(size_t)row * rope_half + rope_index]);
        if (dimension < rope_half) {
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else {
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
    }
    size_t output_index = ((size_t)head * sequence + row) * head_dim + dimension;
    query[output_index] = h3_store<T>(q0);
    key[output_index] = h3_store<T>(k0);
    value[output_index] = qkv[v_base + dimension];
}

static int h3_qkv_shape(h3_gpu *gpu, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t rope_half, size_t *elements) {
    if (!sequence || !heads || !head_dim || rope_half > head_dim / 2)
        return h3_set_error(gpu, "invalid QKV/RoPE shape");
    size_t count = (size_t)sequence * heads;
    if (count > SIZE_MAX / head_dim) return h3_set_error(gpu, "QKV shape overflow");
    *elements = count * head_dim;
    return 1;
}

static int h3_qkv_validate(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_weight, const h3_gpu_tensor *k_weight,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, h3_gpu_dtype dtype, int weighted,
        size_t *elements) {
    if (!h3_qkv_shape(gpu, sequence, heads, head_dim, rope_half, elements) ||
        *elements > SIZE_MAX / 3 ||
        !h3_tensor_is(query, gpu, dtype, *elements) ||
        !h3_tensor_is(key, gpu, dtype, *elements) ||
        !h3_tensor_is(value, gpu, dtype, *elements) ||
        !h3_tensor_is(qkv, gpu, dtype, *elements * 3) ||
        !h3_tensor_is(rope_cos, gpu, dtype, (size_t)sequence * rope_half) ||
        !h3_tensor_is(rope_sin, gpu, dtype, (size_t)sequence * rope_half) ||
        (weighted && (!h3_tensor_is(q_weight, gpu, dtype, head_dim) ||
                      !h3_tensor_is(k_weight, gpu, dtype, head_dim))))
        return h3_set_error(gpu, "invalid QKV/RoPE tensors");
    return 1;
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    size_t elements = 0;
    if (!h3_qkv_validate(gpu, query, key, value, qkv, q_norm, k_norm,
        rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        H3_GPU_F32, 1, &elements) || epsilon < 0.0f) return 0;
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3_qkv_rope_kernel<float><<<grid, 128, 0, gpu->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, (const float *)q_norm->data,
        (const float *)k_norm->data, (const float *)rope_cos->data,
        (const float *)rope_sin->data, sequence, heads, head_dim,
        rope_half, epsilon, 0, 1, 1);
    return h3_launch_ok(gpu, "F32 QKV RoPE");
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    size_t elements = 0;
    if (!h3_qkv_validate(gpu, query, key, value, qkv, NULL, NULL,
        rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        H3_GPU_F32, 0, &elements) || epsilon < 0.0f) return 0;
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3_qkv_rope_kernel<float><<<grid, 128, 0, gpu->stream>>>(
        (float *)query->data, (float *)key->data, (float *)value->data,
        (const float *)qkv->data, NULL, NULL, (const float *)rope_cos->data,
        (const float *)rope_sin->data, sequence, heads, head_dim,
        rope_half, epsilon, 1, 1, 0);
    return h3_launch_ok(gpu, "F32 video QKV RoPE");
}

static int h3_qkv_rope_bf16_dispatch(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon, int grouped) {
    size_t elements = 0;
    if (!h3_qkv_validate(gpu, query, key, value, qkv, q_norm, k_norm,
        rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        H3_GPU_BF16, 1, &elements) || epsilon < 0.0f) return 0;
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3_qkv_rope_kernel<__nv_bfloat16><<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        (const __nv_bfloat16 *)q_norm->data,
        (const __nv_bfloat16 *)k_norm->data,
        (const __nv_bfloat16 *)rope_cos->data,
        (const __nv_bfloat16 *)rope_sin->data, sequence, heads, head_dim,
        rope_half, epsilon, grouped, 1, 1);
    return h3_launch_ok(gpu, grouped ? "BF16 grouped QKV RoPE" : "BF16 QKV RoPE");
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    return h3_qkv_rope_bf16_dispatch(gpu, query, key, value, qkv, q_norm,
        k_norm, rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        epsilon, 0);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half, float epsilon) {
    return h3_qkv_rope_bf16_dispatch(gpu, query, key, value, qkv, q_norm,
        k_norm, rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        epsilon, 1);
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
        h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t rope_half) {
    size_t elements = 0;
    if (!h3_qkv_validate(gpu, query, key, value, qkv, NULL, NULL,
        rope_cos, rope_sin, sequence, heads, head_dim, rope_half,
        H3_GPU_BF16, 0, &elements)) return 0;
    dim3 grid((head_dim + 127) / 128, heads, sequence);
    h3_qkv_rope_kernel<__nv_bfloat16><<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query->data, (__nv_bfloat16 *)key->data,
        (__nv_bfloat16 *)value->data, (const __nv_bfloat16 *)qkv->data,
        NULL, NULL, (const __nv_bfloat16 *)rope_cos->data,
        (const __nv_bfloat16 *)rope_sin->data, sequence, heads, head_dim,
        rope_half, 0.0f, 0, 0, 0);
    return h3_launch_ok(gpu, "BF16 vision QKV RoPE");
}

__global__ static void h3_token_pool_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *input, size_t input_offset,
        __nv_bfloat16 *original, size_t original_offset,
        __nv_bfloat16 *baseline, size_t baseline_offset,
        const uint32_t *baseline_indices, const uint32_t *pairs,
        uint32_t rows, uint32_t width) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    uint32_t first_row = pairs[(size_t)row * 2];
    uint32_t second_row = pairs[(size_t)row * 2 + 1];
    __nv_bfloat16 first = input[input_offset + (size_t)first_row * width + column];
    original[original_offset + (size_t)first_row * width + column] = first;
    __nv_bfloat16 pooled = first;
    if (first_row != second_row) {
        __nv_bfloat16 second = input[input_offset + (size_t)second_row * width + column];
        original[original_offset + (size_t)second_row * width + column] = second;
        pooled = __float2bfloat16((__bfloat162float(first) +
                                   __bfloat162float(second)) * 0.5f);
    }
    output[(size_t)row * width + column] = pooled;
    uint32_t baseline_row = baseline_indices[row];
    if (baseline_row != UINT32_MAX)
        baseline[baseline_offset + (size_t)baseline_row * width + column] = pooled;
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *input, size_t input_offset,
        h3_gpu_tensor *original, size_t original_offset,
        h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
        uint32_t input_rows, uint32_t rows, uint32_t baseline_rows,
        uint32_t width) {
    size_t input_elements = 0, output_elements = 0, baseline_elements = 0;
    if (!h3_matrix_elements(gpu, input_rows, width, &input_elements) ||
        !h3_matrix_elements(gpu, rows, width, &output_elements) ||
        !h3_matrix_elements(gpu, baseline_rows, width, &baseline_elements) ||
        input_offset > SIZE_MAX - input_elements ||
        original_offset > SIZE_MAX - input_elements ||
        baseline_offset > SIZE_MAX - baseline_elements ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, output_elements) ||
        !h3_tensor_is(input, gpu, H3_GPU_BF16, input_offset + input_elements) ||
        !h3_tensor_is(original, gpu, H3_GPU_BF16, original_offset + input_elements) ||
        !h3_tensor_is(baseline, gpu, H3_GPU_BF16, baseline_offset + baseline_elements) ||
        !h3_tensor_is(baseline_indices, gpu, H3_GPU_U32, rows) ||
        rows > SIZE_MAX / 2 || !h3_tensor_is(pairs, gpu, H3_GPU_U32, (size_t)rows * 2))
        return h3_set_error(gpu, "invalid token pool arguments");
    dim3 grid((width + 255) / 256, rows);
    h3_token_pool_kernel<<<grid, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data, (const __nv_bfloat16 *)input->data,
        input_offset, (__nv_bfloat16 *)original->data, original_offset,
        (__nv_bfloat16 *)baseline->data, baseline_offset,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)pairs->data, rows, width);
    return h3_launch_ok(gpu, "BF16 token pool");
}

__global__ static void h3_token_expand_kernel(__nv_bfloat16 *output,
        const __nv_bfloat16 *original, size_t original_offset,
        const __nv_bfloat16 *reduced, const __nv_bfloat16 *baseline,
        size_t baseline_offset, const uint32_t *baseline_indices,
        const uint32_t *parents, uint32_t rows, uint32_t width,
        uint32_t exact_prefix_rows, float update_scale) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= width) return;
    uint32_t parent = parents[row];
    size_t destination = (size_t)row * width + column;
    size_t reduced_index = (size_t)parent * width + column;
    uint32_t baseline_row = baseline_indices[parent];
    if (row < exact_prefix_rows || baseline_row == UINT32_MAX) {
        output[destination] = reduced[reduced_index];
        return;
    }
    float update = __bfloat162float(reduced[reduced_index]) -
                   __bfloat162float(baseline[baseline_offset +
                                             (size_t)baseline_row * width + column]);
    output[destination] = __float2bfloat16(
        __bfloat162float(original[original_offset + destination]) +
        update_scale * update);
}

int h3_gpu_token_expand_delta_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
        const h3_gpu_tensor *original, size_t original_offset,
        const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
        size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
        const h3_gpu_tensor *parents, uint32_t rows, uint32_t reduced_rows,
        uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows,
        float update_scale) {
    size_t output_elements = 0, reduced_elements = 0, baseline_elements = 0;
    if (!h3_matrix_elements(gpu, rows, width, &output_elements) ||
        !h3_matrix_elements(gpu, reduced_rows, width, &reduced_elements) ||
        !h3_matrix_elements(gpu, baseline_rows, width, &baseline_elements) ||
        exact_prefix_rows > rows || original_offset > SIZE_MAX - output_elements ||
        baseline_offset > SIZE_MAX - baseline_elements ||
        !h3_tensor_is(output, gpu, H3_GPU_BF16, output_elements) ||
        !h3_tensor_is(original, gpu, H3_GPU_BF16, original_offset + output_elements) ||
        !h3_tensor_is(reduced, gpu, H3_GPU_BF16, reduced_elements) ||
        !h3_tensor_is(baseline, gpu, H3_GPU_BF16, baseline_offset + baseline_elements) ||
        !h3_tensor_is(baseline_indices, gpu, H3_GPU_U32, reduced_rows) ||
        !h3_tensor_is(parents, gpu, H3_GPU_U32, rows))
        return h3_set_error(gpu, "invalid token expand arguments");
    dim3 grid((width + 255) / 256, rows);
    h3_token_expand_kernel<<<grid, 256, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->data,
        (const __nv_bfloat16 *)original->data, original_offset,
        (const __nv_bfloat16 *)reduced->data,
        (const __nv_bfloat16 *)baseline->data, baseline_offset,
        (const uint32_t *)baseline_indices->data,
        (const uint32_t *)parents->data, rows, width, exact_prefix_rows,
        update_scale);
    return h3_launch_ok(gpu, "BF16 token expand");
}

int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
        h3_gpu_tensor *output, const h3_gpu_tensor *residual,
        const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight,
        const h3_gpu_tensor *gate_modulation,
        const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
        uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    if (!h3_gpu_gate_bf16(gpu, gated_residual, residual, branch,
                          gate_modulation, row_map, rows, width, slots,
                          gate_slot)) return 0;
    return h3_gpu_adaln_bf16(gpu, output, gated_residual, norm_weight,
                             norm_modulation, row_map, rows, width, slots,
                             shift_slot, scale_slot, epsilon);
}

int h3_gpu_token_pool_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual,
        h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset,
        h3_gpu_tensor *original, size_t original_offset,
        h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
        const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
        const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
        uint32_t baseline_rows, uint32_t width, uint32_t slots,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    if (!h3_gpu_token_pool_bf16(gpu, residual, input, input_offset, original,
        original_offset, baseline, baseline_offset, baseline_indices, pairs,
        input_rows, rows, baseline_rows, width)) return 0;
    return h3_gpu_adaln_bf16(gpu, output, residual, norm_weight, modulation,
        row_map, rows, width, slots, shift_slot, scale_slot, epsilon);
}

int h3_gpu_token_expand_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual,
        h3_gpu_tensor *output, const h3_gpu_tensor *original,
        size_t original_offset, const h3_gpu_tensor *reduced,
        const h3_gpu_tensor *baseline, size_t baseline_offset,
        const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
        const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
        const h3_gpu_tensor *row_map, uint32_t rows, uint32_t reduced_rows,
        uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows,
        float update_scale, uint32_t slots, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    if (!h3_gpu_token_expand_delta_bf16(gpu, residual, original,
        original_offset, reduced, baseline, baseline_offset, baseline_indices,
        parents, rows, reduced_rows, baseline_rows, width, exact_prefix_rows,
        update_scale)) return 0;
    return h3_gpu_adaln_bf16(gpu, output, residual, norm_weight, modulation,
        row_map, rows, width, slots, shift_slot, scale_slot, epsilon);
}

__global__ static void h3_text_qk_rope_kernel(__nv_bfloat16 *query_output,
        __nv_bfloat16 *key_output, const __nv_bfloat16 *query_input,
        const __nv_bfloat16 *key_input, const __nv_bfloat16 *q_weight,
        const __nv_bfloat16 *k_weight, const __nv_bfloat16 *rope_cos,
        const __nv_bfloat16 *rope_sin, uint32_t sequence,
        uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
        float epsilon) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (dimension >= head_dim || head >= query_heads || row >= sequence) return;
    uint32_t half = head_dim / 2;
    uint32_t pair = dimension < half ? dimension + half : dimension - half;
    uint32_t rope_index = dimension % half;
    float c = __bfloat162float(rope_cos[(size_t)row * half + rope_index]);
    float s = __bfloat162float(rope_sin[(size_t)row * half + rope_index]);
    size_t q_base = ((size_t)row * query_heads + head) * head_dim;
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) {
        float value = __bfloat162float(query_input[q_base + d]);
        q_sum = fmaf(value, value, q_sum);
    }
    float q_inverse = rsqrtf(q_sum / (float)head_dim + epsilon);
    float q0 = __bfloat162float(query_input[q_base + dimension]) * q_inverse *
               __bfloat162float(q_weight[dimension]);
    float q1 = __bfloat162float(query_input[q_base + pair]) * q_inverse *
               __bfloat162float(q_weight[pair]);
    query_output[q_base + dimension] = __float2bfloat16(
        dimension < half ? q0 * c - q1 * s : q0 * c + q1 * s);
    if (head < kv_heads) {
        size_t k_base = ((size_t)row * kv_heads + head) * head_dim;
        float k_sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float value = __bfloat162float(key_input[k_base + d]);
            k_sum = fmaf(value, value, k_sum);
        }
        float k_inverse = rsqrtf(k_sum / (float)head_dim + epsilon);
        float k0 = __bfloat162float(key_input[k_base + dimension]) * k_inverse *
                   __bfloat162float(k_weight[dimension]);
        float k1 = __bfloat162float(key_input[k_base + pair]) * k_inverse *
                   __bfloat162float(k_weight[pair]);
        key_output[k_base + dimension] = __float2bfloat16(
            dimension < half ? k0 * c - k1 * s : k0 * c + k1 * s);
    }
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
        h3_gpu_tensor *query_output, h3_gpu_tensor *key_output,
        const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input,
        const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
        const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
        uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
        uint32_t head_dim, float epsilon) {
    if (!sequence || !query_heads || !kv_heads || !head_dim || head_dim % 2 ||
        query_heads < kv_heads || epsilon < 0.0f)
        return h3_set_error(gpu, "invalid text QK/RoPE shape");
    size_t query_elements = (size_t)sequence * query_heads * head_dim;
    size_t key_elements = (size_t)sequence * kv_heads * head_dim;
    size_t rope_elements = (size_t)sequence * (head_dim / 2);
    if (!h3_tensor_is(query_output, gpu, H3_GPU_BF16, query_elements) ||
        !h3_tensor_is(key_output, gpu, H3_GPU_BF16, key_elements) ||
        !h3_tensor_is(query_input, gpu, H3_GPU_BF16, query_elements) ||
        !h3_tensor_is(key_input, gpu, H3_GPU_BF16, key_elements) ||
        !h3_tensor_is(q_norm, gpu, H3_GPU_BF16, head_dim) ||
        !h3_tensor_is(k_norm, gpu, H3_GPU_BF16, head_dim) ||
        !h3_tensor_is(rope_cos, gpu, H3_GPU_BF16, rope_elements) ||
        !h3_tensor_is(rope_sin, gpu, H3_GPU_BF16, rope_elements))
        return h3_set_error(gpu, "invalid text QK/RoPE tensors");
    dim3 grid((head_dim + 127) / 128, query_heads, sequence);
    h3_text_qk_rope_kernel<<<grid, 128, 0, gpu->stream>>>(
        (__nv_bfloat16 *)query_output->data,
        (__nv_bfloat16 *)key_output->data,
        (const __nv_bfloat16 *)query_input->data,
        (const __nv_bfloat16 *)key_input->data,
        (const __nv_bfloat16 *)q_norm->data,
        (const __nv_bfloat16 *)k_norm->data,
        (const __nv_bfloat16 *)rope_cos->data,
        (const __nv_bfloat16 *)rope_sin->data, sequence, query_heads,
        kv_heads, head_dim, epsilon);
    return h3_launch_ok(gpu, "BF16 text QK RoPE");
}
