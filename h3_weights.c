#include "h3_weights.h"
#include "h3_lora.h"

#include <Accelerate/Accelerate.h>

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

size_t h3_weight_store_tensor_count(const h3_weight_store *store) {
    if (!store) return 0;
    size_t total = 0;
    for (size_t shard = 0; shard < store->count; shard++) {
        if (store->headers[shard].tensor_count > SIZE_MAX - total)
            return 0;
        total += store->headers[shard].tensor_count;
    }
    return total;
}

const h3_st_tensor *h3_weight_store_tensor_at(
        const h3_weight_store *store, size_t index) {
    if (!store) return NULL;
    for (size_t shard = 0; shard < store->count; shard++) {
        size_t count = store->headers[shard].tensor_count;
        if (index < count) return &store->headers[shard].tensors[index];
        index -= count;
    }
    return NULL;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        dtype == H3_DTYPE_I8 ?
        h3_gpu_tensor_load_i8(gpu, header->path, tensor->file_offset,
                              (size_t)elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

static int pread_all(int descriptor, void *buffer, size_t bytes,
                     uint64_t offset) {
    unsigned char *destination = buffer;
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t count = pread(descriptor, destination + completed,
                              bytes - completed, (off_t)(offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        completed += (size_t)count;
    }
    return 1;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

h3_gpu_tensor *h3_weight_load_bf16_with_loras(
        const h3_weight_store *store, h3_gpu *gpu, const char *name,
        uint64_t rows, uint64_t columns, const h3_lora_set *loras,
        size_t *peak_scratch, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    if (!loras || !h3_lora_set_has_target(loras, name))
        return h3_weight_load_bf16(store, gpu, name, 2, shape,
                                   error, error_size);
    if (!store || !gpu || !name || !rows || !columns || rows > INT_MAX ||
        columns > INT_MAX || rows > SIZE_MAX / columns) {
        fail(error, error_size, "invalid LoRA weight overlay request");
        return NULL;
    }
    const h3_st_header *header = NULL;
    const h3_st_tensor *base = h3_weight_find(store, name, &header);
    if (!base || !header || base->dtype != H3_DTYPE_BF16 || base->ndim != 2 ||
        base->shape[0] != rows || base->shape[1] != columns) {
        fail(error, error_size, "LoRA base weight has the wrong schema: %s", name);
        return NULL;
    }
    typedef struct {
        const h3_lora *lora;
        size_t pair;
        uint64_t rank;
        float effective_scale;
        float *a;
    } application;
    application applications[8];
    size_t application_count = 0;
    size_t a_elements = 0;
    uint64_t maximum_rank = 0;
    for (size_t index = 0; index < h3_lora_set_count(loras); index++) {
        const h3_lora *lora = NULL;
        float scale = 0.0f;
        size_t pair_index;
        if (!h3_lora_set_entry(loras, index, &lora, &scale, NULL) ||
            !h3_lora_find_pair(lora, name, &pair_index)) continue;
        const h3_lora_pair_info *info = h3_lora_pair_at(lora, pair_index);
        float effective = scale * info->alpha_over_rank;
        if (!info || info->output_dim != rows || info->input_dim != columns ||
            !info->rank || !isfinite(effective) ||
            info->rank > SIZE_MAX / columns ||
            a_elements > SIZE_MAX - (size_t)info->rank * (size_t)columns) {
            fail(error, error_size, "invalid LoRA overlay for %s", name);
            goto failed;
        }
        applications[application_count++] =
            (application){lora, pair_index, info->rank, effective, NULL};
        a_elements += (size_t)info->rank * (size_t)columns;
        if (info->rank > maximum_rank) maximum_rank = info->rank;
    }
    enum { TILE_ROWS = 64 };
    size_t tile_elements = (size_t)TILE_ROWS * (size_t)columns;
    size_t b_elements = (size_t)TILE_ROWS * (size_t)maximum_rank;
    if (!application_count || tile_elements > SIZE_MAX / sizeof(float) ||
        a_elements > SIZE_MAX / sizeof(float) ||
        b_elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "LoRA overlay scratch size overflows for %s", name);
        goto failed;
    }
    size_t scratch = tile_elements * (sizeof(float) + sizeof(uint16_t)) +
                     a_elements * sizeof(float) + b_elements * sizeof(float);
    if (scratch > 256u * 1024u * 1024u) {
        fail(error, error_size, "LoRA overlay exceeds 256 MiB scratch for %s", name);
        goto failed;
    }
    if (peak_scratch && scratch > *peak_scratch) *peak_scratch = scratch;
    uint16_t *tile_bf16 = malloc(tile_elements * sizeof(*tile_bf16));
    float *tile_f32 = malloc(tile_elements * sizeof(*tile_f32));
    float *b = malloc((b_elements ? b_elements : 1) * sizeof(*b));
    if (!tile_bf16 || !tile_f32 || !b) {
        free(tile_bf16); free(tile_f32); free(b);
        fail(error, error_size, "out of memory allocating LoRA overlay scratch");
        goto failed;
    }
    for (size_t index = 0; index < application_count; index++) {
        size_t elements = (size_t)applications[index].rank * (size_t)columns;
        applications[index].a = malloc(elements * sizeof(float));
        if (!applications[index].a || !h3_lora_read_a(
                applications[index].lora, applications[index].pair,
                applications[index].a, elements, error, error_size)) {
            free(tile_bf16); free(tile_f32); free(b);
            goto failed;
        }
    }
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * (size_t)columns);
    int descriptor = open(header->path, O_RDONLY | O_CLOEXEC);
    if (!output || descriptor < 0) {
        if (descriptor >= 0) close(descriptor);
        h3_gpu_tensor_free(output);
        free(tile_bf16); free(tile_f32); free(b);
        fail(error, error_size, "cannot open LoRA base weight %s", name);
        goto failed;
    }
    int ok = 1;
    for (uint64_t first_row = 0; first_row < rows && ok;
         first_row += TILE_ROWS) {
        uint64_t tile_rows = rows - first_row;
        if (tile_rows > TILE_ROWS) tile_rows = TILE_ROWS;
        size_t elements = (size_t)tile_rows * (size_t)columns;
        ok = pread_all(descriptor, tile_bf16, elements * sizeof(*tile_bf16),
                       base->file_offset + first_row * columns * sizeof(uint16_t));
        for (size_t adapter = 0; adapter < application_count && ok; adapter++) {
            for (size_t element = 0; element < elements; element++)
                tile_f32[element] = bf16_to_f32(tile_bf16[element]);
            application *item = &applications[adapter];
            size_t b_count = (size_t)tile_rows * (size_t)item->rank;
            ok = h3_lora_read_b_rows(
                item->lora, item->pair, first_row, tile_rows, b, b_count,
                error, error_size);
            if (!ok) break;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)tile_rows, (int)columns, (int)item->rank,
                        item->effective_scale, b, (int)item->rank,
                        item->a, (int)columns, 1.0f, tile_f32, (int)columns);
            for (size_t element = 0; element < elements; element++) {
                if (!isfinite(tile_f32[element])) {
                    fail(error, error_size,
                         "LoRA overlay produced NaN or Inf for %s", name);
                    ok = 0;
                    break;
                }
                tile_bf16[element] = f32_to_bf16(tile_f32[element]);
            }
        }
        if (ok) ok = h3_gpu_tensor_write_bf16_range(
            output, (size_t)first_row * (size_t)columns,
            tile_bf16, elements);
    }
    close(descriptor);
    free(tile_bf16); free(tile_f32); free(b);
    for (size_t index = 0; index < application_count; index++) {
        free(applications[index].a);
        applications[index].a = NULL;
    }
    if (!ok) {
        h3_gpu_tensor_free(output);
        if (!error || !*error)
            fail(error, error_size, "cannot apply LoRA overlay to %s", name);
        return NULL;
    }
    return output;

failed:
    for (size_t index = 0; index < application_count; index++)
        free(applications[index].a);
    return NULL;
}
