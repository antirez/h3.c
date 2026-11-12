#include "h3_weights.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (!name) {
        fail(error, error_size, "weight name is missing for this model family");
        return NULL;
    }
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

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

static int find_bf16_matrix(const h3_weight_store *store, const char *name,
                            uint64_t rows, uint64_t columns,
                            const h3_st_header **header,
                            const h3_st_tensor **tensor,
                            char *error, size_t error_size) {
    *tensor = name ? h3_weight_find(store, name, header) : NULL;
    if (!*tensor) {
        if (!name) {
            fail(error, error_size,
                 "weight name is missing for this model family");
            return 0;
        }
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if ((*tensor)->dtype != H3_DTYPE_BF16 || (*tensor)->ndim != 2 ||
        (*tensor)->shape[0] != rows || (*tensor)->shape[1] != columns ||
        columns == 0 || rows > SIZE_MAX / columns) {
        fail(error, error_size, "weight %s has the wrong BF16 matrix shape",
             name);
        return 0;
    }
    return 1;
}

int h3_weight_qkv_slab(size_t slab, unsigned heads, unsigned head_dim,
                       uint64_t columns, unsigned *projection,
                       uint64_t *source_offset, uint64_t *destination_offset) {
    if (!heads || !head_dim || !columns || !projection || !source_offset ||
        !destination_offset || slab >= (size_t)3 * heads) return 0;
    uint64_t slab_elements = (uint64_t)head_dim * columns;
    *projection = (unsigned)(slab % 3);
    *source_offset = (slab / 3) * slab_elements;
    *destination_offset = slab * slab_elements;
    return 1;
}

h3_gpu_tensor *h3_weight_load_bf16_qkv(const h3_weight_store *store,
                                       h3_gpu *gpu, const char *q_name,
                                       const char *k_name, const char *v_name,
                                       unsigned heads, unsigned head_dim,
                                       uint64_t columns,
                                       char *error, size_t error_size) {
    const char *names[3] = {q_name, k_name, v_name};
    const h3_st_header *headers[3];
    const h3_st_tensor *tensors[3];
    uint64_t rows = (uint64_t)heads * head_dim;
    for (unsigned index = 0; index < 3; index++) {
        if (!find_bf16_matrix(store, names[index], rows, columns,
                              &headers[index], &tensors[index],
                              error, error_size)) return NULL;
    }
    size_t elements = (size_t)(3 * rows * columns);
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!result) {
        fail(error, error_size, "cannot allocate grouped QKV for %s: %s",
             q_name, h3_gpu_error(gpu));
        return NULL;
    }
    size_t slab_elements = (size_t)head_dim * columns;
    for (size_t slab = 0; slab < (size_t)3 * heads; slab++) {
        unsigned projection;
        uint64_t source, destination;
        if (!h3_weight_qkv_slab(slab, heads, head_dim, columns, &projection,
                                &source, &destination) ||
            !h3_gpu_tensor_read_file_bf16_range(
                result, (size_t)destination, headers[projection]->path,
                tensors[projection]->file_offset +
                    source * sizeof(uint16_t),
                slab_elements, error, error_size)) {
            if (error && error_size && !error[0])
                fail(error, error_size, "invalid grouped QKV slab %zu", slab);
            h3_gpu_tensor_free(result);
            return NULL;
        }
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16_halves(const h3_weight_store *store,
                                          h3_gpu *gpu, const char *name,
                                          uint64_t rows, uint64_t columns,
                                          int swap_halves,
                                          char *error, size_t error_size) {
    if (!name || rows % 2) {
        fail(error, error_size, "weight %s cannot be split into halves",
             name ? name : "(unnamed)");
        return NULL;
    }
    const h3_st_header *header;
    const h3_st_tensor *tensor;
    if (!find_bf16_matrix(store, name, rows, columns, &header, &tensor,
                          error, error_size)) return NULL;
    size_t half = (size_t)(rows / 2 * columns);
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, 2 * half);
    if (!result) {
        fail(error, error_size, "cannot allocate %s: %s", name,
             h3_gpu_error(gpu));
        return NULL;
    }
    for (unsigned index = 0; index < 2; index++) {
        size_t destination = (swap_halves ? 1 - index : index) * half;
        if (!h3_gpu_tensor_read_file_bf16_range(
                result, destination, header->path,
                tensor->file_offset + (uint64_t)index * half *
                    sizeof(uint16_t),
                half, error, error_size)) {
            if (error && error_size && !error[0])
                fail(error, error_size, "cannot read %s", name);
            h3_gpu_tensor_free(result);
            return NULL;
        }
    }
    return result;
}
