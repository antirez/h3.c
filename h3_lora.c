#include "h3_lora.h"

#include <CommonCrypto/CommonDigest.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    H3_LORA_NATIVE = 1,
    H3_LORA_DIFFUSERS = 2,
    H3_LORA_KOHYA = 4,
    H3_LORA_SCAN_ELEMENTS = 1 << 18
};

typedef struct {
    h3_lora_pair_info info;
    const h3_st_tensor *a;
    const h3_st_tensor *b;
} h3_lora_pair;

struct h3_lora {
    int descriptor;
    char *path;
    h3_st_header header;
    h3_lora_pair *pairs;
    size_t pair_count;
    size_t pair_capacity;
    unsigned formats;
    char sha256[65];
    struct stat identity;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int ends_with(const char *text, const char *suffix) {
    size_t text_length = strlen(text), suffix_length = strlen(suffix);
    return text_length >= suffix_length &&
           !strcmp(text + text_length - suffix_length, suffix);
}

static char *join_suffix(const char *prefix, size_t prefix_length,
                         const char *suffix) {
    size_t suffix_length = strlen(suffix);
    if (prefix_length > SIZE_MAX - suffix_length - 1) return NULL;
    char *result = malloc(prefix_length + suffix_length + 1);
    if (!result) return NULL;
    memcpy(result, prefix, prefix_length);
    memcpy(result + prefix_length, suffix, suffix_length + 1);
    return result;
}

static int same_identity(const struct stat *left, const struct stat *right) {
    return left->st_dev == right->st_dev && left->st_ino == right->st_ino &&
           left->st_size == right->st_size &&
           left->st_mtimespec.tv_sec == right->st_mtimespec.tv_sec &&
           left->st_mtimespec.tv_nsec == right->st_mtimespec.tv_nsec;
}

static int pread_all(int descriptor, void *buffer, size_t bytes,
                     uint64_t offset) {
    unsigned char *destination = buffer;
    size_t completed = 0;
    while (completed < bytes) {
        size_t wanted = bytes - completed;
        if (wanted > (size_t)SSIZE_MAX) wanted = (size_t)SSIZE_MAX;
        ssize_t count = pread(descriptor, destination + completed, wanted,
                              (off_t)(offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        completed += (size_t)count;
    }
    return 1;
}

static float f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (!mantissa) bits = sign;
        else {
            int shift = 0;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= 0x3ffu;
            bits = sign | (uint32_t)(127 - 15 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int decode_values(const void *raw, h3_dtype dtype, size_t count,
                         float *values) {
    if (dtype == H3_DTYPE_F32) {
        memcpy(values, raw, count * sizeof(*values));
    } else if (dtype == H3_DTYPE_BF16) {
        const uint16_t *source = raw;
        for (size_t index = 0; index < count; index++) {
            uint32_t bits = (uint32_t)source[index] << 16;
            memcpy(&values[index], &bits, sizeof(bits));
        }
    } else if (dtype == H3_DTYPE_F16) {
        const uint16_t *source = raw;
        for (size_t index = 0; index < count; index++)
            values[index] = f16_to_f32(source[index]);
    } else return 0;
    return 1;
}

static int read_values(const h3_lora *lora, const h3_st_tensor *tensor,
                       uint64_t element_offset, size_t elements, float *values,
                       char *error, size_t error_size) {
    size_t item_size = h3_dtype_size(tensor->dtype);
    uint64_t total = h3_st_tensor_elements(tensor);
    if (!item_size || element_offset > total || elements > total - element_offset ||
        elements > SIZE_MAX / item_size) {
        fail(error, error_size, "invalid LoRA tensor read for %s", tensor->name);
        return 0;
    }
    size_t bytes = elements * item_size;
    void *raw = malloc(bytes ? bytes : 1);
    if (!raw || !pread_all(lora->descriptor, raw, bytes,
                           tensor->file_offset + element_offset * item_size) ||
        !decode_values(raw, tensor->dtype, elements, values)) {
        free(raw);
        fail(error, error_size, "cannot decode LoRA tensor %s", tensor->name);
        return 0;
    }
    free(raw);
    for (size_t index = 0; index < elements; index++)
        if (!isfinite(values[index])) {
            fail(error, error_size, "LoRA tensor %s contains NaN or Inf",
                 tensor->name);
            return 0;
        }
    return 1;
}

static int tensor_is_float(const h3_st_tensor *tensor) {
    return tensor && (tensor->dtype == H3_DTYPE_F16 ||
                      tensor->dtype == H3_DTYPE_BF16 ||
                      tensor->dtype == H3_DTYPE_F32);
}

static int validate_finite(const h3_lora *lora, const h3_st_tensor *tensor,
                           char *error, size_t error_size) {
    uint64_t total = h3_st_tensor_elements(tensor);
    float *values = malloc(H3_LORA_SCAN_ELEMENTS * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory validating LoRA tensor");
        return 0;
    }
    for (uint64_t offset = 0; offset < total;) {
        size_t count = total - offset > H3_LORA_SCAN_ELEMENTS ?
            H3_LORA_SCAN_ELEMENTS : (size_t)(total - offset);
        if (!read_values(lora, tensor, offset, count, values,
                         error, error_size)) {
            free(values);
            return 0;
        }
        offset += count;
    }
    free(values);
    return 1;
}

static int scalar_value(const h3_lora *lora, const h3_st_tensor *tensor,
                        float *value, char *error, size_t error_size) {
    if (!tensor_is_float(tensor) || h3_st_tensor_elements(tensor) != 1)
        return 0;
    return read_values(lora, tensor, 0, 1, value, error, error_size);
}

static char *kohya_target(const h3_weight_store *base, const char *encoded,
                          char *error, size_t error_size) {
    char *match = NULL;
    size_t count = h3_weight_store_tensor_count(base);
    for (size_t index = 0; index < count; index++) {
        const h3_st_tensor *tensor = h3_weight_store_tensor_at(base, index);
        if (!tensor || !ends_with(tensor->name, ".weight")) continue;
        size_t length = strlen(tensor->name) - strlen(".weight");
        if (length != strlen(encoded)) continue;
        int equal = 1;
        for (size_t at = 0; at < length; at++) {
            char expected = tensor->name[at] == '.' ? '_' : tensor->name[at];
            if (expected != encoded[at]) { equal = 0; break; }
        }
        if (!equal) continue;
        if (match) {
            free(match);
            fail(error, error_size, "Kohya key %s maps to multiple weights",
                 encoded);
            return NULL;
        }
        match = strdup(tensor->name);
    }
    if (!match)
        fail(error, error_size, "Kohya key %s has no checkpoint target", encoded);
    return match;
}

static int pair_uses(const h3_lora *lora, const h3_st_tensor *tensor) {
    for (size_t index = 0; index < lora->pair_count; index++)
        if (lora->pairs[index].a == tensor || lora->pairs[index].b == tensor)
            return 1;
    return 0;
}

static int add_pair(h3_lora *lora, const h3_weight_store *base,
                    char *target, const char *a_name, const char *b_name,
                    const char *alpha_name, unsigned format,
                    char *error, size_t error_size) {
    const h3_st_tensor *a = h3_st_find(&lora->header, a_name);
    const h3_st_tensor *b = h3_st_find(&lora->header, b_name);
    const h3_st_tensor *alpha = alpha_name ?
        h3_st_find(&lora->header, alpha_name) : NULL;
    const h3_st_tensor *base_tensor = h3_weight_find(base, target, NULL);
    if (!a || !b) {
        fail(error, error_size, "%s is missing its LoRA pair", a_name);
        return 0;
    }
    if (!base_tensor) {
        fail(error, error_size, "LoRA target is absent from checkpoint: %s",
             target);
        return 0;
    }
    for (size_t index = 0; index < lora->pair_count; index++)
        if (!strcmp(lora->pairs[index].info.target, target)) {
            fail(error, error_size, "multiple LoRA pairs map to %s", target);
            return 0;
        }
    if (!tensor_is_float(a) || !tensor_is_float(b) ||
        a->ndim != 2 || b->ndim != 2 || base_tensor->dtype != H3_DTYPE_BF16 ||
        base_tensor->ndim != 2 || !a->shape[0] || b->shape[1] != a->shape[0] ||
        b->shape[0] != base_tensor->shape[0] ||
        a->shape[1] != base_tensor->shape[1]) {
        fail(error, error_size, "LoRA shapes are incompatible for %s", target);
        return 0;
    }
    float alpha_over_rank = 1.0f;
    if (alpha) {
        float alpha_value;
        if (!scalar_value(lora, alpha, &alpha_value, error, error_size)) {
            if (!error || !*error)
                fail(error, error_size, "LoRA alpha is invalid for %s", target);
            return 0;
        }
        alpha_over_rank = alpha_value / (float)a->shape[0];
        if (!isfinite(alpha_over_rank)) {
            fail(error, error_size, "LoRA alpha/rank is not finite for %s",
                 target);
            return 0;
        }
    }
    if (!validate_finite(lora, a, error, error_size) ||
        !validate_finite(lora, b, error, error_size)) return 0;
    if (lora->pair_count == lora->pair_capacity) {
        size_t capacity = lora->pair_capacity ? lora->pair_capacity * 2 : 32;
        h3_lora_pair *grown = realloc(lora->pairs, capacity * sizeof(*grown));
        if (!grown) {
            fail(error, error_size, "out of memory indexing LoRA pairs");
            return 0;
        }
        lora->pairs = grown;
        lora->pair_capacity = capacity;
    }
    h3_lora_pair *pair = &lora->pairs[lora->pair_count++];
    memset(pair, 0, sizeof(*pair));
    pair->info.target = target;
    pair->info.output_dim = base_tensor->shape[0];
    pair->info.input_dim = base_tensor->shape[1];
    pair->info.rank = a->shape[0];
    pair->info.alpha_over_rank = alpha_over_rank;
    pair->a = a;
    pair->b = b;
    lora->formats |= format;
    return 1;
}

static int inspect_pairs(h3_lora *lora, const h3_weight_store *base,
                         char *error, size_t error_size) {
    static const char native_a[] = ".lora_A.weight";
    static const char native_b[] = ".lora_B.weight";
    static const char kohya_prefix[] = "lora_unet_";
    static const char kohya_a[] = ".lora_down.weight";
    static const char kohya_b[] = ".lora_up.weight";
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const char *name = lora->header.tensors[index].name;
        char *target = NULL, *b_name = NULL, *alpha_name = NULL;
        unsigned format = 0;
        if (ends_with(name, native_a)) {
            size_t stem_length = strlen(name) - strlen(native_a);
            const char *target_start = name;
            size_t target_length = stem_length;
            static const char diffusion[] = "diffusion_model.";
            if (stem_length > strlen(diffusion) &&
                !strncmp(name, diffusion, strlen(diffusion))) {
                target_start += strlen(diffusion);
                target_length -= strlen(diffusion);
                format = H3_LORA_DIFFUSERS;
            } else format = H3_LORA_NATIVE;
            target = join_suffix(target_start, target_length, ".weight");
            b_name = join_suffix(name, stem_length, native_b);
            alpha_name = join_suffix(name, stem_length, ".alpha");
        } else if (!strncmp(name, kohya_prefix, strlen(kohya_prefix)) &&
                   ends_with(name, kohya_a)) {
            size_t prefix_length = strlen(name) - strlen(kohya_a);
            size_t encoded_length = prefix_length - strlen(kohya_prefix);
            char *encoded = strndup(name + strlen(kohya_prefix), encoded_length);
            target = encoded ? kohya_target(base, encoded, error, error_size) : NULL;
            free(encoded);
            b_name = join_suffix(name, prefix_length, kohya_b);
            alpha_name = join_suffix(name, prefix_length, ".alpha");
            format = H3_LORA_KOHYA;
        } else continue;
        if (!target || !b_name || !alpha_name ||
            !add_pair(lora, base, target, name, b_name, alpha_name, format,
                      error, error_size)) {
            free(target); free(b_name); free(alpha_name);
            return 0;
        }
        free(b_name); free(alpha_name);
    }
    if (!lora->pair_count) {
        fail(error, error_size, "no supported LoRA pairs found");
        return 0;
    }
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const h3_st_tensor *tensor = &lora->header.tensors[index];
        if ((ends_with(tensor->name, native_b) ||
             ends_with(tensor->name, kohya_b)) && !pair_uses(lora, tensor)) {
            fail(error, error_size, "unpaired LoRA tensor: %s", tensor->name);
            return 0;
        }
    }
    return 1;
}

static int compute_sha256(int descriptor, off_t size, char output[65]) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    unsigned char buffer[1024 * 1024];
    off_t offset = 0;
    while (offset < size) {
        size_t wanted = (off_t)sizeof(buffer) < size - offset ?
            sizeof(buffer) : (size_t)(size - offset);
        if (!pread_all(descriptor, buffer, wanted, (uint64_t)offset)) return 0;
        CC_SHA256_Update(&context, buffer, (CC_LONG)wanted);
        offset += (off_t)wanted;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    for (size_t index = 0; index < sizeof(digest); index++)
        snprintf(output + index * 2, 3, "%02x", digest[index]);
    output[64] = '\0';
    return 1;
}

h3_lora *h3_lora_open(const char *path, const h3_weight_store *base,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !base) {
        fail(error, error_size, "LoRA path and checkpoint are required");
        return NULL;
    }
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        fail(error, error_size, "out of memory opening LoRA");
        return NULL;
    }
    lora->descriptor = -1;
    lora->path = strdup(path);
    lora->descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (!lora->path || lora->descriptor < 0 ||
        fstat(lora->descriptor, &lora->identity) != 0 ||
        !S_ISREG(lora->identity.st_mode)) {
        fail(error, error_size, "cannot open LoRA %s: %s", path,
             strerror(errno));
        h3_lora_free(lora);
        return NULL;
    }
    if (!h3_st_read_header(path, &lora->header, error, error_size) ||
        !compute_sha256(lora->descriptor, lora->identity.st_size,
                        lora->sha256) ||
        !inspect_pairs(lora, base, error, error_size)) {
        if (error && error_size && !*error)
            fail(error, error_size, "cannot inspect LoRA %s", path);
        h3_lora_free(lora);
        return NULL;
    }
    struct stat descriptor_after, path_after;
    if (fstat(lora->descriptor, &descriptor_after) != 0 ||
        stat(path, &path_after) != 0 ||
        !same_identity(&lora->identity, &descriptor_after) ||
        !same_identity(&lora->identity, &path_after)) {
        fail(error, error_size, "LoRA changed while it was being inspected");
        h3_lora_free(lora);
        return NULL;
    }
    return lora;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    if (lora->descriptor >= 0) close(lora->descriptor);
    for (size_t index = 0; index < lora->pair_count; index++)
        free((char *)lora->pairs[index].info.target);
    free(lora->pairs);
    h3_st_free_header(&lora->header);
    free(lora->path);
    free(lora);
}

const char *h3_lora_sha256(const h3_lora *lora) {
    return lora ? lora->sha256 : NULL;
}

const char *h3_lora_format(const h3_lora *lora) {
    if (!lora) return "unknown";
    if (lora->formats == H3_LORA_NATIVE) return "native";
    if (lora->formats == H3_LORA_DIFFUSERS) return "diffusion_model";
    if (lora->formats == H3_LORA_KOHYA) return "kohya";
    return "mixed";
}

size_t h3_lora_pair_count(const h3_lora *lora) {
    return lora ? lora->pair_count : 0;
}

const h3_lora_pair_info *h3_lora_pair_at(const h3_lora *lora, size_t index) {
    return lora && index < lora->pair_count ? &lora->pairs[index].info : NULL;
}

int h3_lora_find_pair(const h3_lora *lora, const char *target, size_t *index) {
    if (!lora || !target) return 0;
    for (size_t at = 0; at < lora->pair_count; at++)
        if (!strcmp(lora->pairs[at].info.target, target)) {
            if (index) *index = at;
            return 1;
        }
    return 0;
}

int h3_lora_read_a(const h3_lora *lora, size_t pair_index,
                   float *values, size_t elements,
                   char *error, size_t error_size) {
    if (!lora || pair_index >= lora->pair_count || !values) return 0;
    const h3_lora_pair *pair = &lora->pairs[pair_index];
    if (pair->info.rank > SIZE_MAX / pair->info.input_dim ||
        elements != pair->info.rank * pair->info.input_dim) return 0;
    return read_values(lora, pair->a, 0, elements, values, error, error_size);
}

int h3_lora_read_b_rows(const h3_lora *lora, size_t pair_index,
                        uint64_t first_row, uint64_t rows,
                        float *values, size_t elements,
                        char *error, size_t error_size) {
    if (!lora || pair_index >= lora->pair_count || !values) return 0;
    const h3_lora_pair *pair = &lora->pairs[pair_index];
    if (first_row > pair->info.output_dim ||
        rows > pair->info.output_dim - first_row ||
        rows > SIZE_MAX / pair->info.rank ||
        elements != rows * pair->info.rank) return 0;
    return read_values(lora, pair->b, first_row * pair->info.rank,
                       elements, values, error, error_size);
}
