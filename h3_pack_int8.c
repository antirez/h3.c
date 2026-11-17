/* Pack a BF16 MiniMax H3 DiT checkpoint into the INT8 form h3_dit_load reads
 * from <variant>/transformer-int8:
 *
 *   base.safetensors         every tensor except the 200 block matrices and
 *                            the AdaLN projections, copied unchanged
 *   blocks-NN.safetensors    blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,
 *                            mlp.fc2}.{qweight I8, qscale F32}, produced by the
 *                            same GPU quantizer the resident loader runs, so a
 *                            packed load is byte-identical to quantizing on load
 *   adaln-table.safetensors  per kernel regime r (see h3_dit_schedule.h):
 *                            adaln_table.rR.times F32 [T] and the AdaLN outputs
 *                            for those timesteps: adaln_table.rR.blocks.N BF16
 *                            [T, 96768] and adaln_table.rR.final BF16 [T, 10752]
 *   config.json, h3-int8-pack.json (marker, written last)
 *
 * Regime R holds every distinct timestep of the serving schedules with
 * 2..--max-steps evaluations (with and without each condition) whose row count
 * falls in R. --verify then rebuilds the AdaLN tables of all those schedules
 * from both checkpoints and compares them bit for bit. --table-only rewrites
 * just adaln-table.safetensors in an existing pack. */
#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum { HIDDEN = 5376, INNER = 7168, FFN = 14336 };

typedef struct {
    char name[128];
    const char *dtype;
    int ndim;
    uint64_t shape[2];
    uint64_t bytes;
} entry;

typedef struct {
    FILE *file;
    char path[1024];
    char temporary[1040];
    entry *entries;
    size_t count;
    size_t written;
    uint64_t remaining;
} writer;

static int die(const char *format, ...) __attribute__((format(printf, 1, 2)));
static int die(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    fputs("h3_pack_int8: ", stderr);
    vfprintf(stderr, format, arguments);
    fputc('\n', stderr);
    va_end(arguments);
    return 0;
}

/* Declare every tensor first, then append their payloads in order. The file
 * is written under a temporary name and renamed when complete. */
static int writer_open(writer *out, const char *directory, const char *name,
                       entry *entries, size_t count) {
    memset(out, 0, sizeof(*out));
    snprintf(out->path, sizeof(out->path), "%s/%s", directory, name);
    snprintf(out->temporary, sizeof(out->temporary), "%s.tmp", out->path);
    out->entries = entries;
    out->count = count;
    size_t capacity = 256 + count * 256;
    char *json = malloc(capacity);
    if (!json) return die("out of memory");
    size_t length = (size_t)snprintf(json, capacity,
        "{\"__metadata__\":{\"format\":\"pt\",\"h3\":\"int8-pack-v1\"}");
    uint64_t offset = 0;
    for (size_t index = 0; index < count; index++) {
        const entry *item = &entries[index];
        length += (size_t)snprintf(json + length, capacity - length,
            ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu", item->name,
            item->dtype, (unsigned long long)item->shape[0]);
        if (item->ndim == 2)
            length += (size_t)snprintf(json + length, capacity - length,
                ",%llu", (unsigned long long)item->shape[1]);
        length += (size_t)snprintf(json + length, capacity - length,
            "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)offset,
            (unsigned long long)(offset + item->bytes));
        offset += item->bytes;
        if (length + 300 > capacity) {
            free(json);
            return die("header for %s is too large", name);
        }
    }
    json[length++] = '}';
    while (length % 8) json[length++] = ' ';
    uint64_t header_size = length;
    out->file = fopen(out->temporary, "wb");
    int ok = out->file &&
        fwrite(&header_size, sizeof(header_size), 1, out->file) == 1 &&
        fwrite(json, 1, length, out->file) == length;
    free(json);
    if (!ok) return die("cannot write %s: %s", out->temporary, strerror(errno));
    out->remaining = entries[0].bytes;
    return 1;
}

static int writer_append(writer *out, const char *name, const void *data,
                         uint64_t bytes) {
    if (out->written >= out->count ||
        strcmp(out->entries[out->written].name, name) ||
        out->entries[out->written].bytes != bytes)
        return die("%s: unexpected tensor %s", out->path, name);
    if (bytes && fwrite(data, 1, (size_t)bytes, out->file) != bytes)
        return die("cannot write %s: %s", out->temporary, strerror(errno));
    out->written++;
    return 1;
}

static int writer_close(writer *out) {
    int ok = out->written == out->count;
    if (fclose(out->file) != 0) ok = 0;
    out->file = NULL;
    if (ok && rename(out->temporary, out->path) == 0) return 1;
    remove(out->temporary);
    return die("cannot finish %s", out->path);
}

/* ---- source enumeration ------------------------------------------------ */

typedef struct {
    h3_st_header *headers;
    size_t count;
} shards;

static int has_suffix(const char *text, const char *suffix) {
    size_t a = strlen(text), b = strlen(suffix);
    return a >= b && !strcmp(text + a - b, suffix);
}

static int compare_names(const void *left, const void *right) {
    return strcmp(*(char *const *)left, *(char *const *)right);
}

static int open_shards(const char *directory, shards *out) {
    DIR *dir = opendir(directory);
    if (!dir) return die("cannot open %s: %s", directory, strerror(errno));
    char *names[256];
    size_t count = 0;
    struct dirent *item;
    while ((item = readdir(dir)) && count < 256)
        if (item->d_name[0] != '.' && has_suffix(item->d_name, ".safetensors"))
            names[count++] = strdup(item->d_name);
    closedir(dir);
    qsort(names, count, sizeof(*names), compare_names);
    out->headers = calloc(count, sizeof(*out->headers));
    out->count = count;
    for (size_t index = 0; index < count; index++) {
        char path[1024], error[512];
        snprintf(path, sizeof(path), "%s/%s", directory, names[index]);
        free(names[index]);
        if (!h3_st_read_header(path, &out->headers[index], error,
                               sizeof(error)))
            return die("%s", error);
    }
    return count ? 1 : die("%s has no safetensors shards", directory);
}

static int block_matrix(const char *name) {
    unsigned block;
    char rest[96];
    if (sscanf(name, "blocks.%u.%95s", &block, rest) != 2) return 0;
    return !strcmp(rest, "attn.qkv_proj.weight") ||
           !strcmp(rest, "attn.out_proj.weight") ||
           !strcmp(rest, "mlp.fc1.weight") || !strcmp(rest, "mlp.fc2.weight");
}

static int copy_base(const shards *source, const char *destination) {
    size_t total = 0;
    for (size_t shard = 0; shard < source->count; shard++)
        total += source->headers[shard].tensor_count;
    entry *entries = calloc(total, sizeof(*entries));
    const h3_st_tensor **tensors = calloc(total, sizeof(*tensors));
    const h3_st_header **owners = calloc(total, sizeof(*owners));
    size_t count = 0;
    for (size_t shard = 0; shard < source->count; shard++) {
        const h3_st_header *header = &source->headers[shard];
        for (size_t index = 0; index < header->tensor_count; index++) {
            const h3_st_tensor *tensor = &header->tensors[index];
            if (block_matrix(tensor->name) || strstr(tensor->name, "adaln_proj"))
                continue;
            if (tensor->ndim < 1 || tensor->ndim > 2)
                return die("unexpected rank for %s", tensor->name);
            entry *item = &entries[count];
            snprintf(item->name, sizeof(item->name), "%s", tensor->name);
            item->dtype = h3_dtype_name(tensor->dtype);
            item->ndim = tensor->ndim;
            item->shape[0] = tensor->shape[0];
            item->shape[1] = tensor->ndim == 2 ? tensor->shape[1] : 0;
            item->bytes = tensor->data_end - tensor->data_begin;
            tensors[count] = tensor;
            owners[count++] = header;
        }
    }
    writer out;
    if (!writer_open(&out, destination, "base.safetensors", entries, count))
        return 0;
    for (size_t index = 0; index < count; index++) {
        void *data = malloc((size_t)entries[index].bytes);
        char error[512];
        if (!data || !h3_st_read_data(owners[index], tensors[index], data,
                                      (size_t)entries[index].bytes,
                                      error, sizeof(error)))
            return die("cannot read %s: %s", entries[index].name, error);
        int ok = writer_append(&out, entries[index].name, data,
                               entries[index].bytes);
        free(data);
        if (!ok) return 0;
    }
    free(tensors);
    free(owners);
    int ok = writer_close(&out);
    free(entries);
    fprintf(stderr, "base: %zu tensors\n", count);
    return ok;
}

/* ---- block matrices ---------------------------------------------------- */

static const struct {
    const char *suffix;
    uint32_t rows;
    uint32_t columns;
} matrices[] = {
    {"attn.qkv_proj", INNER * 3, HIDDEN},
    {"attn.out_proj", HIDDEN, INNER},
    {"mlp.fc1", FFN * 2, HIDDEN},
    {"mlp.fc2", HIDDEN, FFN},
};

static int pack_block(const h3_weight_store *store, h3_gpu *gpu,
                      unsigned block, const char *destination) {
    entry entries[8];
    for (unsigned index = 0; index < 4; index++) {
        entry *weight = &entries[2 * index];
        entry *scale = &entries[2 * index + 1];
        snprintf(weight->name, sizeof(weight->name), "blocks.%u.%s.qweight",
                 block, matrices[index].suffix);
        weight->dtype = "I8";
        weight->ndim = 2;
        weight->shape[0] = matrices[index].rows;
        weight->shape[1] = matrices[index].columns;
        weight->bytes = (uint64_t)matrices[index].rows *
                        matrices[index].columns;
        snprintf(scale->name, sizeof(scale->name), "blocks.%u.%s.qscale",
                 block, matrices[index].suffix);
        scale->dtype = "F32";
        scale->ndim = 1;
        scale->shape[0] = matrices[index].rows;
        scale->bytes = (uint64_t)matrices[index].rows * sizeof(float);
    }
    char file[64];
    snprintf(file, sizeof(file), "blocks-%02u.safetensors", block);
    writer out;
    if (!writer_open(&out, destination, file, entries, 8)) return 0;
    for (unsigned index = 0; index < 4; index++) {
        uint32_t rows = matrices[index].rows;
        uint32_t columns = matrices[index].columns;
        char name[160], error[512] = "";
        snprintf(name, sizeof(name), "blocks.%u.%s.weight", block,
                 matrices[index].suffix);
        uint64_t shape[] = {rows, columns};
        h3_gpu_tensor *source = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                                    error, sizeof(error));
        size_t elements = (size_t)rows * columns;
        h3_gpu_tensor *weight = h3_gpu_tensor_new_i8(gpu, elements);
        h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, rows);
        int8_t *host_weight = malloc(elements);
        float *host_scales = malloc((size_t)rows * sizeof(float));
        int ok = source && weight && scales && host_weight && host_scales &&
            h3_gpu_begin(gpu) &&
            h3_gpu_quantize_weight_int8(gpu, weight, scales, source, rows,
                                        columns) &&
            h3_gpu_submit(gpu) &&
            h3_gpu_tensor_read_i8(weight, host_weight, elements) &&
            h3_gpu_tensor_read_f32(scales, host_scales, rows);
        h3_gpu_tensor_free(source);
        h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(scales);
        if (!ok) {
            free(host_weight);
            free(host_scales);
            return die("cannot quantize %s: %s %s", name, error,
                       h3_gpu_error(gpu));
        }
        ok = writer_append(&out, entries[2 * index].name, host_weight,
                           entries[2 * index].bytes) &&
             writer_append(&out, entries[2 * index + 1].name, host_scales,
                           entries[2 * index + 1].bytes);
        free(host_weight);
        free(host_scales);
        if (!ok) return 0;
    }
    return writer_close(&out);
}

/* ---- AdaLN table ------------------------------------------------------- */

static int compare_floats(const void *left, const void *right) {
    float a = *(const float *)left, b = *(const float *)right;
    return (a > b) - (a < b);
}

static void add_time(float **times, size_t *count, size_t *capacity,
                     float value) {
    for (size_t index = 0; index < *count; index++)
        if (!memcmp(&(*times)[index], &value, sizeof(float))) return;
    if (*count == *capacity)
        *times = realloc(*times, (*capacity = *capacity ? *capacity * 2 : 64) *
                                 sizeof(**times));
    (*times)[(*count)++] = value;
}

static int collect_times(int max_steps, float **times_out,
                         uint32_t *counts_out) {
    size_t capacity[H3_DIT_TABLE_REGIMES] = {0};
    size_t count[H3_DIT_TABLE_REGIMES] = {0};
    for (int steps = 2; steps <= max_steps; steps++) {
        h3_sigma_schedule sigmas;
        if (!h3_serving_schedule_build(steps, &sigmas))
            return die("cannot build the %d-step schedule", steps);
        for (int conditions = 0; conditions < 4; conditions++) {
            float *rows = NULL;
            uint32_t row_count = 0;
            if (!h3_dit_schedule_timesteps(&sigmas, conditions & 1,
                                           conditions >> 1, &rows, &row_count))
                return die("cannot build the %d-step timesteps", steps);
            unsigned regime = h3_dit_schedule_table_regime(row_count);
            for (uint32_t row = 0; row < row_count; row++)
                add_time(&times_out[regime], &count[regime], &capacity[regime],
                         rows[row]);
            free(rows);
        }
    }
    for (unsigned regime = 0; regime < H3_DIT_TABLE_REGIMES; regime++) {
        if (count[regime])
            qsort(times_out[regime], count[regime], sizeof(float),
                  compare_floats);
        counts_out[regime] = (uint32_t)count[regime];
    }
    return 1;
}

typedef struct {
    writer *out;
    unsigned regime;
    uint32_t rows;
} table_state;

static int table_sink(unsigned index, h3_gpu_tensor *table, void *opaque,
                      char *error, size_t error_size) {
    table_state *state = opaque;
    int final = index == H3_DIT_BLOCKS;
    size_t elements = (size_t)state->rows *
        (final ? H3_DIT_FINAL_ADALN_OUTPUT : H3_DIT_ADALN_OUTPUT);
    uint16_t *host = malloc(elements * sizeof(*host));
    char name[64];
    if (final) snprintf(name, sizeof(name), "adaln_table.r%u.final",
                        state->regime);
    else snprintf(name, sizeof(name), "adaln_table.r%u.blocks.%u",
                  state->regime, index);
    int ok = host && h3_gpu_tensor_read_bf16(table, host, elements) &&
        writer_append(state->out, name, host, elements * sizeof(*host));
    free(host);
    h3_gpu_tensor_free(table);
    if (!ok) snprintf(error, error_size, "cannot write %s", name);
    else if (!final) fprintf(stderr, "\radaln r%u: block %u/%u", state->regime,
                             index + 1, H3_DIT_BLOCKS);
    return ok;
}

static int pack_table(const h3_weight_store *store, h3_gpu *gpu,
                      int max_steps, const char *destination,
                      uint32_t *rows_out) {
    float *times[H3_DIT_TABLE_REGIMES] = {0};
    uint32_t count[H3_DIT_TABLE_REGIMES] = {0};
    if (!collect_times(max_steps, times, count)) return 0;
    entry entries[H3_DIT_TABLE_REGIMES * (H3_DIT_BLOCKS + 2)];
    memset(entries, 0, sizeof(entries));
    size_t total = 0;
    for (unsigned regime = 0; regime < H3_DIT_TABLE_REGIMES; regime++) {
        if (!count[regime]) continue;
        entry *item = &entries[total++];
        snprintf(item->name, sizeof(item->name), "adaln_table.r%u.times", regime);
        item->dtype = "F32";
        item->ndim = 1;
        item->shape[0] = count[regime];
        item->bytes = (uint64_t)count[regime] * sizeof(float);
        for (unsigned index = 0; index <= H3_DIT_BLOCKS; index++) {
            int final = index == H3_DIT_BLOCKS;
            item = &entries[total++];
            if (final)
                snprintf(item->name, sizeof(item->name),
                         "adaln_table.r%u.final", regime);
            else
                snprintf(item->name, sizeof(item->name),
                         "adaln_table.r%u.blocks.%u", regime, index);
            item->dtype = "BF16";
            item->ndim = 2;
            item->shape[0] = count[regime];
            item->shape[1] = final ? H3_DIT_FINAL_ADALN_OUTPUT :
                                     H3_DIT_ADALN_OUTPUT;
            item->bytes = (uint64_t)count[regime] * item->shape[1] *
                          sizeof(uint16_t);
        }
    }
    writer out;
    if (!writer_open(&out, destination, "adaln-table.safetensors", entries,
                     total)) return 0;
    *rows_out = 0;
    for (unsigned regime = 0; regime < H3_DIT_TABLE_REGIMES; regime++) {
        if (!count[regime]) continue;
        char name[64], error[512] = "";
        snprintf(name, sizeof(name), "adaln_table.r%u.times", regime);
        if (!writer_append(&out, name, times[regime],
                           (uint64_t)count[regime] * sizeof(float))) return 0;
        table_state state = {&out, regime, count[regime]};
        int ok = h3_dit_schedule_build_tables(
            store, gpu, times[regime], count[regime],
            h3_dit_schedule_regime_batch(regime), table_sink, &state,
            error, sizeof(error));
        fprintf(stderr, " (%u rows)\n", count[regime]);
        free(times[regime]);
        if (!ok) return die("AdaLN table: %s", error);
        *rows_out += count[regime];
    }
    return writer_close(&out);
}

/* ---- verification ------------------------------------------------------ */

static int same_tensor(const h3_gpu_tensor *left, const h3_gpu_tensor *right) {
    if (!left || !right) return left == right;
    size_t elements = h3_gpu_tensor_elements(left);
    if (elements != h3_gpu_tensor_elements(right)) return 0;
    uint16_t *a = malloc(elements * sizeof(*a));
    uint16_t *b = malloc(elements * sizeof(*b));
    int same = a && b && h3_gpu_tensor_read_bf16(left, a, elements) &&
        h3_gpu_tensor_read_bf16(right, b, elements) &&
        !memcmp(a, b, elements * sizeof(*a));
    free(a);
    free(b);
    return same;
}

static int verify_schedule(const h3_weight_store *bf16,
                           const h3_weight_store *packed, h3_gpu *gpu,
                           int steps, int visual, int audio) {
    h3_sigma_schedule sigmas;
    char error[512] = "";
    if (!h3_serving_schedule_build(steps, &sigmas))
        return die("cannot build the %d-step schedule", steps);
    h3_dit_schedule *expected = h3_dit_schedule_precompute(
        bf16, gpu, &sigmas, visual, audio, NULL, NULL, error, sizeof(error));
    if (!expected) return die("BF16 schedule: %s", error);
    h3_dit_schedule *actual = h3_dit_schedule_precompute(
        packed, gpu, &sigmas, visual, audio, NULL, NULL, error, sizeof(error));
    if (!actual) {
        h3_dit_schedule_free(expected);
        return die("packed schedule: %s", error);
    }
    int same = h3_dit_schedule_time_rows(expected) ==
               h3_dit_schedule_time_rows(actual) &&
        same_tensor(h3_dit_schedule_final(expected),
                    h3_dit_schedule_final(actual));
    for (unsigned block = 0; same && block < H3_DIT_BLOCKS; block++)
        same = same_tensor(h3_dit_schedule_block(expected, block),
                           h3_dit_schedule_block(actual, block));
    fprintf(stderr, "verify steps=%d visual=%d audio=%d rows=%u: %s\n", steps,
            visual, audio, h3_dit_schedule_time_rows(expected),
            same ? "identical" : "DIFFERENT");
    h3_dit_schedule_free(expected);
    h3_dit_schedule_free(actual);
    return same;
}

static int verify(const char *source, const char *destination, h3_gpu *gpu,
                  int max_steps) {
    char error[512] = "";
    h3_weight_store *bf16 = h3_weight_store_open(source, error, sizeof(error));
    h3_weight_store *packed = bf16 ?
        h3_weight_store_open(destination, error, sizeof(error)) : NULL;
    if (!packed) return die("%s", error);
    int ok = 1;
    for (int steps = 2; steps <= max_steps; steps++)
        for (int conditions = 0; conditions < 4; conditions++)
            ok &= verify_schedule(bf16, packed, gpu, steps, conditions & 1,
                                  conditions >> 1);
    h3_weight_store_free(bf16);
    h3_weight_store_free(packed);
    return ok;
}

/* ---- main -------------------------------------------------------------- */

static int copy_file(const char *from, const char *to) {
    FILE *in = fopen(from, "rb"), *out = in ? fopen(to, "wb") : NULL;
    if (!out) {
        if (in) fclose(in);
        return die("cannot copy %s to %s", from, to);
    }
    char buffer[65536];
    size_t count;
    int ok = 1;
    while ((count = fread(buffer, 1, sizeof(buffer), in)))
        if (fwrite(buffer, 1, count, out) != count) ok = 0;
    fclose(in);
    if (fclose(out) != 0) ok = 0;
    return ok ? 1 : die("cannot write %s", to);
}

int main(int argc, char **argv) {
    const char *source = NULL, *destination = NULL;
    int max_steps = 20, verify_only = 0, run_verify = 0, table_only = 0;
    for (int index = 1; index < argc; index++) {
        if (!strcmp(argv[index], "--max-steps") && index + 1 < argc)
            max_steps = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--verify")) run_verify = 1;
        else if (!strcmp(argv[index], "--verify-only")) verify_only = 1;
        else if (!strcmp(argv[index], "--table-only")) table_only = 1;
        else if (!source) source = argv[index];
        else if (!destination) destination = argv[index];
        else destination = NULL, source = NULL, index = argc;
    }
    if (!source || !destination || max_steps < 2 || max_steps > 100) {
        fprintf(stderr, "usage: h3_pack_int8 SRC/transformer DST/transformer-int8 "
                        "[--max-steps N] [--table-only] [--verify | --verify-only]\n"
                        "Run from the h3.c directory (it loads h3_shaders.metal).\n");
        return 2;
    }
    char error[512] = "";
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) return die("%s", error) ? 0 : 1;
    if (!h3_gpu_has_int8_mlp(gpu)) return die("this GPU has no int8 kernels") ? 0 : 1;
    if (verify_only) {
        int same = verify(source, destination, gpu, max_steps);
        h3_gpu_free(gpu);
        return same ? 0 : 1;
    }
    char marker[1024];
    snprintf(marker, sizeof(marker), "%s/h3-int8-pack.json", destination);
    remove(marker);
    if (mkdir(destination, 0755) != 0 && errno != EEXIST)
        return die("cannot create %s: %s", destination, strerror(errno)) ? 0 : 1;
    shards source_shards = {0};
    h3_weight_store *store = h3_weight_store_open(source, error, sizeof(error));
    if (!store) return die("%s", error) ? 0 : 1;
    time_t started = time(NULL);
    if (!table_only) {
        if (!open_shards(source, &source_shards) ||
            !copy_base(&source_shards, destination)) return 1;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            if (!pack_block(store, gpu, block, destination)) return 1;
            fprintf(stderr, "\rblocks: %u/%u (%lds)", block + 1,
                    H3_DIT_BLOCKS, (long)(time(NULL) - started));
        }
        fputc('\n', stderr);
    }
    uint32_t rows = 0;
    if (!pack_table(store, gpu, max_steps, destination, &rows)) return 1;
    h3_weight_store_free(store);
    char from[1024], to[1024];
    snprintf(from, sizeof(from), "%s/config.json", source);
    snprintf(to, sizeof(to), "%s/config.json", destination);
    if (!copy_file(from, to)) return 1;
    if (run_verify && !verify(source, destination, gpu, max_steps)) {
        h3_gpu_free(gpu);
        return die("verification failed; marker not written") ? 0 : 1;
    }
    FILE *file = fopen(marker, "w");
    if (!file) return die("cannot write %s", marker) ? 0 : 1;
    fprintf(file, "{\"format\": \"h3-int8-pack-v1\", \"source\": \"%s\", "
                  "\"max_steps\": %d, \"adaln_rows\": %u, \"verified\": %s, "
                  "\"packed_at\": %ld}\n", source, max_steps, rows,
            run_verify ? "true" : "false", (long)time(NULL));
    fclose(file);
    h3_gpu_free(gpu);
    fprintf(stderr, "packed %s -> %s in %lds (%u AdaLN rows)\n", source,
            destination, (long)(time(NULL) - started), rows);
    return 0;
}
