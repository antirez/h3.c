#include "h3_dit_schedule.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MIN_ROWS(a, b) ((a) < (b) ? (a) : (b))

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_ADALN_OUTPUT,
    FINAL_OUTPUT = H3_DIT_FINAL_ADALN_OUTPUT
};

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition,
                        float **times_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 1.0f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    if (!times) {
        fail(error, error_size, "out of memory allocating timestep rows");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = H3_DIT_VISUAL_CONDITION_TIME;
    if (audio_condition) times[audio_condition_row] = H3_DIT_AUDIO_CONDITION_TIME;
    *times_out = times;
    return 1;
}

static float *timestep_features(const float *times, uint32_t count) {
    float *features = malloc((size_t)count * TIME_INPUT * sizeof(*features));
    if (!features) return NULL;
    for (uint32_t row = 0; row < count; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    return features;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

/* Run the block and final AdaLN projections for every row of time. sink
 * receives each finished table (index H3_DIT_BLOCKS is the final layer) and
 * takes ownership of it. One projection is resident at a time, as before. */
typedef h3_dit_schedule_table_sink table_sink;

static int project_tables(const h3_weight_store *weights, h3_gpu *gpu,
                          const h3_gpu_tensor *time, uint32_t rows,
                          table_sink sink, void *sink_opaque,
                          h3_dit_schedule_progress progress,
                          void *progress_opaque,
                          char *error, size_t error_size) {
    for (unsigned block = 0; block <= H3_DIT_BLOCKS; block++) {
        int final = block == H3_DIT_BLOCKS;
        uint64_t width = final ? FINAL_OUTPUT : BLOCK_OUTPUT;
        char weight_name[128], bias_name[128], operation[128];
        if (final) {
            snprintf(weight_name, sizeof(weight_name),
                     "final_layer.adaln_proj.linear.weight");
            snprintf(bias_name, sizeof(bias_name),
                     "final_layer.adaln_proj.linear.bias");
            snprintf(operation, sizeof(operation), "final AdaLN projection");
        } else {
            snprintf(weight_name, sizeof(weight_name),
                     "blocks.%u.adaln_proj.linear.weight", block);
            snprintf(bias_name, sizeof(bias_name),
                     "blocks.%u.adaln_proj.linear.bias", block);
            snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        }
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, width, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, width, error, error_size);
        h3_gpu_tensor *table = h3_gpu_tensor_new_bf16(
            gpu, (size_t)rows * width);
        if (!weight || !bias || !table) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate %s: %s", operation,
                     h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            free_tensor(&table);
            return 0;
        }
        int ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, table, time, weight, bias, rows, H3_DIT_TIME_DIM,
                (uint32_t)width), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            free_tensor(&table);
            return 0;
        }
        if (!sink(block, table, sink_opaque, error, error_size)) return 0;
        if (progress && !final) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                                         progress_opaque);
    }
    return 1;
}

static int schedule_sink(unsigned index, h3_gpu_tensor *table, void *opaque,
                         char *error, size_t error_size) {
    (void)error;
    (void)error_size;
    h3_dit_schedule *schedule = opaque;
    if (index == H3_DIT_BLOCKS) schedule->final = table;
    else schedule->blocks[index] = table;
    return 1;
}

/* Packed checkpoints drop the 24 GiB of AdaLN projections and keep only their
 * output rows, one per timestep, in adaln_table.r<regime>.*. A row depends on
 * its own timestep and on which kernels ran: h3_gpu_linear_f32/bf16 use direct
 * tiles below 32 rows and MPS from 32 rows, and the two round differently. So
 * the rows of each regime are computed in batches of that regime, and gathering
 * the rows a schedule needs from its regime reproduces the tables computed
 * above bit for bit (h3_pack_int8 --verify checks every schedule it covers). */
enum { MPS_ROWS = 32 };

unsigned h3_dit_schedule_table_regime(uint32_t rows) {
    return rows >= MPS_ROWS ? 1u : 0u;
}

uint32_t h3_dit_schedule_regime_batch(unsigned regime) {
    /* Any batch inside the regime selects the same kernels; 40 rows is the
     * largest MPS batch the 20-step schedules use. */
    return regime ? 40u : MPS_ROWS - 1u;
}

static int read_table_rows(const h3_weight_store *weights, const char *name,
                           uint64_t width, uint64_t table_rows,
                           const uint32_t *source_rows, uint32_t rows,
                           h3_gpu_tensor *destination,
                           char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(weights, name, &header);
    if (!tensor || !header || tensor->dtype != H3_DTYPE_BF16 ||
        tensor->ndim != 2 || tensor->shape[0] != table_rows ||
        tensor->shape[1] != width) {
        fail(error, error_size, "packed AdaLN table %s is absent or malformed",
             name);
        return 0;
    }
    size_t row_bytes = (size_t)width * sizeof(uint16_t);
    uint16_t *row = malloc(row_bytes);
    int descriptor = row ? open(header->path, O_RDONLY | O_CLOEXEC) : -1;
    if (descriptor < 0) {
        free(row);
        fail(error, error_size, "cannot open packed AdaLN table %s: %s",
             header->path, strerror(errno));
        return 0;
    }
    int ok = 1;
    for (uint32_t index = 0; ok && index < rows; index++) {
        off_t offset = (off_t)(tensor->file_offset +
                               (uint64_t)source_rows[index] * row_bytes);
        size_t completed = 0;
        while (ok && completed < row_bytes) {
            ssize_t count = pread(descriptor, (unsigned char *)row + completed,
                                  row_bytes - completed,
                                  offset + (off_t)completed);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) ok = 0;
            else completed += (size_t)count;
        }
        if (ok) ok = h3_gpu_tensor_write_bf16_range(
            destination, (size_t)index * width, row, (size_t)width);
    }
    close(descriptor);
    free(row);
    if (!ok) fail(error, error_size, "cannot read packed AdaLN table %s", name);
    return ok;
}

static int restore_tables(h3_dit_schedule *schedule,
                          const h3_weight_store *weights, const float *times,
                          char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    unsigned regime = h3_dit_schedule_table_regime(schedule->time_rows);
    char name[64];
    snprintf(name, sizeof(name), "adaln_table.r%u.times", regime);
    const h3_st_tensor *tensor = h3_weight_find(weights, name, &header);
    if (!tensor || !header || tensor->dtype != H3_DTYPE_F32 ||
        tensor->ndim != 1 || !tensor->shape[0] ||
        tensor->shape[0] > UINT32_MAX) {
        fail(error, error_size, "packed AdaLN table %s is absent or malformed",
             name);
        return 0;
    }
    uint32_t table_rows = (uint32_t)tensor->shape[0];
    float *table_times = malloc((size_t)table_rows * sizeof(*table_times));
    uint32_t *source_rows = calloc(schedule->time_rows, sizeof(*source_rows));
    int ok = table_times && source_rows &&
        h3_st_read_data(header, tensor, table_times,
                        (size_t)table_rows * sizeof(*table_times),
                        error, error_size);
    for (uint32_t row = 0; ok && row < schedule->time_rows; row++) {
        uint32_t found = UINT32_MAX;
        for (uint32_t index = 0; index < table_rows; index++)
            if (!memcmp(&table_times[index], &times[row], sizeof(float))) {
                found = index;
                break;
            }
        if (found == UINT32_MAX) {
            fail(error, error_size,
                 "the packed DiT has no AdaLN row for timestep %.9g "
                 "(%d steps); repack it with a larger --max-steps",
                 (double)times[row], schedule->steps);
            ok = 0;
        }
        source_rows[row] = found;
    }
    free(table_times);
    for (unsigned block = 0; ok && block <= H3_DIT_BLOCKS; block++) {
        int final = block == H3_DIT_BLOCKS;
        uint64_t width = final ? FINAL_OUTPUT : BLOCK_OUTPUT;
        if (final) snprintf(name, sizeof(name), "adaln_table.r%u.final", regime);
        else snprintf(name, sizeof(name), "adaln_table.r%u.blocks.%u", regime,
                      block);
        h3_gpu_tensor *table = h3_gpu_tensor_new_bf16(
            schedule->gpu, (size_t)schedule->time_rows * width);
        if (!table) {
            fail(error, error_size, "cannot allocate AdaLN table: %s",
                 h3_gpu_error(schedule->gpu));
            ok = 0;
            break;
        }
        if (final) schedule->final = table;
        else schedule->blocks[block] = table;
        ok = read_table_rows(weights, name, width, table_rows, source_rows,
                             schedule->time_rows, table, error, error_size);
    }
    free(source_rows);
    return ok;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    float *times = NULL;
    float *features = NULL;
    h3_gpu_tensor *time = NULL;
    if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                      &times, error, error_size)) goto failed;
    if (h3_dit_schedule_is_packed(weights)) {
        if (!restore_tables(schedule, weights, times, error, error_size))
            goto failed;
        free(times);
        if (progress) progress((int)H3_DIT_BLOCKS, (int)H3_DIT_BLOCKS,
                               progress_opaque);
        return schedule;
    }
    features = timestep_features(times, schedule->time_rows);
    if (!features) {
        fail(error, error_size, "out of memory allocating timestep features");
        goto failed;
    }
    time = time_embeddings(weights, gpu, schedule->time_rows, features,
                           error, error_size);
    if (!time || !project_tables(weights, gpu, time, schedule->time_rows,
                                 schedule_sink, schedule, progress,
                                 progress_opaque, error, error_size))
        goto failed;
    h3_gpu_tensor_free(time);
    free(features);
    free(times);
    return schedule;

failed:
    h3_gpu_tensor_free(time);
    free(features);
    free(times);
    h3_dit_schedule_free(schedule);
    return NULL;
}

int h3_dit_schedule_is_packed(const h3_weight_store *weights) {
    return h3_weight_find(weights, "adaln_table.r0.times", NULL) != NULL ||
           h3_weight_find(weights, "adaln_table.r1.times", NULL) != NULL;
}

int h3_dit_schedule_timesteps(const h3_sigma_schedule *sigmas,
                              int visual_condition, int audio_condition,
                              float **times, uint32_t *count) {
    if (!sigmas || !times || !count || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) return 0;
    h3_dit_schedule schedule = {0};
    float *values = NULL;
    int ok = prepare_rows(&schedule, sigmas, visual_condition,
                          audio_condition, &values, NULL, 0);
    free(schedule.video_rows);
    free(schedule.audio_rows);
    free(schedule.visual_condition_rows);
    free(schedule.audio_condition_rows);
    if (!ok) return 0;
    *times = values;
    *count = schedule.time_rows;
    return 1;
}

int h3_dit_schedule_build_tables(const h3_weight_store *weights, h3_gpu *gpu,
                                 const float *times, uint32_t count,
                                 uint32_t batch,
                                 h3_dit_schedule_table_sink sink,
                                 void *opaque, char *error,
                                 size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !times || !count || !batch || !sink ||
        count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid AdaLN table arguments");
        return 0;
    }
    /* Every projection runs on full batches (the last one padded with its
     * final timestep), so all rows go through the kernels of one regime. */
    uint32_t chunks = (count + batch - 1) / batch;
    float *padded = malloc((size_t)chunks * batch * sizeof(*padded));
    h3_gpu_tensor **time = calloc(chunks, sizeof(*time));
    uint16_t *host = malloc((size_t)batch * BLOCK_OUTPUT * sizeof(*host));
    int ok = padded && time && host;
    if (!ok) fail(error, error_size, "out of memory building AdaLN tables");
    for (size_t index = 0; ok && index < (size_t)chunks * batch; index++)
        padded[index] = times[index < count ? index : count - 1];
    for (uint32_t chunk = 0; ok && chunk < chunks; chunk++) {
        float *features = timestep_features(padded + (size_t)chunk * batch,
                                            batch);
        time[chunk] = features ? time_embeddings(weights, gpu, batch, features,
                                                 error, error_size) : NULL;
        free(features);
        ok = time[chunk] != NULL;
    }
    for (unsigned block = 0; ok && block <= H3_DIT_BLOCKS; block++) {
        int final = block == H3_DIT_BLOCKS;
        uint64_t width = final ? FINAL_OUTPUT : BLOCK_OUTPUT;
        char weight_name[128], bias_name[128];
        if (final) {
            snprintf(weight_name, sizeof(weight_name),
                     "final_layer.adaln_proj.linear.weight");
            snprintf(bias_name, sizeof(bias_name),
                     "final_layer.adaln_proj.linear.bias");
        } else {
            snprintf(weight_name, sizeof(weight_name),
                     "blocks.%u.adaln_proj.linear.weight", block);
            snprintf(bias_name, sizeof(bias_name),
                     "blocks.%u.adaln_proj.linear.bias", block);
        }
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, width, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, width, error, error_size);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, (size_t)batch * width);
        h3_gpu_tensor *table = h3_gpu_tensor_new_bf16(gpu, (size_t)count * width);
        ok = weight && bias && output && table;
        for (uint32_t chunk = 0; ok && chunk < chunks; chunk++) {
            uint32_t rows = MIN_ROWS(batch, count - chunk * batch);
            ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                        "AdaLN table") &&
                gpu_op(gpu, h3_gpu_linear_bf16(
                    gpu, output, time[chunk], weight, bias, batch,
                    H3_DIT_TIME_DIM, (uint32_t)width), error, error_size,
                    "AdaLN table projection") &&
                gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                       "AdaLN table") &&
                h3_gpu_tensor_read_bf16(output, host, (size_t)rows * width) &&
                h3_gpu_tensor_write_bf16_range(
                    table, (size_t)chunk * batch * width, host,
                    (size_t)rows * width);
        }
        free_tensor(&weight);
        free_tensor(&bias);
        free_tensor(&output);
        if (!ok) {
            if (!error || !*error)
                fail(error, error_size, "cannot build AdaLN table %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&table);
            break;
        }
        ok = sink(block, table, opaque, error, error_size);
    }
    for (uint32_t chunk = 0; time && chunk < chunks; chunk++)
        h3_gpu_tensor_free(time[chunk]);
    free(time);
    free(padded);
    free(host);
    return ok;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}
