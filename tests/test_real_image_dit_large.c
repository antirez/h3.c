#include "h3_dit.h"
#include "h3_gpu.h"
#include "h3_image.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { TEXT_DIM = 5120, PATCH = 96 };

/* One step-0 velocity at a large canvas, against the float32 CPU run of the
 * same checkpoint (the large step fixture). Rows are the image rows of the
 * fixture: 4096 is the 2048x2048 canvas (4101 tokens), 9216 the 3072x3072 one.
 * The bounds are about 1.4x the values measured on the development Mac.
 *
 *   rows   mode   rel-L2   median row  worst row
 *   4096   bf16   0.0133   0.0106      0.058    (stream is bit-identical)
 *   4096   int8   0.0269   0.0258      0.074    (0.0252 from bf16)
 *   9216   bf16   0.0831   0.0121      0.556    (stream is bit-identical)
 *   9216   int8   0.0673   0.0264      0.358    (0.0364 from bf16)
 *
 * At 9216 rows the median row is as good as at 4096, but about 1% of the rows,
 * one spatial blob in the middle of the image, carry a quarter of the squared
 * error, so the global rel-L2 is not comparable between the two sizes. Running
 * the attention in float32 (cast query, key and value, h3_gpu_sdpa_f32) brings
 * the 9216-row bf16 run to 0.0390 with worst row 0.238, so the BF16 attention
 * is the largest single contributor to that cluster. int8 is closer to the
 * truth than bf16 at 9216 rows in the global number only. */
typedef struct {
    size_t rows;
    double bf16_l2, bf16_median, int8_l2, int8_median;
    double int8_vs_bf16_l2;
} bounds;

static const bounds BOUNDS[] = {
    {4096, 0.019, 0.015, 0.038, 0.036, 0.036},
    {9216, 0.12, 0.017, 0.095, 0.037, 0.051},
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_dit_large.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_dit_large.c: fixture "
                "tensor %s is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static int32_t read_scalar(const h3_st_header *fixture, const char *name) {
    int32_t *data = read_fixture(fixture, name, H3_DTYPE_I32, 1);
    int32_t value = data[0];
    free(data);
    return value;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

typedef struct {
    double l2, maximum, row_median, row_worst;
} comparison;

static comparison compare(const float *got, const float *want, size_t rows,
                          size_t width) {
    double *row_errors = malloc(rows * sizeof(*row_errors));
    require(row_errors != NULL, "comparison allocation failed");
    comparison result = {0.0, 0.0, 0.0, 0.0};
    double square_error = 0.0, square_value = 0.0;
    for (size_t row = 0; row < rows; row++) {
        double row_error = 0.0, row_value = 0.0;
        for (size_t index = row * width; index < (row + 1) * width; index++) {
            double delta = (double)got[index] - (double)want[index];
            if (fabs(delta) > result.maximum) result.maximum = fabs(delta);
            row_error += delta * delta;
            row_value += (double)want[index] * (double)want[index];
        }
        square_error += row_error;
        square_value += row_value;
        row_errors[row] = sqrt(row_error / (row_value > 1e-24 ? row_value :
                                                                 1e-24));
    }
    qsort(row_errors, rows, sizeof(*row_errors), compare_doubles);
    result.row_median = row_errors[rows / 2];
    result.row_worst = row_errors[rows - 1];
    result.l2 = sqrt(square_error /
                     (square_value > 1e-24 ? square_value : 1e-24));
    free(row_errors);
    return result;
}

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void progress(const char *phase, int completed, int total,
                     void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "  %s %d/%d\n", phase, completed, total);
}

typedef struct {
    uint32_t magic;
    uint32_t mode;
    uint32_t elements;
    uint32_t reserved;
} peer_header;

#define PEER_MAGIC 0x4c524750u

/* The first run stores its velocity in `path`, the second compares to it. The
 * stream and bf16 modes read the same weights and run the same kernels, so
 * they must agree bit for bit; int8 is only compared by rel-L2. */
static void check_peer(const char *path, uint32_t mode, const float *velocity,
                       size_t elements, size_t rows,
                       const bounds *limit) {
    FILE *file = fopen(path, "rb");
    peer_header header = {PEER_MAGIC, mode, (uint32_t)elements, 0};
    if (!file) {
        file = fopen(path, "wb");
        require(file != NULL, "cannot create the peer velocity file");
        int ok = fwrite(&header, sizeof(header), 1, file) == 1 &&
                 fwrite(velocity, sizeof(float), elements, file) == elements;
        require(fclose(file) == 0 && ok, "cannot write the peer file");
        printf("velocity stored for the peer comparison\n");
        return;
    }
    peer_header stored;
    require(fread(&stored, sizeof(stored), 1, file) == 1 &&
            stored.magic == PEER_MAGIC && stored.elements == elements,
            "peer velocity file is invalid or for another fixture");
    require(stored.mode != mode, "peer velocity file is from the same mode: "
            "remove the stale file");
    float *other = malloc(elements * sizeof(*other));
    require(other != NULL, "oom");
    require(fread(other, sizeof(float), elements, file) == elements,
            "peer velocity file is short");
    fclose(file);
    size_t different = 0;
    for (size_t index = 0; index < elements; index++)
        if (memcmp(&other[index], &velocity[index], sizeof(float)))
            different++;
    comparison peer = compare(velocity, other, rows, PATCH);
    printf("vs the other run: rel-L2 %.6g, max abs %.6g, median row %.6g, "
           "%zu of %zu elements differ\n", peer.l2, peer.maximum,
           peer.row_median, different, elements);
    if (stored.mode != 2 && mode != 2)
        require(different == 0, "stream and bf16 velocities are not "
                "bit-identical");
    else
        require(peer.l2 < limit->int8_vs_bf16_l2,
                "int8 differs from bf16 by more than the bound");
    free(other);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s WEIGHT_DIR TEXT_FIXTURE LARGE_FIXTURE "
                "stream|bf16|int8 [PEER_FILE]\n", argv[0]);
        return 2;
    }
    const char *mode = argv[4];
    int stream = !strcmp(mode, "stream");
    int bf16 = !strcmp(mode, "bf16");
    int int8 = !strcmp(mode, "int8");
    require(stream || bf16 || int8, "mode must be stream, bf16 or int8");
    char error[512];
    h3_st_header text_fixture, large_fixture;
    if (!h3_st_read_header(argv[2], &text_fixture, error, sizeof(error)))
        die(error);
    if (!h3_st_read_header(argv[3], &large_fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *prompt = h3_st_find(&text_fixture, "prompt_embeds");
    require(prompt && prompt->ndim == 2 && prompt->shape[1] == TEXT_DIM,
            "bad prompt_embeds fixture");
    size_t tokens = (size_t)prompt->shape[0];
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    text.tokens = tokens;
    text.width = TEXT_DIM;
    text.values = read_fixture(&text_fixture, "prompt_embeds", H3_DTYPE_BF16,
                               tokens * TEXT_DIM);
    int latent_height = read_scalar(&large_fixture, "latent_height");
    int latent_width = read_scalar(&large_fixture, "latent_width");
    require(latent_height > 0 && latent_width > 0 &&
            latent_height % 2 == 0 && latent_width % 2 == 0,
            "bad latent size in the large step fixture");
    size_t rows = (size_t)(latent_height / 2) * (size_t)(latent_width / 2);
    size_t elements = rows * PATCH;
    float *input = read_fixture(&large_fixture, "latents_in_step0",
                                H3_DTYPE_F32, elements);
    float *truth = read_fixture(&large_fixture, "noise_pred_f32_step0",
                                H3_DTYPE_F32, elements);
    const bounds *limit = NULL;
    for (size_t index = 0; index < sizeof(BOUNDS) / sizeof(BOUNDS[0]); index++)
        if (BOUNDS[index].rows == rows) limit = &BOUNDS[index];
    require(limit != NULL, "no bounds for this fixture size");

    printf("mode %s: %zu text tokens, latent %dx%d, %zu image rows\n", mode,
           tokens, latent_height, latent_width, rows);
    double started = seconds_now();
    h3_dit *dit = h3_dit_load_image(
        argv[1], "h3_shaders.metal", &text, latent_height, latent_width, 50,
        stream, bf16, bf16, bf16, 0, 0, 0, 0, 0, 0, 0, 0, progress, NULL,
        error, sizeof(error));
    if (!dit) die(error);
    printf("load: %.1f s\n", seconds_now() - started);
    require(h3_dit_image_rows(dit) == rows, "image row count mismatch");

    float *velocity = malloc(elements * sizeof(*velocity));
    require(velocity != NULL, "oom");
    double forward_started = seconds_now();
    if (!h3_dit_forward_image(dit, 0, input, velocity, error, sizeof(error)))
        die(error);
    double elapsed = seconds_now() - forward_started;
    for (size_t index = 0; index < elements; index++)
        require(isfinite(velocity[index]), "non-finite velocity");
    comparison result = compare(velocity, truth, rows, PATCH);
    double footprint = 0.0, footprint_peak = 0.0;
    h3_gpu_process_footprint(&footprint, &footprint_peak);
    printf("step 0 vs F32 truth: rel-L2 %.6g, max abs %.6g, median row %.6g, "
           "worst row %.6g (%.2f s, footprint peak %.1f GiB)\n", result.l2,
           result.maximum, result.row_median, result.row_worst, elapsed,
           footprint_peak);
    require(result.l2 < (int8 ? limit->int8_l2 : limit->bf16_l2),
            "velocity rel-L2 against the F32 truth over the bound");
    require(result.row_median < (int8 ? limit->int8_median :
                                        limit->bf16_median),
            "velocity median row error against the F32 truth over the bound");

    float *again = malloc(elements * sizeof(*again));
    require(again != NULL, "oom");
    require(h3_dit_forward_image(dit, 0, input, again, error, sizeof(error)),
            error);
    int same = !memcmp(again, velocity, elements * sizeof(float));
    printf("repeated forward bit-identical: %s\n", same ? "yes" : "NO");
    require(same, "a repeated forward changed the velocity");
    free(again);
    h3_dit_free(dit);
    if (argc > 5) check_peer(argv[5], int8 ? 2u : stream ? 1u : 0u, velocity,
                             elements, rows, limit);
    printf("ok: image DiT %s against the F32 truth at %zu rows\n", mode, rows);
    return 0;
}
