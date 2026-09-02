#include "h3_lora.h"
#include "h3_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL real LoRA parity: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s BASE_TRANSFORMER FOLDED_TRANSFORMER LORA\n",
                argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *base = h3_weight_store_open(argv[1], error, sizeof(error));
    h3_weight_store *folded = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!base || !folded) die(error);
    h3_lora_spec spec = {argv[3], 1.0f, H3_LORA_FL2VA};
    h3_lora_set *set = h3_lora_set_open(
        &spec, 1, base, H3_LORA_FL2VA, error, sizeof(error));
    if (!set) die(error);
    const h3_lora *lora = NULL;
    if (!h3_lora_set_entry(set, 0, &lora, NULL, NULL)) die("missing adapter");
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) die(error);
    enum { CHUNK = 1 << 20 };
    uint16_t *actual = malloc(CHUNK * sizeof(*actual));
    uint16_t *expected = malloc(CHUNK * sizeof(*expected));
    if (!actual || !expected) die("host allocation failed");
    size_t peak_scratch = 0;
    for (size_t pair = 0; pair < h3_lora_pair_count(lora); pair++) {
        const h3_lora_pair_info *info = h3_lora_pair_at(lora, pair);
        uint64_t shape[] = {info->output_dim, info->input_dim};
        h3_gpu_tensor *got = h3_weight_load_bf16_with_loras(
            base, gpu, info->target, info->output_dim, info->input_dim,
            set, &peak_scratch, error, sizeof(error));
        h3_gpu_tensor *want = h3_weight_load_bf16(
            folded, gpu, info->target, 2, shape, error, sizeof(error));
        if (!got || !want) die(error);
        size_t elements = (size_t)info->output_dim * (size_t)info->input_dim;
        for (size_t offset = 0; offset < elements; offset += CHUNK) {
            size_t count = elements - offset;
            if (count > CHUNK) count = CHUNK;
            if (!h3_gpu_tensor_read_bf16_range(got, offset, actual, count) ||
                !h3_gpu_tensor_read_bf16_range(want, offset, expected, count))
                die("cannot read compared tensor");
            if (memcmp(actual, expected, count * sizeof(*actual))) {
                size_t mismatch = 0;
                while (mismatch < count && actual[mismatch] == expected[mismatch])
                    mismatch++;
                fprintf(stderr,
                        "mismatch: %s at element %zu: runtime=%04x folded=%04x\n",
                        info->target, offset + mismatch,
                        actual[mismatch], expected[mismatch]);
                return 1;
            }
        }
        h3_gpu_tensor_free(want);
        h3_gpu_tensor_free(got);
        printf("%zu/%zu %s\n", pair + 1, h3_lora_pair_count(lora),
               info->target);
        fflush(stdout);
    }
    printf("ok: %zu tensors byte-identical; peak scratch %.1f MiB\n",
           h3_lora_pair_count(lora),
           (double)peak_scratch / (1024.0 * 1024.0));
    free(expected); free(actual);
    h3_gpu_free(gpu);
    h3_lora_set_free(set);
    h3_weight_store_free(folded);
    h3_weight_store_free(base);
    return 0;
}
