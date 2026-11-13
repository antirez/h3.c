#include "h3_image_pipeline.h"
#include "h3_safetensors.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEXT_DIM = 5120 };
#define EMBEDDING_BOUND 1e-3

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_image_text.c: %s\n", message);
    exit(1);
}

static void *read_fixture(const h3_st_header *fixture, const char *name,
                          h3_dtype dtype, size_t elements) {
    char error[512];
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != elements) {
        fprintf(stderr, "FAIL tests/test_real_image_text.c: fixture tensor %s "
                "is missing or has the wrong type or size\n", name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes);
    if (!data || !h3_st_read_data(fixture, tensor, data, bytes, error,
                                  sizeof(error))) die(data ? error : "oom");
    return data;
}

static float bf16_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == total || completed % 10 == 0)
        fprintf(stderr, "  text encoder %d/%d\n", completed, total);
}

/* The tokenizer-only rules: no padding, a hard cut at 4096 tokens. */
static void check_tokenizer_rules(const char *model_directory) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/tokenizer/tokenizer.json",
             model_directory);
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) die(error);
    uint32_t *ids = NULL;
    size_t count = 0;
    if (h3_image_tokenize(tokenizer, "", &ids, &count, error, sizeof(error)))
        die("an empty prompt must be rejected");
    char *long_prompt = malloc(2 * 5000 + 1);
    if (!long_prompt) die("oom");
    for (size_t index = 0; index < 5000; index++)
        memcpy(long_prompt + 2 * index, "a ", 2);
    long_prompt[2 * 5000] = '\0';
    if (!h3_image_tokenize(tokenizer, long_prompt, &ids, &count, error,
                           sizeof(error))) die(error);
    if (count != H3_IMAGE_MAX_PROMPT_TOKENS)
        die("a long prompt must be cut to 4096 tokens");
    h3_tokenizer_ids_free(ids);
    free(long_prompt);
    h3_tokenizer_free(tokenizer);
    printf("tokenizer: empty prompt rejected, long prompt cut to %d\n",
           H3_IMAGE_MAX_PROMPT_TOKENS);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TEXT_FIXTURE\n", argv[0]);
        return 2;
    }
    char error[512];
    check_tokenizer_rules(argv[1]);

    h3_st_header fixture;
    if (!h3_st_read_header(argv[2], &fixture, error, sizeof(error)))
        die(error);
    const h3_st_tensor *prompt_tensor = h3_st_find(&fixture, "prompt");
    const h3_st_tensor *ids_tensor = h3_st_find(&fixture, "token_ids");
    if (!prompt_tensor || !ids_tensor) die("fixture has no prompt or token_ids");
    size_t prompt_bytes = h3_st_tensor_elements(prompt_tensor);
    size_t tokens = h3_st_tensor_elements(ids_tensor);
    char *prompt = read_fixture(&fixture, "prompt", H3_DTYPE_U8, prompt_bytes);
    prompt = realloc(prompt, prompt_bytes + 1);
    if (!prompt) die("oom");
    prompt[prompt_bytes] = '\0';
    int32_t *want_ids = read_fixture(&fixture, "token_ids", H3_DTYPE_I32,
                                     tokens);
    uint16_t *want = read_fixture(&fixture, "prompt_embeds", H3_DTYPE_BF16,
                                  tokens * TEXT_DIM);
    printf("prompt \"%s\": %zu tokens\n", prompt, tokens);

    uint32_t *ids = NULL;
    size_t count = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    if (!h3_image_encode_prompt(argv[1], prompt, progress, NULL, &ids, &count,
                                &text, error, sizeof(error))) die(error);
    if (count != tokens) die("token count differs from the reference");
    for (size_t index = 0; index < tokens; index++)
        if (ids[index] != (uint32_t)want_ids[index])
            die("token ids differ from the reference");
    printf("token ids: identical (%zu)\n", tokens);
    if (text.tokens != tokens || text.width != TEXT_DIM)
        die("embedding has the wrong shape");

    double square_error = 0.0, square_value = 0.0, maximum = 0.0;
    for (size_t index = 0; index < tokens * TEXT_DIM; index++) {
        double got = (double)bf16_value(text.values[index]);
        double expected = (double)bf16_value(want[index]);
        if (!isfinite(got)) die("non-finite embedding");
        double delta = got - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += expected * expected;
    }
    double relative = sqrt(square_error / square_value);
    printf("prompt_embeds vs reference: rel-L2 %.6g, max abs %.6g\n", relative,
           maximum);
    if (!(relative < EMBEDDING_BOUND)) die("prompt embeddings are off");
    h3_text_embedding_free(&text);
    h3_tokenizer_ids_free(ids);
    free(prompt);
    free(want_ids);
    free(want);
    printf("ok: image prompt encoding matches the diffusers reference\n");
    return 0;
}
