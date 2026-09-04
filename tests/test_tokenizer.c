#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void check_case(h3_tokenizer *tokenizer, const char *text,
                       const uint32_t *expected, size_t expected_count) {
    char error[256];
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, text, 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == expected_count);
    for (size_t index = 0; index < count; index++) CHECK(ids[index] == expected[index]);
    char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
    CHECK(decoded != NULL);
    CHECK(!strcmp(decoded, text));
    free(decoded);
    h3_tokenizer_ids_free(ids);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "MiniMax-H3/tokenizer/tokenizer.json";
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "FAIL tests/test_tokenizer.c: %s\n", error);
        return 1;
    }
    const uint32_t fox[] = {32, 2518, 38835, 11435, 1526, 11794};
    check_case(tokenizer, "A red fox walking through snow", fox,
               sizeof(fox) / sizeof(fox[0]));
    const uint32_t unicode[] = {9707, 11, 50891, 0, 220, 220, 17, 15, 17,
                                21, 198, 104811, 51950, 594};
    check_case(tokenizer, "Hello, WORLD!  2026\n中文 café's", unicode,
               sizeof(unicode) / sizeof(unicode[0]));
    const uint32_t emoji[] = {145080, 64, 4894};
    check_case(tokenizer, "🙂a!\n", emoji,
               sizeof(emoji) / sizeof(emoji[0]));
    const uint32_t special[] = {151644};
    check_case(tokenizer, "<|im_start|>", special, 1);
    /* The seven H3 control tokens live in tokenizer_config.json rather than in
     * tokenizer.json, and the reference tokenizer appends them after the last
     * declared added token. */
    const uint32_t dialogue[] = {151669, 58, 22574, 60, 21927, 1052, 13, 151670};
    check_case(tokenizer, "<d>[English] Hello there.</d>", dialogue,
               sizeof(dialogue) / sizeof(dialogue[0]));
    const uint32_t cutoff[] = {151671};
    check_case(tokenizer, "<|cutoff|>", cutoff, 1);
    const uint32_t caption[] = {151674, 151675};
    check_case(tokenizer, "<|caption_start|><|caption_end|>", caption,
               sizeof(caption) / sizeof(caption[0]));
    const uint32_t cinematic[] = {32, 64665, 3265, 5239, 315, 264, 8866,
                                  1778, 11958, 4633, 10971, 13};
    check_case(tokenizer,
               "A cinematic close-up of a clockwork bird taking flight.",
               cinematic, sizeof(cinematic) / sizeof(cinematic[0]));

    uint32_t *ids = NULL;
    size_t count = 99;
    CHECK(h3_tokenizer_encode(tokenizer, "", 0, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 0 && ids == NULL);
    CHECK(h3_tokenizer_encode(tokenizer, "", 1, &ids, &count, error,
                              sizeof(error)));
    CHECK(count == 1 && ids[0] == H3_PAD_TOKEN_ID);
    h3_tokenizer_ids_free(ids);

    h3_tokenizer_free(tokenizer);
    printf("ok: %d tokenizer checks against released Qwen vocabulary\n", checks);
    return 0;
}
