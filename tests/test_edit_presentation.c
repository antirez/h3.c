#include "h3_multimodal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))

static void check_ids(const h3_tokenizer *tokenizer, const char *text,
                      const uint32_t *expected, size_t count) {
    char error[256];
    uint32_t *ids = NULL;
    size_t got = 0;
    CHECK(h3_tokenizer_encode(tokenizer, text, 0, &ids, &got, error,
                              sizeof(error)));
    CHECK(got == count);
    for (size_t index = 0; index < count; index++) CHECK(ids[index] == expected[index]);
    h3_tokenizer_ids_free(ids);
}

static void check_presentation(const h3_edit_presentation *p, size_t count,
                               const uint32_t *ids, const uint8_t *tags,
                               const uint8_t *types, const uint32_t *positions) {
    CHECK(p->count == count);
    CHECK(memcmp(p->ids, ids, count * sizeof(*ids)) == 0);
    CHECK(memcmp(p->tags, tags, count) == 0);
    CHECK(memcmp(p->mm_token_type_ids, types, count) == 0);
    CHECK(memcmp(p->positions, positions, 3 * count * sizeof(*positions)) == 0);
}

/* Reference values: HF tokenizer ids and Qwen3-VL get_rope_index on the same
 * id list, for grids (8, 8) and (4, 8) (16 and 8 merged tokens). */
static void test_presentation(const h3_tokenizer *tokenizer) {
    char error[256];
    const uint32_t picture_1[] = {24669, 220, 16, 25, 220};
    const uint32_t picture_12[] = {24669, 220, 16, 17, 25, 220};
    check_ids(tokenizer, "Picture 1: ", picture_1, COUNT(picture_1));
    check_ids(tokenizer, "Picture 12: ", picture_12, COUNT(picture_12));

    const uint32_t one_ids[] = {
        24669, 220, 16, 25, 220, 151652,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151653, 21604, 3826, 220, 16, 29, 279, 1852, 38835, 11, 1431, 304,
        41674, 10901};
    const uint8_t one_tags[] = {
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const uint8_t one_types[] = {
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t one_positions[] = {
        0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    h3_edit_presentation presentation;
    const int one_h[] = {8}, one_w[] = {8};
    CHECK(h3_multimodal_edit_presentation(
        tokenizer, "<Picture 1> the same fox, now in autumn leaves", one_h,
        one_w, 1, &presentation, error, sizeof(error)));
    check_presentation(&presentation, COUNT(one_ids), one_ids, one_tags,
                       one_types, one_positions);
    h3_edit_presentation_free(&presentation);
    CHECK(presentation.ids == NULL && presentation.count == 0);

    const uint32_t two_ids[] = {
        24669, 220, 16, 25, 220, 151652,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151653, 24669, 220, 17, 25, 220, 151652,
        151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
        151653, 46179, 366, 24669, 220, 16, 29, 323, 366, 24669, 220, 17,
        14276};
    const uint8_t two_tags[] = {
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const uint8_t two_types[] = {
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t two_positions[] = {
        0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 10,
        11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 18, 18, 18, 18, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
    const int two_h[] = {8, 4}, two_w[] = {8, 8};
    CHECK(h3_multimodal_edit_presentation(
        tokenizer, "Swap <Picture 1> and <Picture 2>.", two_h, two_w, 2,
        &presentation, error, sizeof(error)));
    check_presentation(&presentation, COUNT(two_ids), two_ids, two_tags,
                       two_types, two_positions);
    h3_edit_presentation_free(&presentation);

    /* A 5000-token prompt keeps 4096 tokens; the label keeps all of its own.
     * "word" is token 1158 and " word" token 3409. */
    size_t words = 5000;
    char *prompt = malloc(words * 5 + 1);
    CHECK(prompt != NULL);
    prompt[0] = '\0';
    for (size_t index = 0; index < words; index++)
        strcat(prompt, index ? " word" : "word");
    CHECK(h3_multimodal_edit_presentation(tokenizer, prompt, one_h, one_w, 1,
                                          &presentation, error,
                                          sizeof(error)));
    free(prompt);
    size_t head = 5 + 1 + 16 + 1;
    CHECK(presentation.count == head + 4096);
    CHECK(memcmp(presentation.ids, picture_1, sizeof(picture_1)) == 0);
    CHECK(presentation.ids[head] == 1158);
    for (size_t index = 1; index < 4096; index++)
        CHECK(presentation.ids[head + index] == 3409);
    CHECK(presentation.tags[presentation.count - 1] == 1);
    CHECK(presentation.positions[presentation.count - 1] == 10 + 4096);
    CHECK(presentation.positions[2 * presentation.count - 1] == 10 + 4096);
    CHECK(presentation.positions[3 * presentation.count - 1] == 10 + 4096);
    h3_edit_presentation_free(&presentation);

    CHECK(!h3_multimodal_edit_presentation(tokenizer, "", one_h, one_w, 1,
                                           &presentation, error, sizeof(error)));
    CHECK(!h3_multimodal_edit_presentation(tokenizer, "x", one_h, one_w, 0,
                                           &presentation, error, sizeof(error)));
    const int odd_h[] = {8, 7};
    CHECK(!h3_multimodal_edit_presentation(tokenizer, "x", odd_h, two_w, 2,
                                           &presentation, error, sizeof(error)));
    CHECK(strstr(error, "reference 2") != NULL);
    const int tiny_h[] = {0};
    CHECK(!h3_multimodal_edit_presentation(tokenizer, "x", tiny_h, one_w, 1,
                                           &presentation, error, sizeof(error)));
    CHECK(!h3_multimodal_edit_presentation(NULL, "x", one_h, one_w, 1,
                                           &presentation, error, sizeof(error)));
}

static void test_encode_arguments(const h3_tokenizer *tokenizer) {
    char error[256];
    static uint16_t rows[16], deepstack[3][16];
    h3_vision_output vision = {8, 8, 16, rows,
                               {deepstack[0], deepstack[1], deepstack[2]}, {0}};
    h3_text_embedding output;
    const char *missing = "/nonexistent-h3-weights";
    CHECK(!h3_multimodal_encode_edit_bf16(NULL, missing, "s", "p", &vision, 1,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "", &vision, 1,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "p", NULL, 1,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "p", &vision, 0,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "p", &vision, 1,
                                          NULL, NULL, NULL, error, sizeof(error)));
    h3_vision_output empty = vision;
    empty.merged = NULL;
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "p", &empty, 1,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(strstr(error, "invalid Qwen vision output") != NULL);
    h3_vision_output wrong = vision;
    wrong.tokens = 15;
    CHECK(!h3_multimodal_encode_edit_bf16(tokenizer, missing, "s", "p", &wrong, 1,
                                          NULL, NULL, &output, error, sizeof(error)));
    CHECK(strstr(error, "does not match") != NULL);
    CHECK(output.values == NULL && output.tokens == 0);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "MiniMax-H3-Image/tokenizer/tokenizer.json";
    if (access(path, R_OK) != 0) {
        printf("skip: tokenizer %s is not installed\n", path);
        return 0;
    }
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    if (!tokenizer) {
        fprintf(stderr, "FAIL tests/test_edit_presentation.c: %s\n", error);
        return 1;
    }
    test_presentation(tokenizer);
    test_encode_arguments(tokenizer);
    h3_tokenizer_free(tokenizer);
    printf("ok: %d edit presentation checks\n", checks);
    return 0;
}
