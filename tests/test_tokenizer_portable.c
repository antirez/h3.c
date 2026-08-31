#include "h3_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int check(h3_tokenizer *tokenizer, const char *text,
                 const uint32_t *expected, size_t expected_count,
                 const char *decoded_expected) {
    char error[256];
    uint32_t *ids = NULL;
    size_t count = 0;
    CHECK(h3_tokenizer_encode(tokenizer, text, 0, &ids, &count,
                              error, sizeof(error)));
    CHECK(count == expected_count);
    CHECK(!memcmp(ids, expected, count * sizeof(*ids)));
    char *decoded = h3_tokenizer_decode(tokenizer, ids, count,
                                        error, sizeof(error));
    CHECK(decoded && !strcmp(decoded, decoded_expected));
    free(decoded); h3_tokenizer_ids_free(ids); return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
        "tests/tokenizer_portable_fixture.json";
    char error[256];
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, sizeof(error));
    CHECK(tokenizer != NULL);
    const uint32_t spaced[] = {1, 3};
    CHECK(!check(tokenizer, "A A", spaced, 2, "A A"));
    const uint32_t normalized[] = {4};
    CHECK(!check(tokenizer, "e\xcc\x81", normalized, 1, "\xc3\xa9"));
    const uint32_t added[] = {10, 1};
    CHECK(!check(tokenizer, "<special>A", added, 2, "<special>A"));
    uint32_t *ids = NULL; size_t count = 99;
    CHECK(h3_tokenizer_encode(tokenizer, "", 1, &ids, &count,
                              error, sizeof(error)));
    CHECK(count == 1 && ids[0] == H3_PAD_TOKEN_ID);
    h3_tokenizer_ids_free(ids); h3_tokenizer_free(tokenizer);
    puts("ok: portable ICU byte-level BPE tokenizer");
    return 0;
}
