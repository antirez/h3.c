#include "h3_host.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

int main(void) {
    const uint8_t constant[] = {
        17,33,201, 17,33,201,
        17,33,201, 17,33,201
    };
    uint8_t *output = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 8, 8, &output));
    for (size_t pixel = 0; pixel < 64; pixel++)
        CHECK(!memcmp(output + pixel * 3, constant, 3));
    free(output);
    puts("ok: portable RGB resize");
    return 0;
}
