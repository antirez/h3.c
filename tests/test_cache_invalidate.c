/* Host-only test for h3_cache_invalidate_inputs (no GPU, no model, no Metal).
 *
 * Contract under test: invalidating input-bound caches drops the prompt/
 * reference conditioning and the prepared DiT while a Video VAE decoder with
 * an unchanged key stays resident; a full h3_cache_clear() still clears the
 * decoder too.
 */
#include "h3.h"
#include "h3_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int asserts;

#define CHECK(cond) do {                                                   \
        asserts++;                                                         \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            return 0;                                                      \
        }                                                                  \
    } while (0)

static int test_invalidate_keeps_decoder(void)
{
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    ctx->cache_enabled = 1;
    ctx->conditioning_key = strdup("cond");
    ctx->conditioning_values = malloc(8);
    ctx->conditioning_present = 1;
    ctx->dit_key = strdup("dit");
    ctx->video_decoder_key = strdup("vae|22x16");
    ctx->video_decoder = (struct h3_video_vae_decoder *)(uintptr_t)1;

    h3_cache_invalidate_inputs(ctx);
    CHECK(ctx->conditioning_key == NULL);
    CHECK(ctx->conditioning_values == NULL);
    CHECK(ctx->conditioning_present == 0);
    CHECK(ctx->dit == NULL);
    CHECK(ctx->dit_key == NULL);
    CHECK(ctx->video_decoder_key != NULL);
    CHECK(strcmp(ctx->video_decoder_key, "vae|22x16") == 0);
    CHECK(ctx->video_decoder ==
          (struct h3_video_vae_decoder *)(uintptr_t)1);

    ctx->video_decoder = NULL;
    free(ctx->video_decoder_key);
    free(ctx);
    return 1;
}

static int test_full_clear_still_drops_decoder(void)
{
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    ctx->cache_enabled = 1;
    ctx->video_decoder_key = strdup("vae|22x16");
    h3_cache_clear(ctx);
    CHECK(ctx->video_decoder == NULL);
    CHECK(ctx->video_decoder_key == NULL);
    CHECK(ctx->dit == NULL);
    CHECK(ctx->dit_key == NULL);
    free(ctx);
    return 1;
}

static int test_null_safety(void)
{
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    CHECK(ctx != NULL);
    h3_cache_invalidate_inputs(ctx);   /* all-NULL ctx fields: no-op */
    h3_cache_invalidate_inputs(NULL);  /* NULL ctx: no-op */
    free(ctx);
    return 1;
}

int main(void)
{
    int ok = test_invalidate_keeps_decoder() &&
             test_full_clear_still_drops_decoder() &&
             test_null_safety();
    printf("cache_invalidate: %d asserts, %s\n", asserts, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
