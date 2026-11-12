#include "h3_multimodal.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_VISION_START UINT32_C(151652)
#define H3_VISION_END UINT32_C(151653)
#define H3_IMAGE_PAD UINT32_C(151655)
#define H3_VIDEO_PAD UINT32_C(151656)

typedef struct {
    uint32_t *values;
    size_t count;
    size_t capacity;
} h3_ids;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int ids_reserve(h3_ids *ids, size_t extra) {
    if (extra > SIZE_MAX - ids->count) return 0;
    size_t needed = ids->count + extra;
    if (needed <= ids->capacity) return 1;
    size_t capacity = ids->capacity ? ids->capacity : 32;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*ids->values)) return 0;
    uint32_t *values = realloc(ids->values, capacity * sizeof(*values));
    if (!values) return 0;
    ids->values = values;
    ids->capacity = capacity;
    return 1;
}

static int ids_append(h3_ids *ids, const uint32_t *values, size_t count) {
    if (!ids_reserve(ids, count)) return 0;
    if (count) memcpy(ids->values + ids->count, values,
                      count * sizeof(*values));
    ids->count += count;
    return 1;
}

static int ids_push(h3_ids *ids, uint32_t value) {
    return ids_append(ids, &value, 1);
}

static int tokenize_append_limited(const h3_tokenizer *tokenizer,
                                   const char *text, size_t limit,
                                   h3_ids *ids, char *error,
                                   size_t error_size) {
    uint32_t *values = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, text, 0, &values, &count,
                             error, error_size)) return 0;
    int ok = ids_append(ids, values, count < limit ? count : limit);
    h3_tokenizer_ids_free(values);
    if (!ok) fail(error, error_size,
                  "out of memory constructing multimodal token sequence");
    return ok;
}

static int tokenize_append(const h3_tokenizer *tokenizer, const char *text,
                           h3_ids *ids, char *error, size_t error_size) {
    return tokenize_append_limited(tokenizer, text, SIZE_MAX, ids,
                                   error, error_size);
}

static int build_positions(const h3_vision_output *const *images,
                           const h3_text_vision_span *spans,
                           size_t image_count, size_t sequence,
                           uint32_t *positions,
                           char *error, size_t error_size) {
    int64_t offset = 0;
    for (size_t image = 0; image < image_count; image++) {
        const h3_vision_output *vision = images[image];
        size_t start = spans[image].start;
        size_t end = start + spans[image].tokens;
        if (!vision || !start || start > sequence || end > sequence ||
            vision->grid_h < 2 || vision->grid_w < 2 ||
            vision->grid_h % 2 || vision->grid_w % 2) {
            fail(error, error_size, "invalid Qwen vision span geometry");
            return 0;
        }
        if (image == 0) {
            for (size_t axis = 0; axis < 3; axis++)
                for (size_t index = 0; index < start; index++)
                    positions[axis * sequence + index] = (uint32_t)index;
        }
        size_t merged_h = (size_t)vision->grid_h / 2;
        size_t merged_w = (size_t)vision->grid_w / 2;
        size_t length_max = merged_h > merged_w ? merged_h : merged_w;
        int64_t base = (int64_t)start + offset;
        int64_t next = (int64_t)start + (int64_t)length_max + offset;
        if (base < 0 || next < 0 ||
            (uint64_t)next + (uint64_t)(sequence - end) > UINT32_MAX) {
            fail(error, error_size, "multimodal mRoPE position overflow");
            return 0;
        }
        for (size_t index = start; index < end; index++)
            positions[index] = (uint32_t)base;
        size_t cursor = start;
        for (size_t row = 0; row < merged_h; row++) {
            for (size_t column = 0; column < merged_w; column++) {
                positions[sequence + cursor] = (uint32_t)(base + (int64_t)row);
                positions[2 * sequence + cursor] =
                    (uint32_t)(base + (int64_t)column);
                cursor++;
            }
        }
        if (cursor != end) {
            fail(error, error_size, "Qwen merged grid does not match token span");
            return 0;
        }
        for (size_t axis = 0; axis < 3; axis++)
            for (size_t index = end; index < sequence; index++)
                positions[axis * sequence + index] =
                    (uint32_t)(next + (int64_t)(index - end));
        offset += (int64_t)length_max - (int64_t)spans[image].tokens;
    }
    return 1;
}

static int append_pad_span(h3_ids *ids, h3_text_vision_span *span,
                           size_t tokens, uint32_t pad) {
    if (!ids_push(ids, H3_VISION_START)) return 0;
    span->start = ids->count;
    span->tokens = tokens;
    if (!ids_reserve(ids, tokens + 1)) return 0;
    for (size_t token = 0; token < tokens; token++)
        ids->values[ids->count++] = pad;
    ids->values[ids->count++] = H3_VISION_END;
    return 1;
}

static int append_vision(h3_ids *ids, h3_text_vision_span *span,
                         const h3_vision_output *vision, uint32_t pad,
                         char *error, size_t error_size) {
    if (!vision || !vision->merged || !vision->deepstack[0] ||
        !vision->deepstack[1] || !vision->deepstack[2] || !vision->tokens ||
        vision->tokens > UINT32_MAX) {
        fail(error, error_size, "invalid Qwen vision output");
        return 0;
    }
    if (!append_pad_span(ids, span, vision->tokens, pad)) return 0;
    span->embeddings = vision->merged;
    for (size_t layer = 0; layer < H3_VISION_DEEPSTACKS; layer++)
        span->deepstack[layer] = vision->deepstack[layer];
    return 1;
}

/* Modality tags (0 on every vision span including its boundary tokens) and
 * axis-major [3, count] mRoPE positions of a finished id list. */
static int describe_presentation(const h3_ids *ids,
                                 const h3_text_vision_span *spans,
                                 const h3_vision_output *const *visions,
                                 size_t span_count, uint32_t *positions,
                                 uint8_t *tags, char *error,
                                 size_t error_size) {
    memset(tags, 1, ids->count);
    for (size_t image = 0; image < span_count; image++) {
        size_t first = spans[image].start - 1;
        size_t count = spans[image].tokens + 2;
        if (first > ids->count || count > ids->count - first) {
            fail(error, error_size, "multimodal tag span overflow");
            return 0;
        }
        memset(tags + first, 0, count);
    }
    return build_positions(visions, spans, span_count, ids->count, positions,
                           error, error_size);
}

static int encode_presentation(const char *weight_directory,
                               const char *shader_source_path,
                               h3_ids *ids, h3_text_vision_span *spans,
                               const h3_vision_output *const *visions,
                               size_t span_count,
                               h3_text_progress progress,
                               void *progress_opaque,
                               h3_text_embedding *output,
                               char *error, size_t error_size) {
    if (!ids->count || ids->count > INT_MAX || ids->count > SIZE_MAX / 3 ||
        ids->count > SIZE_MAX / sizeof(uint32_t)) {
        fail(error, error_size, "multimodal presentation is too large");
        return 0;
    }
    uint32_t *positions = calloc(3 * ids->count, sizeof(*positions));
    uint8_t *tags = malloc(ids->count);
    if (!positions || !tags) {
        free(positions); free(tags);
        fail(error, error_size,
             "out of memory finalizing multimodal presentation");
        return 0;
    }
    if (!describe_presentation(ids, spans, visions, span_count, positions,
                               tags, error, error_size)) {
        free(positions); free(tags);
        return 0;
    }
    int ok = h3_text_encode_multimodal_bf16(
        weight_directory, shader_source_path, ids->values, ids->count,
        spans, span_count, positions, tags, progress, progress_opaque,
        output, error, error_size);
    free(positions); free(tags);
    return ok;
}

int h3_multimodal_encode_fl2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!tokenizer || !weight_directory || !shader_source_path || !prompt ||
        !*prompt || !images || !image_count || !output) {
        fail(error, error_size, "invalid FL2VA multimodal presentation arguments");
        return 0;
    }
    h3_ids ids = {0};
    h3_text_vision_span *spans = calloc(image_count, sizeof(*spans));
    const h3_vision_output **visions = calloc(image_count, sizeof(*visions));
    int ok = 0;
    if (!spans || !visions) goto oom;
    for (size_t image = 0; image < image_count; image++) {
        const h3_vision_output *vision = &images[image];
        char prefix[64];
        int length = snprintf(prefix, sizeof(prefix), "<Picture %zu>: ", image + 1);
        visions[image] = vision;
        if (length < 0 || (size_t)length >= sizeof(prefix) ||
            !tokenize_append(tokenizer, prefix, &ids, error, error_size) ||
            !append_vision(&ids, &spans[image], vision, H3_IMAGE_PAD,
                           error, error_size)) goto cleanup;
    }
    if (!tokenize_append(tokenizer, prompt, &ids, error, error_size)) goto cleanup;
    ok = encode_presentation(weight_directory, shader_source_path,
        &ids, spans, visions, image_count, progress, progress_opaque,
        output, error, error_size);
    goto cleanup;

oom:
    fail(error, error_size, "out of memory constructing FL2VA presentation");
cleanup:
    free(ids.values);
    free(spans);
    free(visions);
    return ok;
}

int h3_multimodal_encode_ref2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_reference_presentation *references,
                        size_t reference_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!tokenizer || !weight_directory || !shader_source_path || !prompt ||
        !*prompt || !references || !reference_count || !output) {
        fail(error, error_size, "invalid Ref2VA presentation arguments");
        return 0;
    }
    size_t span_count = 0;
    int have_visual = 0;
    for (size_t index = 0; index < reference_count; index++) {
        const h3_reference_presentation *reference = &references[index];
        if (reference->kind == H3_PRESENTATION_IMAGE) {
            if (reference->has_audio || reference->vision_count != 1 ||
                !reference->vision) goto invalid;
            span_count++;
            have_visual = 1;
        } else if (reference->kind == H3_PRESENTATION_VIDEO) {
            if (!reference->vision || !reference->vision_count ||
                !reference->timestamps ||
                span_count > SIZE_MAX - reference->vision_count) goto invalid;
            span_count += reference->vision_count;
            have_visual = 1;
        } else if (reference->kind == H3_PRESENTATION_AUDIO) {
            if (!reference->has_audio || reference->vision ||
                reference->vision_count) goto invalid;
        } else goto invalid;
    }
    if (!have_visual || !span_count) {
        fail(error, error_size,
             "Ref2VA audio requires an image or video reference");
        return 0;
    }
    h3_ids ids = {0};
    h3_text_vision_span *spans = calloc(span_count, sizeof(*spans));
    const h3_vision_output **visions = calloc(span_count, sizeof(*visions));
    if (!spans || !visions) goto oom;
    size_t span_index = 0;
    size_t image_ordinal = 0, video_ordinal = 0, audio_ordinal = 0;
    for (size_t index = 0; index < reference_count; index++) {
        const h3_reference_presentation *reference = &references[index];
        char prefix[96];
        if (reference->has_audio) {
            int length = snprintf(prefix, sizeof(prefix), "<Audio %zu>: ",
                                  ++audio_ordinal);
            if (length < 0 || (size_t)length >= sizeof(prefix) ||
                !tokenize_append(tokenizer, prefix, &ids,
                                 error, error_size)) goto cleanup;
        }
        if (reference->kind == H3_PRESENTATION_AUDIO) continue;
        if (reference->kind == H3_PRESENTATION_IMAGE) {
            int length = snprintf(prefix, sizeof(prefix), "<Picture %zu>: ",
                                  ++image_ordinal);
            visions[span_index] = reference->vision;
            if (length < 0 || (size_t)length >= sizeof(prefix) ||
                !tokenize_append(tokenizer, prefix, &ids,
                                 error, error_size) ||
                !append_vision(&ids, &spans[span_index], reference->vision,
                               H3_IMAGE_PAD, error, error_size)) goto cleanup;
            span_index++;
            continue;
        }
        video_ordinal++;
        for (size_t block = 0; block < reference->vision_count; block++) {
            if (!isfinite(reference->timestamps[block]) ||
                reference->timestamps[block] < 0.0) goto invalid_cleanup;
            if (block == 0) {
                int length = snprintf(prefix, sizeof(prefix), "<Video %zu>: ",
                                      video_ordinal);
                if (length < 0 || (size_t)length >= sizeof(prefix) ||
                    !tokenize_append(tokenizer, prefix, &ids,
                                     error, error_size)) goto cleanup;
            }
            int length = snprintf(prefix, sizeof(prefix), "<%.1f seconds>",
                                  reference->timestamps[block]);
            visions[span_index] = &reference->vision[block];
            if (length < 0 || (size_t)length >= sizeof(prefix) ||
                !tokenize_append(tokenizer, prefix, &ids,
                                 error, error_size) ||
                !append_vision(&ids, &spans[span_index],
                               &reference->vision[block], H3_VIDEO_PAD,
                               error, error_size)) goto cleanup;
            span_index++;
        }
    }
    if (span_index != span_count ||
        !tokenize_append(tokenizer, prompt, &ids, error, error_size))
        goto cleanup;
    {
        int ok = encode_presentation(weight_directory, shader_source_path,
            &ids, spans, visions, span_count, progress, progress_opaque,
            output, error, error_size);
        free(ids.values); free(spans); free(visions);
        return ok;
    }

invalid:
    fail(error, error_size, "invalid Ref2VA reference presentation");
    return 0;
invalid_cleanup:
    fail(error, error_size, "invalid Ref2VA video timestamp");
cleanup:
    free(ids.values); free(spans); free(visions);
    return 0;
oom:
    fail(error, error_size, "out of memory constructing Ref2VA presentation");
    free(spans); free(visions);
    return 0;
}

static int build_edit_ids(const h3_tokenizer *tokenizer, const char *prompt,
                          const h3_vision_output *const *visions,
                          size_t count, int attach, h3_ids *ids,
                          h3_text_vision_span *spans,
                          char *error, size_t error_size) {
    for (size_t image = 0; image < count; image++) {
        char label[48];
        int length = snprintf(label, sizeof(label), "Picture %zu: ", image + 1);
        if (length < 0 || (size_t)length >= sizeof(label) ||
            !tokenize_append_limited(tokenizer, label, H3_EDIT_PIECE_TOKENS,
                                     ids, error, error_size)) return 0;
        int ok = attach ?
            append_vision(ids, &spans[image], visions[image], H3_IMAGE_PAD,
                          error, error_size) :
            append_pad_span(ids, &spans[image], visions[image]->tokens,
                            H3_IMAGE_PAD);
        if (!ok) {
            if (!attach) fail(error, error_size,
                              "out of memory constructing edit presentation");
            return 0;
        }
    }
    return tokenize_append_limited(tokenizer, prompt, H3_EDIT_PIECE_TOKENS,
                                   ids, error, error_size);
}

int h3_multimodal_edit_presentation(const h3_tokenizer *tokenizer,
                                    const char *prompt, const int *grid_h,
                                    const int *grid_w, size_t image_count,
                                    h3_edit_presentation *presentation,
                                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (presentation) memset(presentation, 0, sizeof(*presentation));
    if (!tokenizer || !prompt || !*prompt || !grid_h || !grid_w ||
        !image_count || !presentation) {
        fail(error, error_size, "invalid edit presentation arguments");
        return 0;
    }
    h3_ids ids = {0};
    h3_vision_output *stubs = calloc(image_count, sizeof(*stubs));
    const h3_vision_output **visions = calloc(image_count, sizeof(*visions));
    h3_text_vision_span *spans = calloc(image_count, sizeof(*spans));
    uint32_t *positions = NULL;
    uint8_t *tags = NULL, *types = NULL;
    int ok = 0;
    if (!stubs || !visions || !spans) goto oom;
    for (size_t image = 0; image < image_count; image++) {
        if (grid_h[image] < 2 || grid_w[image] < 2 || grid_h[image] % 2 ||
            grid_w[image] % 2 ||
            (int64_t)grid_h[image] * grid_w[image] / 4 > INT32_MAX) {
            fail(error, error_size, "invalid Qwen vision grid for reference %zu",
                 image + 1);
            goto cleanup;
        }
        stubs[image].grid_h = grid_h[image];
        stubs[image].grid_w = grid_w[image];
        stubs[image].tokens = (size_t)grid_h[image] * (size_t)grid_w[image] / 4;
        visions[image] = &stubs[image];
    }
    if (!build_edit_ids(tokenizer, prompt, visions, image_count, 0, &ids,
                        spans, error, error_size)) goto cleanup;
    if (ids.count > INT_MAX) {
        fail(error, error_size, "edit presentation is too large");
        goto cleanup;
    }
    positions = calloc(3 * ids.count, sizeof(*positions));
    tags = malloc(ids.count);
    types = calloc(ids.count, 1);
    if (!positions || !tags || !types) goto oom;
    if (!describe_presentation(&ids, spans, visions, image_count, positions,
                               tags, error, error_size)) goto cleanup;
    for (size_t image = 0; image < image_count; image++)
        memset(types + spans[image].start, 1, spans[image].tokens);
    presentation->ids = ids.values;
    presentation->count = ids.count;
    presentation->tags = tags;
    presentation->mm_token_type_ids = types;
    presentation->positions = positions;
    ids.values = NULL; tags = NULL; types = NULL; positions = NULL;
    ok = 1;
    goto cleanup;

oom:
    fail(error, error_size, "out of memory constructing edit presentation");
cleanup:
    free(ids.values); free(stubs); free(visions); free(spans);
    free(positions); free(tags); free(types);
    return ok;
}

void h3_edit_presentation_free(h3_edit_presentation *presentation) {
    if (!presentation) return;
    free(presentation->ids);
    free(presentation->tags);
    free(presentation->mm_token_type_ids);
    free(presentation->positions);
    memset(presentation, 0, sizeof(*presentation));
}

int h3_multimodal_encode_edit_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (output) memset(output, 0, sizeof(*output));
    if (!tokenizer || !weight_directory || !shader_source_path || !prompt ||
        !*prompt || !images || !image_count || !output) {
        fail(error, error_size, "invalid edit presentation arguments");
        return 0;
    }
    h3_ids ids = {0};
    h3_text_vision_span *spans = calloc(image_count, sizeof(*spans));
    const h3_vision_output **visions = calloc(image_count, sizeof(*visions));
    int ok = 0;
    if (!spans || !visions) {
        fail(error, error_size, "out of memory constructing edit presentation");
        goto cleanup;
    }
    for (size_t image = 0; image < image_count; image++)
        visions[image] = &images[image];
    if (!build_edit_ids(tokenizer, prompt, visions, image_count, 1, &ids,
                        spans, error, error_size)) goto cleanup;
    ok = encode_presentation(weight_directory, shader_source_path, &ids,
                             spans, visions, image_count, progress,
                             progress_opaque, output, error, error_size);
cleanup:
    free(ids.values); free(spans); free(visions);
    return ok;
}
