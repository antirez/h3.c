#ifndef H3_MULTIMODAL_H
#define H3_MULTIMODAL_H

#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <stddef.h>

/* Construct the released `<Picture n>` presentation around already encoded
 * Qwen vision outputs, then run the language decoder. FL2VA anchors and
 * image-only Ref2VA entries intentionally share this exact presentation.
 * Keeping the builder separate lets vision and text release their large
 * temporary weights independently. */
int h3_multimodal_encode_fl2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

typedef enum {
    H3_PRESENTATION_IMAGE = 1,
    H3_PRESENTATION_VIDEO = 2,
    H3_PRESENTATION_AUDIO = 3
} h3_presentation_kind;

/* A video owns one Qwen vision output per sampled two-frame block and one
 * timestamp per block. Images own exactly one output; audio owns none. */
typedef struct {
    h3_presentation_kind kind;
    int has_audio;
    const h3_vision_output *vision;
    size_t vision_count;
    const double *timestamps;
} h3_reference_presentation;

/* Construct Ref2VA labels in exact request order, including pending audio
 * labels and per-block video timestamps, then run the Qwen decoder. */
int h3_multimodal_encode_ref2va_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_reference_presentation *references,
                        size_t reference_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

/* Reference-image edit presentation (MiniMax-H3-Image): for each reference a
 * "Picture i: " label (no angle brackets), <|vision_start|>, one
 * <|image_pad|> per merged vision token, <|vision_end|>, then the prompt.
 * Every label and the prompt are truncated to H3_EDIT_PIECE_TOKENS tokens
 * separately; the whole sequence is not. */
#define H3_EDIT_PIECE_TOKENS 4096

typedef struct {
    uint32_t *ids;
    size_t count;
    uint8_t *tags;              /* DiT modality tag: 1 text, 0 on vision spans */
    uint8_t *mm_token_type_ids; /* 1 on <|image_pad|> only */
    uint32_t *positions;        /* mRoPE, axis-major [3, count] */
} h3_edit_presentation;

/* Pure host function: grid_h/grid_w are the Qwen patch grids of the vision
 * copies (even; tokens = grid_h * grid_w / 4). Release with
 * h3_edit_presentation_free(). */
int h3_multimodal_edit_presentation(const h3_tokenizer *tokenizer,
                                    const char *prompt, const int *grid_h,
                                    const int *grid_w, size_t image_count,
                                    h3_edit_presentation *presentation,
                                    char *error, size_t error_size);
void h3_edit_presentation_free(h3_edit_presentation *presentation);

/* The same presentation around already encoded Qwen vision outputs (one per
 * reference, in order), then the Qwen decoder. */
int h3_multimodal_encode_edit_bf16(
                        const h3_tokenizer *tokenizer,
                        const char *weight_directory,
                        const char *shader_source_path,
                        const char *prompt,
                        const h3_vision_output *images, size_t image_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

#endif
