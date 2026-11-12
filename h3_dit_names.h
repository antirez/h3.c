#ifndef H3_DIT_NAMES_H
#define H3_DIT_NAMES_H

#include <stddef.h>

typedef enum {
    H3_DIT_FAMILY_VIDEO = 0,
    H3_DIT_FAMILY_IMAGE
} h3_dit_family;

/* Tensor names of one checkpoint family. Patterns containing %u take a block
 * or head index; the others are literal. Block members are relative to
 * block_prefix or refiner_prefix. A NULL member means the family has no such
 * tensor: qkv is NULL when q, k and v are stored as three tensors, and the
 * rope and audio members are NULL for the image family. */
typedef struct {
    h3_dit_family family;
    const char *block_prefix;
    const char *refiner_prefix;
    const char *norm1, *norm2;
    const char *qkv;
    const char *q, *k, *v;
    const char *q_norm, *k_norm;
    const char *out;
    const char *fc1, *fc2;
    const char *adaln_weight, *adaln_bias;
    const char *refiner_final_norm;
    const char *condition_weight, *condition_bias;
    const char *time_in_weight, *time_in_bias;
    const char *time_out_weight, *time_out_bias;
    const char *patch_weight, *patch_bias;
    const char *audio_patch_weight, *audio_patch_bias;
    const char *final_norm;
    const char *final_adaln_weight, *final_adaln_bias;
    const char *head_weight, *head_bias;
    const char *audio_head_weight, *audio_head_bias;
    const char *rope_inv_freq;
} h3_dit_names;

const h3_dit_names *h3_dit_names_for(h3_dit_family family);

/* Format a name pattern with an index; patterns without %u ignore it. Returns
 * 0 when the pattern is NULL or the result does not fit. */
int h3_dit_name_format(char *out, size_t size, const char *pattern,
                       unsigned index);

/* Format block_prefix + suffix for block `block` (refiner_prefix when
 * refiner is non-zero). */
int h3_dit_block_name(char *out, size_t size, const h3_dit_names *names,
                      int refiner, unsigned block, const char *suffix);

#endif
