#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_lora h3_lora;

typedef struct {
    const char *target;
    uint64_t output_dim;
    uint64_t input_dim;
    uint64_t rank;
    float alpha_over_rank;
} h3_lora_pair_info;

h3_lora *h3_lora_open(const char *path, const h3_weight_store *base,
                      char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);

const char *h3_lora_sha256(const h3_lora *lora);
const char *h3_lora_format(const h3_lora *lora);
size_t h3_lora_pair_count(const h3_lora *lora);
const h3_lora_pair_info *h3_lora_pair_at(const h3_lora *lora, size_t index);
int h3_lora_find_pair(const h3_lora *lora, const char *target,
                      size_t *index);

int h3_lora_read_a(const h3_lora *lora, size_t pair_index,
                   float *values, size_t elements,
                   char *error, size_t error_size);
int h3_lora_read_b_rows(const h3_lora *lora, size_t pair_index,
                        uint64_t first_row, uint64_t rows,
                        float *values, size_t elements,
                        char *error, size_t error_size);

#endif
