#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_lora h3_lora;
typedef struct h3_lora_set h3_lora_set;
typedef void (*h3_lora_progress)(size_t completed_rows, size_t total_rows,
                                 void *opaque);

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

h3_lora_set *h3_lora_set_open(const h3_lora_spec *specs, size_t count,
                              const h3_weight_store *base,
                              h3_lora_scope active_scope,
                              char *error, size_t error_size);
void h3_lora_set_free(h3_lora_set *set);
size_t h3_lora_set_count(const h3_lora_set *set);
const char *h3_lora_set_signature(const h3_lora_set *set);
int h3_lora_set_entry(const h3_lora_set *set, size_t index,
                      const h3_lora **lora, float *scale,
                      h3_lora_scope *scope);
int h3_lora_set_has_target(const h3_lora_set *set, const char *target);
void h3_lora_set_progress(const h3_lora_set *set,
                          h3_lora_progress progress, void *opaque);
void h3_lora_set_advance(const h3_lora_set *set, size_t rows);
void h3_lora_set_finish(const h3_lora_set *set);

#endif
