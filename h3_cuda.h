#ifndef H3_CUDA_H
#define H3_CUDA_H

#include "h3.h"

#ifdef __cplusplus
extern "C" {
#endif

int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
