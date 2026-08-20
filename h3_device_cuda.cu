#include "h3_device.h"

#include <cuda_runtime_api.h>

#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>

static void h3_cuda_error(char *error, size_t error_size, const char *operation,
                          cudaError_t status) {
    if (error && error_size) {
        snprintf(error, error_size, "%s: %s", operation,
                 cudaGetErrorString(status));
    }
}

extern "C" int h3_device_probe(h3_device_info *info, char *error,
                                size_t error_size) {
    if (!info) {
        if (error && error_size) snprintf(error, error_size, "device info is required");
        return 0;
    }
    memset(info, 0, sizeof(*info));

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        h3_cuda_error(error, error_size, "cannot select CUDA device", status);
        return 0;
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        h3_cuda_error(error, error_size, "cannot inspect CUDA device", status);
        return 0;
    }

    snprintf(info->name, sizeof(info->name), "%.127s", properties.name);
    snprintf(info->architecture, sizeof(info->architecture), "CUDA sm_%d%d",
             properties.major, properties.minor);
    struct sysinfo system;
    if (sysinfo(&system) == 0) {
        info->physical_memory =
            (uint64_t)system.totalram * (uint64_t)system.mem_unit;
    }
    info->recommended_working_set = (uint64_t)properties.totalGlobalMem;
    info->max_buffer_length = (uint64_t)properties.totalGlobalMem;
    info->unified_memory = properties.unifiedAddressing ? 1 : 0;
    return 1;
}
