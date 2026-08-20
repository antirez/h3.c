#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu != NULL);
    CHECK(!h3_gpu_is_m5(gpu));
    CHECK(!h3_gpu_has_nax_mlp(gpu));

    const float input[] = {1.0f, -2.5f, 0.125f, 65504.0f};
    h3_gpu_tensor *f32 = h3_gpu_tensor_from_f32(gpu, input, 4);
    h3_gpu_tensor *copy = h3_gpu_tensor_new_f32(gpu, 6);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *bf16_copy = h3_gpu_tensor_new_bf16(gpu, 6);
    CHECK(f32 && copy && bf16 && bf16_copy);
    CHECK(h3_gpu_tensor_elements(f32) == 4);
    CHECK(h3_gpu_tensor_dtype(f32) == H3_GPU_F32);
    CHECK(h3_gpu_tensor_write_f32(bf16, input, 4));

    float roundtrip[4] = {0};
    CHECK(h3_gpu_tensor_read_f32(bf16, roundtrip, 4));
    for (size_t index = 0; index < 4; index++)
        CHECK(fabsf(roundtrip[index] - input[index]) <=
              fmaxf(0.01f, fabsf(input[index]) * 0.008f));

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_copy_f32(gpu, copy, 1, f32, 0, 4));
    CHECK(h3_gpu_continue(gpu));
    CHECK(h3_gpu_submit(gpu));
    float copied[4] = {0};
    CHECK(h3_gpu_tensor_read_f32_range(copy, 1, copied, 4));
    CHECK(memcmp(input, copied, sizeof(input)) == 0);
    CHECK(!h3_gpu_tensor_read_f32_range(copy, 5, copied, 4));
    CHECK(strstr(h3_gpu_error(gpu), "range") != NULL);

    const uint16_t range_values[] = {0x3f80, 0xc020, 0x3e00, 0x4780};
    CHECK(h3_gpu_tensor_write_bf16_range(bf16_copy, 1, range_values, 4));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_copy_bf16(gpu, bf16, 0, bf16_copy, 1, 4));
    CHECK(h3_gpu_submit(gpu));
    uint16_t range_roundtrip[4] = {0};
    CHECK(h3_gpu_tensor_read_bf16(bf16, range_roundtrip, 4));
    CHECK(memcmp(range_values, range_roundtrip, sizeof(range_values)) == 0);

    char path[] = "/tmp/h3-cuda-runtime-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    const uint16_t file_values[] = {0x3f80, 0xc020, 0x3e00, 0x4780};
    CHECK(write(fd, file_values, sizeof(file_values)) ==
          (ssize_t)sizeof(file_values));
    const float file_f32[] = {3.5f, -7.0f};
    CHECK(write(fd, file_f32, sizeof(file_f32)) == (ssize_t)sizeof(file_f32));
    CHECK(close(fd) == 0);
    h3_gpu_tensor *loaded = h3_gpu_tensor_load_bf16(gpu, path, 0, 4);
    CHECK(loaded != NULL);
    uint16_t loaded_values[4] = {0};
    CHECK(h3_gpu_tensor_read_bf16(loaded, loaded_values, 4));
    CHECK(memcmp(file_values, loaded_values, sizeof(file_values)) == 0);
    h3_gpu_tensor *loaded_f32 = h3_gpu_tensor_load_f32(
        gpu, path, sizeof(file_values), 2);
    CHECK(loaded_f32 != NULL);
    float loaded_f32_values[2] = {0};
    CHECK(h3_gpu_tensor_read_f32(loaded_f32, loaded_f32_values, 2));
    CHECK(memcmp(file_f32, loaded_f32_values, sizeof(file_f32)) == 0);
    CHECK(h3_gpu_tensor_stream_file_bf16(bf16, path, 0, 4,
                                         error, sizeof(error)));
    CHECK(!h3_gpu_tensor_stream_file_bf16(
        bf16, path, (uint64_t)INT64_MAX, 4, error, sizeof(error)));
    CHECK(strstr(error, "overflows") != NULL);
    CHECK(unlink(path) == 0);

    h3_gpu_stats stats;
    CHECK(h3_gpu_get_stats(gpu, &stats));
    CHECK(stats.tensor_allocations == 6);
    CHECK(stats.live_bytes > 0 && stats.peak_live_bytes >= stats.live_bytes);
    CHECK(stats.blit_copies == 2 && stats.submissions == 3);
    CHECK(stats.gpu_seconds >= 0.0 && stats.command_encode_seconds >= 0.0);

    h3_gpu_tensor_free(loaded_f32);
    h3_gpu_tensor_free(loaded);
    h3_gpu_tensor_free(bf16_copy);
    h3_gpu_tensor_free(bf16);
    h3_gpu_tensor_free(copy);
    h3_gpu_tensor_free(f32);
    CHECK(h3_gpu_get_stats(gpu, &stats));
    CHECK(stats.live_bytes == 0);
    CHECK(setenv("H3_PROFILE", "1", 1) == 0);
    h3_gpu_profile_set_label(gpu, "runtime test");
    h3_gpu_profile_mark(gpu, "complete");
    h3_gpu_free(gpu);
    puts("ok: CUDA runtime allocation, conversion, copy and file I/O");
    return 0;
}
