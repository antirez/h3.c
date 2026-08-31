#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static const h3_st_tensor *require_tensor(const h3_st_header *header,
        const char *name, h3_dtype dtype, int dimensions,
        const uint64_t *shape) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != dtype || tensor->ndim != dimensions)
        return NULL;
    for (int dimension = 0; dimension < dimensions; dimension++)
        if (tensor->shape[dimension] != shape[dimension]) return NULL;
    return tensor;
}

static int check_partition(const char *root, const char *partition, h3_gpu *gpu) {
    char audio_path[512];
    char audio_directory[512];
    char video_path[512];
    CHECK(snprintf(audio_path, sizeof(audio_path), "%s/%s/audio_vae/model.safetensors",
                   root, partition) > 0);
    CHECK(snprintf(audio_directory, sizeof(audio_directory), "%s/%s/audio_vae",
                   root, partition) > 0);
    CHECK(snprintf(video_path, sizeof(video_path),
                   "%s/%s/video_vae/source/model.safetensors",
                   root, partition) > 0);
    char error[512];
    h3_st_header audio;
    h3_st_header video;
    CHECK(h3_st_read_header(audio_path, &audio, error, sizeof(error)));
    CHECK(h3_st_read_header(video_path, &video, error, sizeof(error)));

    const uint64_t audio_bias_shape[] = {2048};
    const uint64_t filter_shape[] = {1, 1, 12};
    const uint64_t video_post_shape[] = {24, 24, 1, 1, 1};
    const uint64_t video_norm_shape[] = {2048};
    const h3_st_tensor *audio_bias = require_tensor(
        &audio, "dec_in_proj.bias", H3_DTYPE_F32, 1, audio_bias_shape);
    const h3_st_tensor *filter = require_tensor(
        &audio, "decoder.activation_post.downsample.lowpass.filter",
        H3_DTYPE_F32, 3, filter_shape);
    const h3_st_tensor *video_post = require_tensor(
        &video, "post_quant_conv.weight", H3_DTYPE_F32, 5,
        video_post_shape);
    const h3_st_tensor *video_norm = require_tensor(
        &video, "decoder.norm_out.weight", H3_DTYPE_F32, 1,
        video_norm_shape);
    CHECK(audio_bias && filter && video_post && video_norm);

    float filter_values[12];
    float bias_values[2048];
    CHECK(h3_st_read_data(&audio, filter, filter_values,
                          sizeof(filter_values), error, sizeof(error)));
    CHECK(h3_st_read_data(&audio, audio_bias, bias_values,
                          sizeof(bias_values), error, sizeof(error)));
    CHECK(isfinite(bias_values[0]));
    float magnitude = 0.0f;
    for (size_t index = 0; index < 12; index++) {
        CHECK(isfinite(filter_values[index]));
        magnitude += fabsf(filter_values[index]);
    }
    CHECK(magnitude > 0.0f);
    CHECK(video_post->data_end > video_post->data_begin);

    h3_weight_store *store = h3_weight_store_open(
        audio_directory, error, sizeof(error));
    CHECK(store != NULL && h3_weight_store_shards(store) == 1);
    h3_gpu_tensor *loaded_filter = h3_weight_load_f32(
        store, gpu, "decoder.activation_post.downsample.lowpass.filter",
        3, filter_shape, error, sizeof(error));
    CHECK(loaded_filter != NULL);
    float loaded_values[12];
    CHECK(h3_gpu_tensor_read_f32(loaded_filter, loaded_values, 12));
    CHECK(memcmp(filter_values, loaded_values, sizeof(filter_values)) == 0);
    h3_gpu_tensor_free(loaded_filter);
    h3_weight_store_free(store);

    h3_st_free_header(&video);
    h3_st_free_header(&audio);
    return 0;
}

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : "MiniMax-H3";
    char error[512];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu != NULL);
    CHECK(check_partition(root, "FL2VA", gpu) == 0);
    CHECK(check_partition(root, "Ref2VA", gpu) == 0);
    h3_gpu_free(gpu);
    puts("ok: official FL2VA/Ref2VA audio and video VAE schemas/payloads");
    return 0;
}
