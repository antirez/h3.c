#include "h3_lora.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do {                                                   \
    checks++;                                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);   \
        exit(1);                                                                \
    }                                                                           \
} while (0)

static void write_fixture(const char *path, const char *header,
                          const void *payload, size_t payload_size) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    uint64_t header_size = strlen(header);
    CHECK(fwrite(&header_size, sizeof(header_size), 1, file) == 1);
    CHECK(fwrite(header, 1, (size_t)header_size, file) == header_size);
    CHECK(fwrite(payload, 1, payload_size, file) == payload_size);
    CHECK(fclose(file) == 0);
}

static void write_base(const char *directory) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/base.safetensors", directory);
    const char header[] =
        "{\"blocks.0.mlp.fc2.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,3],\"data_offsets\":[0,12]}}";
    uint16_t payload[6] = {0};
    write_fixture(path, header, payload, sizeof(payload));
}

static void test_native(const char *directory, h3_weight_store *base) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/native.safetensors", directory);
    const char header[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,3],\"data_offsets\":[0,12]},"
        "\"blocks.0.mlp.fc2.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,1],\"data_offsets\":[12,20]},"
        "\"blocks.0.mlp.fc2.alpha\":{\"dtype\":\"F32\","
        "\"shape\":[],\"data_offsets\":[20,24]}}";
    float payload[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f};
    write_fixture(path, header, payload, sizeof(payload));
    h3_lora *lora = h3_lora_open(path, base, error, sizeof(error));
    CHECK(lora != NULL);
    CHECK(!strcmp(h3_lora_format(lora), "native"));
    CHECK(strlen(h3_lora_sha256(lora)) == 64);
    CHECK(h3_lora_pair_count(lora) == 1);
    const h3_lora_pair_info *info = h3_lora_pair_at(lora, 0);
    CHECK(info && !strcmp(info->target, "blocks.0.mlp.fc2.weight"));
    CHECK(info->rank == 1 && info->input_dim == 3 && info->output_dim == 2);
    CHECK(fabsf(info->alpha_over_rank - 2.0f) < 1e-7f);
    float a[3], b[2];
    CHECK(h3_lora_read_a(lora, 0, a, 3, error, sizeof(error)));
    CHECK(h3_lora_read_b_rows(lora, 0, 0, 2, b, 2, error, sizeof(error)));
    CHECK(a[0] == 1.0f && a[2] == 3.0f && b[0] == 4.0f && b[1] == 5.0f);
    h3_lora_free(lora);
    CHECK(unlink(path) == 0);
}

static void test_diffusers_f16(const char *directory,
                               h3_weight_store *base) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/diffusers.safetensors", directory);
    const char header[] =
        "{\"diffusion_model.blocks.0.mlp.fc2.lora_A.weight\":{"
        "\"dtype\":\"F16\",\"shape\":[1,3],\"data_offsets\":[0,6]},"
        "\"diffusion_model.blocks.0.mlp.fc2.lora_B.weight\":{"
        "\"dtype\":\"F16\",\"shape\":[2,1],\"data_offsets\":[6,10]}}";
    uint16_t payload[] = {0x3c00, 0xc000, 0x3800, 0x4200, 0xbc00};
    write_fixture(path, header, payload, sizeof(payload));
    h3_lora *lora = h3_lora_open(path, base, error, sizeof(error));
    CHECK(lora != NULL);
    CHECK(!strcmp(h3_lora_format(lora), "diffusion_model"));
    float a[3];
    CHECK(h3_lora_read_a(lora, 0, a, 3, error, sizeof(error)));
    CHECK(a[0] == 1.0f && a[1] == -2.0f && a[2] == 0.5f);
    h3_lora_free(lora);
    CHECK(unlink(path) == 0);
}

static void test_kohya_bf16(const char *directory,
                            h3_weight_store *base) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/kohya.safetensors", directory);
    const char header[] =
        "{\"lora_unet_blocks_0_mlp_fc2.lora_down.weight\":{"
        "\"dtype\":\"BF16\",\"shape\":[1,3],\"data_offsets\":[0,6]},"
        "\"lora_unet_blocks_0_mlp_fc2.lora_up.weight\":{"
        "\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[6,10]},"
        "\"lora_unet_blocks_0_mlp_fc2.alpha\":{"
        "\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[10,12]}}";
    uint16_t payload[] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x4000};
    write_fixture(path, header, payload, sizeof(payload));
    h3_lora *lora = h3_lora_open(path, base, error, sizeof(error));
    CHECK(lora != NULL);
    CHECK(!strcmp(h3_lora_format(lora), "kohya"));
    CHECK(fabsf(h3_lora_pair_at(lora, 0)->alpha_over_rank - 2.0f) < 1e-7f);
    h3_lora_free(lora);
    CHECK(unlink(path) == 0);
}

static void test_rejections(const char *directory, h3_weight_store *base) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/bad.safetensors", directory);
    const char missing[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,3],\"data_offsets\":[0,12]}}";
    float values[] = {1.0f, 2.0f, 3.0f};
    write_fixture(path, missing, values, sizeof(values));
    CHECK(h3_lora_open(path, base, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "missing") != NULL);

    const char nonfinite[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,3],\"data_offsets\":[0,12]},"
        "\"blocks.0.mlp.fc2.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,1],\"data_offsets\":[12,20]}}";
    float bad[] = {1.0f, NAN, 3.0f, 4.0f, 5.0f};
    write_fixture(path, nonfinite, bad, sizeof(bad));
    CHECK(h3_lora_open(path, base, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "NaN or Inf") != NULL);

    const char wrong_shape[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,4],\"data_offsets\":[0,32]},"
        "\"blocks.0.mlp.fc2.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,2],\"data_offsets\":[32,48]}}";
    float wrong_values[12] = {0};
    write_fixture(path, wrong_shape, wrong_values, sizeof(wrong_values));
    CHECK(h3_lora_open(path, base, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "incompatible") != NULL);

    const char duplicate[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,3],\"data_offsets\":[0,12]},"
        "\"blocks.0.mlp.fc2.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,1],\"data_offsets\":[12,20]},"
        "\"diffusion_model.blocks.0.mlp.fc2.lora_A.weight\":{"
        "\"dtype\":\"F32\",\"shape\":[1,3],\"data_offsets\":[20,32]},"
        "\"diffusion_model.blocks.0.mlp.fc2.lora_B.weight\":{"
        "\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[32,40]}}";
    float duplicate_values[10] = {0};
    write_fixture(path, duplicate, duplicate_values, sizeof(duplicate_values));
    CHECK(h3_lora_open(path, base, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "multiple") != NULL);
    CHECK(unlink(path) == 0);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void test_ordered_overlay_and_signature(const char *directory,
                                               h3_weight_store *base) {
    char path[1024], error[512];
    snprintf(path, sizeof(path), "%s/overlay.safetensors", directory);
    const char header[] =
        "{\"blocks.0.mlp.fc2.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,3],\"data_offsets\":[0,12]},"
        "\"blocks.0.mlp.fc2.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[2,1],\"data_offsets\":[12,20]},"
        "\"blocks.0.mlp.fc2.alpha\":{\"dtype\":\"F32\","
        "\"shape\":[],\"data_offsets\":[20,24]}}";
    float payload[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f};
    write_fixture(path, header, payload, sizeof(payload));
    h3_lora_spec specs[] = {
        {path, 0.5f, H3_LORA_FL2VA},
        {path, 0.25f, H3_LORA_FL2VA},
    };
    h3_lora_set *set = h3_lora_set_open(
        specs, 2, base, H3_LORA_FL2VA, error, sizeof(error));
    CHECK(set != NULL);
    CHECK(h3_lora_set_count(set) == 2);
    CHECK(strlen(h3_lora_set_signature(set)) == 64);
    h3_lora_set *different = h3_lora_set_open(
        specs, 1, base, H3_LORA_FL2VA, error, sizeof(error));
    CHECK(different != NULL);
    CHECK(strcmp(h3_lora_set_signature(set),
                 h3_lora_set_signature(different)) != 0);
    h3_lora_set_free(different);
    h3_lora_spec reversed[] = {specs[1], specs[0]};
    different = h3_lora_set_open(
        reversed, 2, base, H3_LORA_FL2VA, error, sizeof(error));
    CHECK(different != NULL);
    CHECK(strcmp(h3_lora_set_signature(set),
                 h3_lora_set_signature(different)) != 0);
    h3_lora_set_free(different);
    h3_lora_spec both = {path, 0.5f, H3_LORA_BOTH};
    different = h3_lora_set_open(
        &both, 1, base, H3_LORA_REF2VA, error, sizeof(error));
    CHECK(different != NULL);
    h3_lora_set_free(different);
    CHECK(h3_lora_set_open(
        specs, 1, base, H3_LORA_REF2VA, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "scope") != NULL);
    CHECK(setenv("H3_DISABLE_RUNTIME_LORA", "1", 1) == 0);
    CHECK(h3_lora_set_open(
        specs, 1, base, H3_LORA_FL2VA, error, sizeof(error)) == NULL);
    CHECK(strstr(error, "disabled") != NULL);
    CHECK(unsetenv("H3_DISABLE_RUNTIME_LORA") == 0);

    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu != NULL);
    size_t peak_scratch = 0;
    h3_gpu_tensor *weight = h3_weight_load_bf16_with_loras(
        base, gpu, "blocks.0.mlp.fc2.weight", 2, 3, set,
        &peak_scratch, error, sizeof(error));
    CHECK(weight != NULL);
    CHECK(peak_scratch > 0 && peak_scratch < 256u * 1024u * 1024u);
    uint16_t got[6];
    CHECK(h3_gpu_tensor_read_bf16(weight, got, 6));
    /* Base is zero. Each adapter applies alpha/rank=2 with scales 0.5 then
     * 0.25, rounding to BF16 after each ordered fold. */
    const float expected[] = {6.0f, 12.0f, 18.0f, 7.5f, 15.0f, 22.5f};
    for (size_t index = 0; index < 6; index++)
        CHECK(got[index] == f32_to_bf16(expected[index]));
    h3_gpu_tensor_free(weight);
    h3_gpu_free(gpu);
    h3_lora_set_free(set);
    CHECK(unlink(path) == 0);
}

int main(void) {
    char directory[] = "/tmp/h3-lora-tests-XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    write_base(directory);
    char error[512];
    h3_weight_store *base = h3_weight_store_open(directory, error, sizeof(error));
    CHECK(base != NULL);
    test_native(directory, base);
    test_diffusers_f16(directory, base);
    test_kohya_bf16(directory, base);
    test_rejections(directory, base);
    test_ordered_overlay_and_signature(directory, base);
    h3_weight_store_free(base);
    char base_path[1024];
    snprintf(base_path, sizeof(base_path), "%s/base.safetensors", directory);
    CHECK(unlink(base_path) == 0);
    CHECK(rmdir(directory) == 0);
    printf("ok: %d runtime LoRA parser checks\n", checks);
    return 0;
}
