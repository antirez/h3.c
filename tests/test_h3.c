#include "h3_host.h"
#include "h3_dit.h"
#include "h3_metal.h"
#include "h3_safetensors.h"
#include "h3_terminal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static int close_enough(double value, double expected, double tolerance) {
    return fabs(value - expected) <= tolerance;
}

static void test_temporal_and_canvas(void) {
    const struct { int requested, frames, video_t, audio_t; } cases[] = {
        {5, 5, 2, 8}, {6, 22, 7, 37}, {22, 22, 7, 37},
        {23, 39, 12, 65}, {56, 56, 17, 93}, {124, 124, 37, 207},
        {361, 362, 107, 603}, {362, 362, 107, 603}
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        h3_temporal_shape got = h3_temporal(cases[index].requested);
        CHECK(got.frame_count == cases[index].frames);
        CHECK(got.video_t == cases[index].video_t);
        CHECK(got.audio_t == cases[index].audio_t);
    }
    CHECK(h3_video_encoder_latent_t(1) == 1);
    CHECK(h3_video_encoder_latent_t(5) == 2);
    CHECK(h3_video_encoder_latent_t(22) == 6);
    CHECK(h3_video_encoder_latent_t(39) == 10);
    int width, height;
    CHECK(h3_adapt_canvas(1920, 1080, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_adapt_canvas(1080, 1920, &width, &height));
    CHECK(width == 768 && height == 1344);
    CHECK(h3_adapt_canvas(864, 480, &width, &height));
    CHECK(width == 1376 && height == 768);
    CHECK(h3_adapt_canvas(32, 32, &width, &height));
    CHECK(width == 768 && height == 768);
    CHECK(h3_reference_image_canvas(1920, 1080, 512, 512, 0,
                                    &width, &height));
    CHECK(width == 672 && height == 384);
    CHECK(h3_reference_image_canvas(640, 480, 1024, 1024, 0,
                                    &width, &height));
    CHECK(width == 640 && height == 480);
    CHECK(h3_reference_image_canvas(4096, 2160, 512, 512, 2048,
                                    &width, &height));
    CHECK(width == 3872 && height == 2048);
    CHECK(!h3_reference_image_canvas(0, 480, 512, 512, 0,
                                     &width, &height));
    CHECK(h3_reference_video_canvas(1920, 1080, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_reference_video_canvas(640, 360, &width, &height));
    CHECK(width == 640 && height == 352);
    CHECK(!h3_reference_video_canvas(0, 360, &width, &height));
}

static void test_schedule(void) {
    h3_params defaults = H3_PARAMS_DEFAULT;
    CHECK(defaults.steps == 20);
    CHECK(defaults.use_reference_rope == 0);

    h3_sigma_schedule schedule;
    CHECK(h3_schedule_build(20, &schedule));
    CHECK(schedule.steps == 20);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[20] == 0.0f && schedule.audio[20] == 0.0f);
    CHECK(close_enough(schedule.video[1], 0.995633185, 1e-7));
    CHECK(close_enough(schedule.audio[1], 0.982758582, 1e-7));
    CHECK(close_enough(schedule.video[19], 0.387096792, 1e-7));
    CHECK(close_enough(schedule.audio[19], 0.136363640, 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }
    double shifted = h3_time_shift_sigma(0.5, 12.0, 3.0);
    double back = h3_time_shift_sigma(shifted, 3.0, 12.0);
    CHECK(close_enough(back, 0.5, 1e-12));
    CHECK(h3_time_shift_slope(0.5, 12.0, 3.0) > 0.0);

    CHECK(h3_serving_schedule_build(50, &schedule));
    CHECK(schedule.steps == 50);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[50] == 0.0f && schedule.audio[50] == 0.0f);
    float base = 0.5f;
    CHECK(close_enough(schedule.video[25],
        12.0f * base / (1.0f + 11.0f * base), 1e-7));
    CHECK(close_enough(schedule.audio[25],
        3.0f * base / (1.0f + 2.0f * base), 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }

    CHECK(h3_serving_schedule_build(4, &schedule));
    CHECK(schedule.steps == 4);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(close_enough(schedule.video[1], 36.0 / 37.0, 1e-7));
    CHECK(close_enough(schedule.audio[1], 9.0 / 10.0, 1e-7));
    CHECK(schedule.video[4] == 0.0f && schedule.audio[4] == 0.0f);
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }
    CHECK(!h3_serving_schedule_build(1, &schedule));
    CHECK(!h3_serving_schedule_build(H3_MAX_STEPS + 1, &schedule));
}

static void test_dit_reuse_schedule(void) {
    uint8_t selected[50];
    const int aggressive[] = {0, 3, 6, 9, 12, 15, 18, 19};
    CHECK(h3_dit_reuse_schedule(20, 3, selected, sizeof(selected)) == 8);
    for (int step = 0; step < 20; step++) {
        int expected = 0;
        for (size_t index = 0;
             index < sizeof(aggressive) / sizeof(*aggressive); index++)
            expected |= step == aggressive[index];
        CHECK(selected[step] == expected);
    }
    CHECK(h3_dit_reuse_schedule(20, 2, selected, sizeof(selected)) == 11);
    for (int step = 0; step < 20; step++)
        CHECK(selected[step] == (step % 2 == 0 || step == 19));
    CHECK(h3_dit_reuse_schedule(50, 3, selected, sizeof(selected)) == 18);
    for (int step = 0; step < 50; step++)
        CHECK(selected[step] == (step % 3 == 0 || step == 49));
    CHECK(h3_dit_reuse_schedule(20, 1, selected, sizeof(selected)) == 20);
    CHECK(h3_dit_reuse_schedule(20, 3, selected, 19) == -1);
}

static void check_segments(const h3_layout *layout,
                           const size_t (*bounds)[2],
                           const h3_segment_kind *kinds, size_t count) {
    CHECK(layout->segment_count == count);
    for (size_t index = 0; index < count; index++) {
        CHECK(layout->segments[index].start == bounds[index][0]);
        CHECK(layout->segments[index].stop == bounds[index][1]);
        CHECK(layout->segments[index].kind == kinds[index]);
        if (index) CHECK(layout->segments[index - 1].stop == bounds[index][0]);
    }
}

static void position_checksums(const h3_layout *layout, double sums[3],
                               double weighted[3]) {
    memset(sums, 0, 3 * sizeof(*sums));
    memset(weighted, 0, 3 * sizeof(*weighted));
    for (size_t index = 0; index < layout->seq_len; index++) {
        const double values[3] = {layout->positions[index].t,
                                  layout->positions[index].h,
                                  layout->positions[index].w};
        for (int axis = 0; axis < 3; axis++) {
            sums[axis] += values[axis];
            weighted[axis] += (double)(index + 1) * values[axis];
        }
    }
}

static void test_layout_tiny(void) {
    h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 30);
    const size_t bounds[][2] = {{0, 12}, {12, 28}, {28, 30}};
    const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_AUDIO, H3_SEG_VIDEO};
    check_segments(&layout, bounds, kinds, 3);
    CHECK(layout.img_target_rows == 2 && layout.audio_target_rows == 16);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 339.6666666666667, 1e-10));
    CHECK(close_enough(weighted[0], 6498.0, 1e-10));
    CHECK(sums[1] == 0.0 && sums[2] == 0.0);
    h3_layout_free(&layout);
}

static void test_layout_fl2va(void) {
    int keyframes[] = {0, 55};
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56,
                           keyframes, 2, NULL, 0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8009);
    const size_t bounds[][2] = {
        {0, 128}, {128, 533}, {533, 938}, {938, 1124}, {1124, 8009}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 5);
    CHECK(layout.img_cond_rows == 810);
    CHECK(layout.img_target_rows == 6885);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 1360927.0, 1e-6));
    CHECK(close_enough(weighted[0], 5883121758.0, 1e-4));
    CHECK(close_enough(sums[1], 117002.11801356057, 1e-7));
    CHECK(close_enough(sums[2], 119830.2393846486, 1e-7));
    h3_layout_free(&layout);
}

static void test_layout_ref2va(void) {
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8287);
    const size_t bounds[][2] = {
        {0, 192}, {192, 288}, {288, 384}, {384, 1056},
        {1056, 1216}, {1216, 1402}, {1402, 8287}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 7);
    CHECK(layout.img_cond_rows == 768);
    CHECK(layout.audio_cond_rows == 256);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 2818905.0, 1e-5));
    CHECK(close_enough(weighted[0], 12788836714.0, 1e-3));
    CHECK(close_enough(sums[1], 115719.96684277552, 1e-7));
    CHECK(close_enough(sums[2], 122360.11344760543, 1e-7));
    h3_layout_free(&layout);
}

static void write_all(int descriptor, const void *data, size_t size) {
    const unsigned char *bytes = data;
    while (size) {
        ssize_t written = write(descriptor, bytes, size);
        CHECK(written > 0);
        bytes += (size_t)written;
        size -= (size_t)written;
    }
}

static void test_safetensors(void) {
    char path[] = "/tmp/h3_safetensors_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    const char header_json[] =
        "{\"scalar\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[24,26]},"
        "\"x\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
    uint64_t length = sizeof(header_json) - 1;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++) prefix[index] = (unsigned char)(length >> (8 * index));
    unsigned char payload[26] = {0};
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header_json, (size_t)length);
    write_all(descriptor, payload, sizeof(payload));
    CHECK(close(descriptor) == 0);

    h3_st_header header;
    char error[256];
    CHECK(h3_st_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 2);
    const h3_st_tensor *x = h3_st_find(&header, "x");
    CHECK(x && x->dtype == H3_DTYPE_F32 && x->ndim == 2);
    CHECK(x->shape[0] == 2 && x->shape[1] == 3);
    CHECK(x->data_end - x->data_begin == 24);
    CHECK(h3_st_tensor_elements(x) == 6);
    unsigned char readback[24];
    CHECK(h3_st_read_data(&header, x, readback, sizeof(readback), error,
                          sizeof(error)));
    for (size_t index = 0; index < sizeof(readback); index++) {
        CHECK(readback[index] == 0);
    }
    CHECK(!h3_st_read_data(&header, x, readback, sizeof(readback) - 1, error,
                           sizeof(error)));
    const h3_st_tensor *scalar = h3_st_find(&header, "scalar");
    CHECK(scalar && scalar->dtype == H3_DTYPE_BF16 && scalar->ndim == 0);
    CHECK(scalar->data_end - scalar->data_begin == 2);
    CHECK(h3_st_tensor_elements(scalar) == 1);
    h3_st_free_header(&header);
    CHECK(unlink(path) == 0);

    char overlap_path[] = "/tmp/h3_safetensors_overlap_XXXXXX";
    descriptor = mkstemp(overlap_path);
    CHECK(descriptor >= 0);
    const char overlap_json[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        "\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[2,6]}}";
    length = sizeof(overlap_json) - 1;
    for (unsigned index = 0; index < 8; index++)
        prefix[index] = (unsigned char)(length >> (8 * index));
    unsigned char overlap_payload[6] = {0};
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, overlap_json, (size_t)length);
    write_all(descriptor, overlap_payload, sizeof(overlap_payload));
    CHECK(close(descriptor) == 0);
    CHECK(!h3_st_read_header(overlap_path, &header, error, sizeof(error)));
    CHECK(unlink(overlap_path) == 0);
}

static void test_rng_and_solver(void) {
    h3_rng a, b;
    h3_rng_seed(&a, 42);
    h3_rng_seed(&b, 42);
    for (int index = 0; index < 64; index++) CHECK(h3_rng_u32(&a) == h3_rng_u32(&b));
    h3_rng_seed(&a, 7);
    float values[1024];
    h3_rng_fill_normal(&a, values, 1024);
    double sum = 0.0;
    for (size_t index = 0; index < 1024; index++) {
        CHECK(isfinite(values[index]));
        sum += values[index];
    }
    CHECK(fabs(sum / 1024.0) < 0.15);

    const float sigmas[] = {1.0f, 0.75f, 0.25f, 0.0f};
    float sample[] = {0.0f};
    float denoised[] = {1.0f};
    float old[] = {0.5f};
    float output[1];
    CHECK(h3_res_step(output, sample, denoised, NULL, 1, sigmas, 0, 3));
    CHECK(close_enough(output[0], 0.25, 1e-7));
    CHECK(h3_res_step(output, sample, denoised, old, 1, sigmas, 1, 3));
    CHECK(isfinite(output[0]));
    float velocity[] = {2.0f, -4.0f};
    float euler[] = {1.0f, 3.0f};
    CHECK(h3_euler_velocity_step(euler, velocity, 2, 0.75f, 0.25f));
    CHECK(euler[0] == 2.0f && euler[1] == 1.0f);
    CHECK(!h3_euler_velocity_step(euler, velocity, 2, 0.25f, 0.25f));
}

static void test_rgb_resize(void) {
    const uint8_t constant[] = {
        17, 33, 201, 17, 33, 201,
        17, 33, 201, 17, 33, 201
    };
    uint8_t *identity = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 2, 2, &identity));
    CHECK(identity != NULL);
    CHECK(memcmp(identity, constant, sizeof(constant)) == 0);
    free(identity);

    uint8_t *upscaled = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 8, 8, &upscaled));
    CHECK(upscaled != NULL);
    for (size_t pixel = 0; pixel < 64; pixel++) {
        CHECK(upscaled[3 * pixel] == 17);
        CHECK(upscaled[3 * pixel + 1] == 33);
        CHECK(upscaled[3 * pixel + 2] == 201);
    }
    free(upscaled);
    CHECK(!h3_resize_rgb24_high_quality(NULL, 1, 2, 2, 8, 8, &upscaled));
}

static void test_dit_row_conversions(void) {
    enum { C = 3, T = 2, H = 4, W = 6, VIDEO = C * T * H * W };
    float latent[VIDEO], rows[VIDEO], roundtrip[VIDEO];
    for (size_t index = 0; index < VIDEO; index++) latent[index] = (float)index;
    CHECK(h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO));
    /* First row is pixel patch (h=0..1,w=0..1), channel-major inside it. */
    const float first[] = {0, 1, 6, 7, 48, 49, 54, 55, 96, 97, 102, 103};
    for (size_t index = 0; index < sizeof(first) / sizeof(*first); index++)
        CHECK(rows[index] == first[index]);
    CHECK(h3_dit_unpatchify_video(rows, C, T, H, W, roundtrip, VIDEO));
    CHECK(memcmp(latent, roundtrip, sizeof(latent)) == 0);
    CHECK(!h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO - 1));

    enum { AC = 4, AT = 3, AUDIO = AC * 2 * AT };
    float audio[AUDIO], packed[AUDIO], unpacked[AUDIO];
    for (size_t index = 0; index < AUDIO; index++) audio[index] = (float)index;
    CHECK(h3_dit_pack_audio(audio, AC, AT, packed, AUDIO));
    const float expected[] = {0, 6, 12, 18, 1, 7, 13, 19,
                              2, 8, 14, 20, 3, 9, 15, 21,
                              4, 10, 16, 22, 5, 11, 17, 23};
    CHECK(memcmp(packed, expected, sizeof(expected)) == 0);
    CHECK(h3_dit_unpack_audio(packed, AC, AT, unpacked, AUDIO));
    CHECK(memcmp(audio, unpacked, sizeof(audio)) == 0);
}

static void test_metal_probe(void) {
    h3_device_info info;
    char error[256];
    CHECK(h3_metal_probe(&info, error, sizeof(error)));
    CHECK(info.name[0] != '\0');
    CHECK(info.physical_memory >= UINT64_C(8) * 1024 * 1024 * 1024);
    CHECK(info.max_buffer_length > 0);
    CHECK(info.apple_gpu_family > 0);
}

static void test_terminal_zoom(void) {
    int width = 0, height = 0;
    CHECK(h3_terminal_set_zoom(2));
    CHECK(h3_terminal_display_dimensions(512, 288, &width, &height));
    CHECK(width == 1024 && height == 576);
    CHECK(h3_terminal_set_zoom(1));
    CHECK(h3_terminal_display_dimensions(512, 288, &width, &height));
    CHECK(width == 512 && height == 288);
    CHECK(!h3_terminal_set_zoom(0));
    CHECK(h3_terminal_set_zoom(2));
    CHECK(!h3_terminal_display_dimensions(INT32_MAX, 1, &width, &height));
}

int main(void) {
    test_temporal_and_canvas();
    test_schedule();
    test_dit_reuse_schedule();
    test_layout_tiny();
    test_layout_fl2va();
    test_layout_ref2va();
    test_safetensors();
    test_rng_and_solver();
    test_rgb_resize();
    test_dit_row_conversions();
    test_metal_probe();
    test_terminal_zoom();
    printf("ok: %d checks\n", tests_run);
    return 0;
}
