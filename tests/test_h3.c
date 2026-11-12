#include "h3_host.h"
#include "h3_dit.h"
#include "h3_dit_names.h"
#include "h3_dit_schedule.h"
#include "h3_image.h"
#include "h3_image_vae.h"
#include "h3_metal.h"
#include "h3_safetensors.h"
#include "h3_vae_tiles.h"
#include "h3_terminal.h"
#include "h3_weights.h"

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
    h3_layout_summary summary;
    CHECK(h3_layout_validate(&layout, 12, &summary, error, sizeof(error)));
    CHECK(!summary.image);
    CHECK(summary.video_target_start == 28 && summary.audio_target_start == 12);
    CHECK(summary.video_condition == 0 && summary.audio_condition == 0);
    CHECK(summary.video_target == 2 && summary.audio_target == 16);
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
    h3_layout_summary summary;
    CHECK(h3_layout_validate(&layout, 128, &summary, error, sizeof(error)));
    CHECK(!summary.image);
    CHECK(summary.video_target_start == 1124 && summary.audio_target_start == 938);
    CHECK(summary.video_condition == 810 && summary.audio_condition == 0);
    CHECK(summary.video_target == 6885 && summary.audio_target == 186);
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
    h3_layout_summary summary;
    CHECK(h3_layout_validate(&layout, 192, &summary, error, sizeof(error)));
    CHECK(!summary.image);
    CHECK(summary.video_target_start == 1402 && summary.audio_target_start == 1216);
    CHECK(summary.video_condition == 768 && summary.audio_condition == 256);
    CHECK(summary.video_target == 6885 && summary.audio_target == 186);
    h3_layout_free(&layout);
}

static void test_layout_validate(void) {
    h3_layout layout;
    h3_layout_summary summary;
    char error[256];
    CHECK(h3_layout_build_image(3, 4, 4, NULL, 0, &layout, error, sizeof(error)));
    CHECK(h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    CHECK(summary.image && summary.video_target == 4 && summary.video_target_start == 3);
    CHECK(summary.video_condition == 0 && summary.audio_target == 0);
    CHECK(!h3_layout_validate(&layout, 4, &summary, error, sizeof(error)));
    layout.signature[1] = 2;
    CHECK(!h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    layout.signature[1] = 1;
    CHECK(h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    h3_layout_free(&layout);

    const h3_layout_ref reference = {H3_LAYOUT_REF_IMAGE, 1, 2, 2, 0};
    CHECK(h3_layout_build_image(3, 4, 4, &reference, 1, &layout,
                                error, sizeof(error)));
    CHECK(h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    CHECK(summary.video_condition == 1 && summary.video_target_start == 4);
    layout.segments[1].kind = H3_SEG_COND;
    CHECK(!h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    layout.segments[1].kind = H3_SEG_REF_AUDIO;
    layout.audio_cond_rows = 1;
    layout.img_cond_rows = 0;
    CHECK(!h3_layout_validate(&layout, 3, &summary, error, sizeof(error)));
    h3_layout_free(&layout);

    const h3_layout_spec no_audio[2] = {
        {12, 1, 2, 2, 0, 5, NULL, 0, NULL, 0},
        {12, 2, 2, 2, 0, 5, NULL, 0, NULL, 0}
    };
    for (size_t index = 0; index < 2; index++) {
        CHECK(h3_layout_build(&no_audio[index], &layout, error, sizeof(error)));
        CHECK(!h3_layout_validate(&layout, 12, &summary, error, sizeof(error)));
        h3_layout_free(&layout);
    }
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
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"scalar\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[24,26]}}";
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

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void test_image_schedule(void) {
    h3_sigma_schedule schedule;
    CHECK(h3_image_schedule_build(&schedule));
    CHECK(schedule.steps == 6);
    const uint32_t sigma_bits[7] = {
        0x3f800000, 0x3f77afa2, 0x3f633661, 0x3f400000,
        0x3f0863df, 0x3e6d93bc, 0x00000000
    };
    const uint32_t timestep_bits[6] = {
        0x00000000, 0x3d0505e0, 0x3de64cf8, 0x3e800000, 0x3eef3842, 0x3f449b11
    };
    const float deltas[6] = {
        0.032476306f, 0.07997519f, 0.1375485f, 0.21722609f, 0.30076504f,
        0.23200887f
    };
    for (int index = 0; index <= 6; index++) {
        CHECK(float_bits(schedule.video[index]) == sigma_bits[index]);
        CHECK(float_bits(schedule.audio[index]) == sigma_bits[index]);
    }
    for (int index = 0; index < 6; index++) {
        CHECK(float_bits(1.0f - schedule.video[index]) == timestep_bits[index]);
        CHECK(close_enough(schedule.video[index] - schedule.video[index + 1],
                           deltas[index], 1e-7));
    }
    CHECK(!h3_image_schedule_build(NULL));
}

static void test_image_canvas(void) {
    const struct { int width, height, min_area, canvas_w, canvas_h; } cases[] = {
        {1024, 1024, 4194304, 2048, 2048},
        {768, 1024, 4194304, 1760, 2368},
        {1024, 768, 4194304, 2368, 1760},
        {1920, 1080, 4194304, 2720, 1536},
        {1000, 1000, 1024, 992, 992},
        {4096, 4096, 4194304, 4096, 4096},
        {16, 16, 1024, 32, 32},
        {48, 80, 1024, 64, 64}
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        int canvas_w = 0, canvas_h = 0;
        CHECK(h3_image_canvas(cases[index].height, cases[index].width,
                              cases[index].min_area, &canvas_h, &canvas_w));
        CHECK(canvas_w == cases[index].canvas_w);
        CHECK(canvas_h == cases[index].canvas_h);
    }
    int canvas_w, canvas_h;
    CHECK(h3_image_canvas(1024, 1024, 1024, &canvas_h, &canvas_w));
    CHECK(h3_image_canvas(1024, 1024, 16777216, &canvas_h, &canvas_w));
    CHECK(canvas_w == 4096 && canvas_h == 4096);
    CHECK(!h3_image_canvas(1024, 1024, 1023, &canvas_h, &canvas_w));
    CHECK(!h3_image_canvas(1024, 1024, 16777217, &canvas_h, &canvas_w));
    CHECK(!h3_image_canvas(0, 1024, 4194304, &canvas_h, &canvas_w));
    CHECK(!h3_image_canvas(1024, -1, 4194304, &canvas_h, &canvas_w));
    CHECK(!h3_image_canvas(1024, 1024, 4194304, NULL, &canvas_w));
}

static void test_image_layout(void) {
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build_image(3, 4, 4, NULL, 0, &layout,
                                error, sizeof(error)));
    CHECK(layout.seq_len == 7);
    const size_t bounds[][2] = {{0, 3}, {3, 7}};
    const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_VIDEO};
    check_segments(&layout, bounds, kinds, 2);
    CHECK(layout.img_cond_rows == 0 && layout.img_target_rows == 4);
    CHECK(layout.audio_cond_rows == 0 && layout.audio_target_rows == 0);
    const int signature[5] = {3, 1, 4, 4, 0};
    CHECK(memcmp(layout.signature, signature, sizeof(signature)) == 0);
    for (size_t row = 0; row < 3; row++) {
        CHECK(layout.positions[row].t == (double)row);
        CHECK(layout.positions[row].h == 0.0 && layout.positions[row].w == 0.0);
    }
    const double axis[2] = {0.0, 16.0};
    for (size_t row = 0; row < 4; row++) {
        CHECK(layout.positions[3 + row].t == 3.0);
        CHECK(close_enough(layout.positions[3 + row].h, axis[row / 2], 1e-6));
        CHECK(close_enough(layout.positions[3 + row].w, axis[row % 2], 1e-6));
    }
    h3_layout_free(&layout);

    CHECK(h3_layout_build_image(3, 6, 4, NULL, 0, &layout,
                                error, sizeof(error)));
    CHECK(layout.seq_len == 9 && layout.img_target_rows == 6);
    const double heights[3] = {-3.59591794, 9.46802735, 22.53197265};
    const double widths[2] = {2.93605471, 16.0};
    for (size_t row = 0; row < 6; row++) {
        CHECK(close_enough(layout.positions[3 + row].h, heights[row / 2], 1e-6));
        CHECK(close_enough(layout.positions[3 + row].w, widths[row % 2], 1e-6));
    }
    h3_layout_free(&layout);

    const h3_layout_ref refs[2] = {
        {H3_LAYOUT_REF_IMAGE, 1, 2, 4, 0}, {H3_LAYOUT_REF_IMAGE, 1, 2, 2, 0}
    };
    CHECK(h3_layout_build_image(2, 4, 4, refs, 2, &layout,
                                error, sizeof(error)));
    CHECK(layout.seq_len == 2 + 2 + 1 + 4);
    const size_t ref_bounds[][2] = {{0, 2}, {2, 4}, {4, 5}, {5, 9}};
    const h3_segment_kind ref_kinds[] = {
        H3_SEG_TEXT, H3_SEG_REF_IMAGE, H3_SEG_REF_IMAGE, H3_SEG_VIDEO
    };
    check_segments(&layout, ref_bounds, ref_kinds, 4);
    CHECK(layout.img_cond_rows == 3 && layout.img_target_rows == 4);
    CHECK(layout.positions[2].t == 2.0 && layout.positions[4].t == 3.0);
    CHECK(layout.positions[5].t == 4.0);
    CHECK(close_enough(layout.positions[2].h, 4.6862915, 1e-6));
    CHECK(close_enough(layout.positions[2].w, -6.627417, 1e-6));
    CHECK(close_enough(layout.positions[3].h, 4.6862915, 1e-6));
    CHECK(close_enough(layout.positions[3].w, 16.0, 1e-6));
    CHECK(close_enough(layout.positions[4].h, 0.0, 1e-6));
    CHECK(close_enough(layout.positions[4].w, 0.0, 1e-6));
    h3_layout_free(&layout);

    const h3_layout_ref audio = {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 4};
    CHECK(!h3_layout_build_image(3, 4, 4, &audio, 1, &layout,
                                 error, sizeof(error)));
    CHECK(error[0] != '\0' && layout.positions == NULL);
    CHECK(!h3_layout_build_image(0, 4, 4, NULL, 0, &layout,
                                 error, sizeof(error)));
    CHECK(!h3_layout_build_image(3, 3, 4, NULL, 0, &layout,
                                 error, sizeof(error)));
}

static void test_image_rope_inv_freq(void) {
    const uint32_t dit[16] = {
        0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2,
        0x3d0186e3, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528,
        0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753
    };
    float values[16];
    h3_image_dit_inv_freq(values);
    for (size_t index = 0; index < 16; index++)
        CHECK(float_bits(values[index]) == dit[index]);
    h3_image_vae_inv_freq(values);
    for (size_t index = 0; index < 8; index++)
        CHECK(float_bits(values[index]) == dit[index]);
}

static void test_image_timestep_features(void) {
    float row[H3_DIT_TIME_INPUT];
    h3_dit_time_features(0.032476306f, row);
    CHECK(close_enough(row[0], 0.9994727, 1e-6));
    CHECK(close_enough(row[128], 0.0324706, 1e-6));
    CHECK(close_enough(row[127], 1.0, 1e-6));
    CHECK(close_enough(row[128 + 2], 0.02811961, 1e-6));
    h3_dit_time_features(0.0f, row);
    CHECK(row[0] == 1.0f && row[128] == 0.0f && row[255] == 0.0f);
}

static void test_image_patchify(void) {
    enum { C = 24, H = 4, W = 4, ELEMENTS = C * H * W, ROWS = 4, ROW = C * 4 };
    float latent[ELEMENTS], rows[ELEMENTS], roundtrip[ELEMENTS];
    for (size_t index = 0; index < ELEMENTS; index++) latent[index] = (float)index;
    CHECK(h3_dit_patchify_video(latent, C, 1, H, W, rows, ELEMENTS));
    for (size_t row = 0; row < ROWS; row++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t dh = 0; dh < 2; dh++) {
                for (size_t dw = 0; dw < 2; dw++) {
                    size_t y = 2 * (row / 2) + dh, x = 2 * (row % 2) + dw;
                    CHECK(rows[row * ROW + c * 4 + dh * 2 + dw] ==
                          latent[(c * H + y) * W + x]);
                }
            }
        }
    }
    CHECK(h3_dit_unpatchify_video(rows, C, 1, H, W, roundtrip, ELEMENTS));
    CHECK(memcmp(latent, roundtrip, sizeof(latent)) == 0);
}

static void test_image_qkv_interleave(void) {
    const struct { unsigned heads, head_dim; } shapes[] = {{56, 128}, {16, 64}};
    for (size_t shape = 0; shape < 2; shape++) {
        unsigned heads = shapes[shape].heads, dim = shapes[shape].head_dim;
        size_t rows = (size_t)3 * heads * dim;
        unsigned char *seen = calloc(rows, 1);
        CHECK(seen != NULL);
        for (size_t dest = 0; dest < rows; dest++) {
            unsigned projection;
            size_t source;
            CHECK(h3_image_qkv_source(dest, heads, dim, &projection, &source));
            size_t head = dest / ((size_t)3 * dim);
            CHECK(projection == (dest / dim) % 3);
            CHECK(source == head * dim + dest % dim);
            size_t flat = projection * (size_t)heads * dim + source;
            CHECK(!seen[flat]);
            seen[flat] = 1;
        }
        free(seen);
        unsigned projection;
        size_t source;
        CHECK(!h3_image_qkv_source(rows, heads, dim, &projection, &source));
    }
    unsigned projection;
    size_t source;
    CHECK(h3_image_qkv_source(56 * 384 - 1, 56, 128, &projection, &source));
    CHECK(projection == 2 && source == 56 * 128 - 1);
    CHECK(h3_image_qkv_source(128, 56, 128, &projection, &source));
    CHECK(projection == 1 && source == 0);
    CHECK(h3_image_qkv_source(384 + 2 * 128 + 5, 56, 128, &projection, &source));
    CHECK(projection == 2 && source == 128 + 5);
    CHECK(h3_image_qkv_source(192 + 64 + 5, 16, 64, &projection, &source));
    CHECK(projection == 1 && source == 64 + 5);
    CHECK(!h3_image_qkv_source(0, 0, 128, &projection, &source));
}

static void test_dit_names(void) {
    const h3_dit_names *video = h3_dit_names_for(H3_DIT_FAMILY_VIDEO);
    const h3_dit_names *image = h3_dit_names_for(H3_DIT_FAMILY_IMAGE);
    CHECK(video && image && video != image);
    CHECK(video->family == H3_DIT_FAMILY_VIDEO);
    CHECK(image->family == H3_DIT_FAMILY_IMAGE);
    char name[160];
    CHECK(h3_dit_block_name(name, sizeof(name), video, 0, 7,
                            video->qkv));
    CHECK(strcmp(name, "blocks.7.attn.qkv_proj.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), video, 0, 49, video->fc1));
    CHECK(strcmp(name, "blocks.49.mlp.fc1.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), video, 0, 0,
                            video->adaln_bias));
    CHECK(strcmp(name, "blocks.0.adaln_proj.linear.bias") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), video, 1, 1, video->q_norm));
    CHECK(strcmp(name, "token_refiner.blocks.1.attn.q_norm.weight") == 0);
    CHECK(strcmp(video->time_in_weight, "time_embedder.proj_in.weight") == 0);
    CHECK(strcmp(video->time_out_bias, "time_embedder.proj_out.bias") == 0);
    CHECK(strcmp(video->condition_weight, "condition_proj.weight") == 0);
    CHECK(strcmp(video->patch_weight, "video_patch_proj.weight") == 0);
    CHECK(strcmp(video->final_norm, "final_layer.norm.weight") == 0);
    CHECK(strcmp(video->final_adaln_weight,
                 "final_layer.adaln_proj.linear.weight") == 0);
    CHECK(strcmp(video->refiner_final_norm,
                 "token_refiner.final_norm.weight") == 0);
    CHECK(strcmp(video->rope_inv_freq, "rope.inv_freq") == 0);
    CHECK(video->q == NULL && video->k == NULL && video->v == NULL);
    CHECK(video->audio_patch_weight && video->audio_head_weight);
    CHECK(h3_dit_name_format(name, sizeof(name), video->head_weight, 3));
    CHECK(strcmp(name, "final_layer.video_out.weight") == 0);

    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 12, image->q));
    CHECK(strcmp(name, "transformer_blocks.12.attn.to_q.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 0, image->v));
    CHECK(strcmp(name, "transformer_blocks.0.attn.to_v.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 0, image->k_norm));
    CHECK(strcmp(name, "transformer_blocks.0.attn.norm_k.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 0, image->out));
    CHECK(strcmp(name, "transformer_blocks.0.attn.to_out.0.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 49, image->fc1));
    CHECK(strcmp(name, "transformer_blocks.49.ff.net.0.proj.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 3, image->fc2));
    CHECK(strcmp(name, "transformer_blocks.3.ff.net.2.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 0, 3,
                            image->adaln_weight));
    CHECK(strcmp(name, "transformer_blocks.3.adaln_proj.linear.weight") == 0);
    CHECK(h3_dit_block_name(name, sizeof(name), image, 1, 0, image->norm1));
    CHECK(strcmp(name, "token_refiner.refiner_blocks.0.norm1.weight") == 0);
    CHECK(strcmp(image->time_in_weight, "time_embedder.linear_1.weight") == 0);
    CHECK(strcmp(image->time_out_bias, "time_embedder.linear_2.bias") == 0);
    CHECK(strcmp(image->condition_bias, "context_embedder.bias") == 0);
    CHECK(strcmp(image->patch_weight, "proj_in.weight") == 0);
    CHECK(strcmp(image->final_norm, "norm_out.norm.weight") == 0);
    CHECK(strcmp(image->final_adaln_bias, "norm_out.linear.bias") == 0);
    CHECK(strcmp(image->refiner_final_norm,
                 "token_refiner.final_norm.weight") == 0);
    CHECK(h3_dit_name_format(name, sizeof(name), image->head_weight, 5));
    CHECK(strcmp(name, "pdd_proj_out.5.weight") == 0);
    CHECK(h3_dit_name_format(name, sizeof(name), image->head_bias, 0));
    CHECK(strcmp(name, "pdd_proj_out.0.bias") == 0);
    CHECK(image->qkv == NULL && image->rope_inv_freq == NULL);
    CHECK(image->audio_patch_weight == NULL && image->audio_head_weight == NULL);

    char tiny[8];
    CHECK(!h3_dit_block_name(tiny, sizeof(tiny), image, 0, 0, image->q));
    CHECK(!h3_dit_block_name(name, sizeof(name), image, 0, 0, NULL));
    CHECK(!h3_dit_name_format(name, sizeof(name), NULL, 0));
    CHECK(h3_dit_names_for((h3_dit_family)99) == NULL);
}

static void test_image_dit_bookkeeping(void) {
    h3_layout layout;
    h3_layout_summary summary;
    char error[256];
    CHECK(h3_layout_build_image(5, 32, 32, NULL, 0, &layout, error,
                                sizeof(error)));
    CHECK(h3_layout_validate(&layout, 5, &summary, error, sizeof(error)));
    CHECK(layout.seq_len == 5 + 256);
    CHECK(layout.audio_target_rows == 0 && layout.img_target_rows == 256);
    CHECK(summary.audio_condition == 0 && summary.audio_target == 0);
    CHECK(summary.video_condition == 0 && summary.video_target == 256);
    CHECK(summary.video_target_start == 5);
    CHECK(layout.segment_count == 2);
    CHECK(layout.segments[0].kind == H3_SEG_TEXT &&
          layout.segments[1].kind == H3_SEG_VIDEO);
    h3_layout_free(&layout);

    for (int step = 0; step < H3_IMAGE_STEPS; step++)
        CHECK(h3_dit_image_head_for_step(step) == step);
    CHECK(h3_dit_image_head_for_step(-1) == -1);
    CHECK(h3_dit_image_head_for_step(H3_IMAGE_STEPS) == -1);
    CHECK(h3_dit_image_rows(NULL) == 0);

    h3_sigma_schedule schedule;
    CHECK(h3_image_schedule_build(&schedule));
    for (int step = 0; step < H3_IMAGE_STEPS; step++) {
        CHECK(schedule.video[step] == schedule.audio[step]);
        for (int other = 0; other < step; other++)
            CHECK(schedule.video[step] != schedule.video[other]);
    }
}

static void test_missing_weight_names(void) {
    const h3_dit_names *image = h3_dit_names_for(H3_DIT_FAMILY_IMAGE);
    const h3_dit_names *video = h3_dit_names_for(H3_DIT_FAMILY_VIDEO);
    CHECK(image->audio_patch_weight == NULL && image->audio_patch_bias == NULL);
    CHECK(image->audio_head_weight == NULL && image->audio_head_bias == NULL);
    CHECK(video->audio_patch_weight && video->audio_patch_bias);
    CHECK(video->audio_head_weight && video->audio_head_bias);
    uint64_t shape[2] = {2, 2};
    char error[256] = "";
    CHECK(h3_weight_load_bf16(NULL, NULL, image->audio_patch_weight, 2, shape,
                              error, sizeof(error)) == NULL);
    CHECK(strstr(error, "missing") != NULL);
    error[0] = '\0';
    CHECK(h3_weight_load_f32(NULL, NULL, image->audio_head_bias, 1, shape,
                             error, sizeof(error)) == NULL);
    CHECK(strstr(error, "missing") != NULL);
    error[0] = '\0';
    CHECK(h3_weight_load_bf16_qkv(NULL, NULL, NULL, "k", "v", 2, 2, 2, error,
                                  sizeof(error)) == NULL);
    CHECK(strstr(error, "missing") != NULL);
    error[0] = '\0';
    CHECK(h3_weight_load_bf16_halves(NULL, NULL, NULL, 4, 2, 1, error,
                                     sizeof(error)) == NULL);
    CHECK(error[0] != '\0');
}

static void test_qkv_slab_plan(void) {
    const uint64_t columns = 5376, slab = 128 * columns;
    unsigned projection;
    uint64_t source, destination;
    CHECK(h3_weight_qkv_slab(0, 56, 128, columns, &projection, &source,
                             &destination));
    CHECK(projection == 0 && source == 0 && destination == 0);
    CHECK(h3_weight_qkv_slab(1, 56, 128, columns, &projection, &source,
                             &destination));
    CHECK(projection == 1 && source == 0 && destination == slab);
    CHECK(h3_weight_qkv_slab(5, 56, 128, columns, &projection, &source,
                             &destination));
    CHECK(projection == 2 && source == slab && destination == 5 * slab);
    CHECK(h3_weight_qkv_slab(167, 56, 128, columns, &projection, &source,
                             &destination));
    CHECK(projection == 2 && source == 55 * slab && destination == 167 * slab);
    CHECK(!h3_weight_qkv_slab(168, 56, 128, columns, &projection, &source,
                              &destination));
    CHECK(!h3_weight_qkv_slab(0, 0, 128, columns, &projection, &source,
                              &destination));

    /* Applying the plan to three small matrices gives the row order that
     * h3_image_qkv_source describes. */
    enum { HEADS = 3, DIM = 4, COLUMNS = 5 };
    int matrices[3][HEADS * DIM * COLUMNS];
    int grouped[3 * HEADS * DIM * COLUMNS];
    for (int p = 0; p < 3; p++)
        for (int index = 0; index < HEADS * DIM * COLUMNS; index++)
            matrices[p][index] = p * 1000 + index;
    for (size_t slab_index = 0; slab_index < 3 * HEADS; slab_index++) {
        CHECK(h3_weight_qkv_slab(slab_index, HEADS, DIM, COLUMNS,
                                 &projection, &source, &destination));
        memcpy(grouped + destination, matrices[projection] + source,
               DIM * COLUMNS * sizeof(int));
    }
    for (size_t row = 0; row < 3 * HEADS * DIM; row++) {
        size_t source_row;
        CHECK(h3_image_qkv_source(row, HEADS, DIM, &projection, &source_row));
        CHECK(memcmp(grouped + row * COLUMNS,
                     matrices[projection] + source_row * COLUMNS,
                     COLUMNS * sizeof(int)) == 0);
    }
}

static void check_tile_axis(int extent, int count, const int *overlaps) {
    h3_vae_tile_axis axis;
    char error[256];
    CHECK(h3_vae_tile_axis_build(extent, 256, &axis, error, sizeof(error)));
    CHECK(axis.count == count && axis.length == 256);
    CHECK(axis.starts[0] == 0);
    for (int index = 1; index < count; index++) {
        CHECK(axis.overlaps[index - 1] == overlaps[index - 1]);
        CHECK(axis.starts[index] ==
              axis.starts[index - 1] + 256 - overlaps[index - 1]);
    }
    CHECK(axis.starts[count - 1] + axis.length == extent);
    h3_vae_tile_axis_free(&axis);
}

static void test_image_vae_tiles(void) {
    int overlaps[20];
    for (int index = 0; index < 20; index++) overlaps[index] = 64;
    check_tile_axis(4096, 21, overlaps);
    h3_vae_tile_axis axis;
    char error[256];
    CHECK(h3_vae_tile_axis_build(4096, 256, &axis, error, sizeof(error)));
    CHECK(axis.starts[1] == 192 && axis.starts[2] == 384);
    h3_vae_tile_axis_free(&axis);

    for (int index = 0; index < 8; index++) overlaps[index] = 80;
    overlaps[8] = overlaps[9] = 64;
    check_tile_axis(2048, 11, overlaps);
    overlaps[0] = overlaps[1] = 128;
    check_tile_axis(512, 3, overlaps);

    CHECK(h3_vae_tile_axis_build(256, 256, &axis, error, sizeof(error)));
    CHECK(axis.count == 1 && axis.length == 256 && axis.starts[0] == 0);
    h3_vae_tile_axis_free(&axis);
    CHECK(h3_vae_tile_axis_build(160, 256, &axis, error, sizeof(error)));
    CHECK(axis.count == 1 && axis.length == 160);
    h3_vae_tile_axis_free(&axis);
    CHECK(!h3_vae_tile_axis_build(250, 256, &axis, error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(h3_vae_tile_count(4096, 256) == 21);
    CHECK(h3_vae_tile_count(2048, 256) == 11);
}

static void test_image_unpack_2d(void) {
    enum { HL = 2, WL = 3, PATCH = 768 };
    float *rows = malloc((size_t)HL * WL * PATCH * sizeof(*rows));
    float *rgb = malloc((size_t)HL * WL * 256 * 3 * sizeof(*rgb));
    CHECK(rows && rgb);
    for (int patch = 0; patch < HL * WL; patch++)
        for (int component = 0; component < PATCH; component++)
            rows[patch * PATCH + component] = (float)(patch * 1000 + component);
    h3_image_vae_unpack_patches(rows, HL, WL, rgb);
    const int width = WL * 16;
    /* pixel (y, x, channel): patch (y / 16) * WL + x / 16, component
     * (channel * 16 + y % 16) * 16 + x % 16 */
    CHECK(rgb[(0 * width + 0) * 3 + 0] == 0.0f);
    CHECK(rgb[(0 * width + 1) * 3 + 0] == 1.0f);
    CHECK(rgb[(1 * width + 0) * 3 + 0] == 16.0f);
    CHECK(rgb[(0 * width + 0) * 3 + 1] == 256.0f);
    CHECK(rgb[(0 * width + 0) * 3 + 2] == 512.0f);
    CHECK(rgb[(0 * width + 16) * 3 + 0] == 1000.0f);
    CHECK(rgb[(17 * width + 35) * 3 + 2] == 5531.0f);
    CHECK(rgb[(31 * width + 47) * 3 + 1] == 5511.0f);
    free(rows);
    free(rgb);

    /* Raw decoder values 0, 3, -5 per channel: 0 maps to the ImageNet mean,
     * 3 and -5 clamp to the ends of [0, 1]; 0.9999 truncates to 254. */
    const float raw_nine = (0.9999f - 0.485f) / 0.229f;
    float data[3 * 4] = {
        0.0f, 3.0f, -5.0f, raw_nine,
        0.0f, 3.0f, -5.0f, 0.0f,
        0.0f, 3.0f, -5.0f, 0.0f
    };
    h3_image_planes planes = {2, 2, data, {0}};
    h3_image_planes_to_rgb(&planes);
    CHECK(close_enough(data[0], 0.485, 1e-6) && data[1] == 1.0f && data[2] == 0.0f);
    CHECK(close_enough(data[3], 0.9999, 1e-6));
    CHECK(close_enough(data[4], 0.456, 1e-6));
    CHECK(close_enough(data[8], 0.406, 1e-6));
    uint8_t bytes[12];
    h3_image_planes_to_u8(&planes, bytes);
    CHECK(bytes[0] == 123 && bytes[1] == 255 && bytes[2] == 0 && bytes[3] == 254);
    CHECK(bytes[4] == 116 && bytes[8] == 103);
}

static void test_image_vae_config_parse(void) {
    float values[3];
    char error[256];
    CHECK(h3_image_vae_parse_float_array("{\"a\": 1, \"m\": [0.5, -1.5,\n 2e1]}",
                                         "m", values, 3, error, sizeof(error)));
    CHECK(values[0] == 0.5f && values[1] == -1.5f && values[2] == 20.0f);
    CHECK(!h3_image_vae_parse_float_array("{\"m\": [1, 2]}", "m", values, 3,
                                          error, sizeof(error)));
    CHECK(!h3_image_vae_parse_float_array("{\"m\": [1, 2, 3, 4]}", "m", values,
                                          3, error, sizeof(error)));
    CHECK(!h3_image_vae_parse_float_array("{\"m\": [1, NaN, 3]}", "m", values,
                                          3, error, sizeof(error)));
    CHECK(!h3_image_vae_parse_float_array("{\"m\": [1, 2, 3]}", "n", values, 3,
                                          error, sizeof(error)));
    CHECK(error[0] != '\0');
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
    test_layout_validate();
    test_safetensors();
    test_rng_and_solver();
    test_rgb_resize();
    test_dit_row_conversions();
    test_image_schedule();
    test_image_canvas();
    test_image_layout();
    test_image_rope_inv_freq();
    test_image_timestep_features();
    test_image_patchify();
    test_image_qkv_interleave();
    test_dit_names();
    test_image_dit_bookkeeping();
    test_qkv_slab_plan();
    test_missing_weight_names();
    test_image_vae_tiles();
    test_image_unpack_2d();
    test_image_vae_config_parse();
    test_metal_probe();
    test_terminal_zoom();
    printf("ok: %d checks\n", tests_run);
    return 0;
}
