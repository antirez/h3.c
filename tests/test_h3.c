#include "h3_host.h"
#include "h3_dit.h"
#include "h3_metal.h"
#include "h3_safetensors.h"
#include "h3_terminal.h"
#include "h3_ffmpeg.h"
#include "h3_motion_context.h"

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
    h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0, 0, 0, 0.0};
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
                           keyframes, 2, NULL, 0, 0, 0, 0.0};
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

static void test_layout_motion_context(void) {
    /* 22 pinned frames = 7 latent steps; 37 audio steps ending at the join. */
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56, NULL, 0, NULL, 0, 7, 37, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 10108);
    const size_t bounds[][2] = {
        {0, 128}, {128, 2963}, {2963, 3037}, {3037, 3223}, {3223, 10108}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_REF_AUDIO, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 5);
    CHECK(layout.img_cond_rows == 2835 && layout.audio_cond_rows == 74);
    CHECK(layout.img_target_rows == 6885 && layout.audio_target_rows == 186);
    /* Every pinned row shares its (t, h, w) with the target row it stands
     * in for: the head occupies the target's own first seven steps. */
    int same = 1;
    for (size_t row = 0; row < 2835; row++) {
        const h3_position *pinned = &layout.positions[128 + row];
        const h3_position *target = &layout.positions[3223 + row];
        if (pinned->t != target->t || pinned->h != target->h ||
            pinned->w != target->w) same = 0;
    }
    CHECK(same);
    CHECK(layout.positions[128].t == 128.0);
    CHECK(close_enough(layout.positions[128 + 6 * 405].t, 158.0, 1e-12));
    /* The audio window: 37 steps from the origin, stereo rows on the target
     * audio w positions, identical to the first 37 target audio steps. */
    same = 1;
    for (size_t step = 0; step < 37; step++) {
        for (int stereo = 0; stereo < 2; stereo++) {
            const h3_position *pinned =
                &layout.positions[2963 + (size_t)stereo * 37 + step];
            const h3_position *target =
                &layout.positions[3037 + (size_t)stereo * 93 + step];
            if (pinned->t != target->t || pinned->w != target->w) same = 0;
        }
    }
    CHECK(same);
    h3_layout_free(&layout);
    /* A negative offset (a previous clip whose audio grid stopped short of
     * its last frame) starts one step before the origin. */
    spec.head_audio_offset = -1.0;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.positions[2963].t == 127.0);
    CHECK(layout.positions[2963 + 36].t == 163.0);
    h3_layout_free(&layout);
    /* The pinned run must leave room to generate, and it replaces the
     * first-frame keyframe rather than stacking on it. */
    spec.head_audio_offset = 0.0;
    spec.head_steps = 17;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    spec.head_steps = 7;
    int first = 0;
    spec.keyframes = &first;
    spec.keyframe_count = 1;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    int last = 55;
    spec.keyframes = &last;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.img_cond_rows == 2835 + 405);
    h3_layout_free(&layout);
}

static void test_motion_context_file(void) {
    CHECK(h3_video_head_steps(22) == 7 && h3_video_head_steps(5) == 2 &&
          h3_video_head_steps(39) == 12 && h3_video_head_steps(56) == 17 &&
          h3_video_head_steps(1) == 1 && h3_video_head_steps(4) == 0 &&
          h3_video_head_steps(0) == 0);
    /* A 39-frame clip: 12 latent steps on a 4x6 canvas, 65 audio steps. */
    const int video_t = 12, lh = 4, lw = 6, audio_t = 65, frame_count = 39;
    size_t video_count = (size_t)24 * video_t * lh * lw;
    size_t audio_count = (size_t)32 * 2 * audio_t;
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    for (size_t index = 0; index < video_count; index++)
        video[index] = (float)(index % 977) * 0.01f;
    for (size_t index = 0; index < audio_count; index++)
        audio[index] = (float)(index % 311) * -0.02f;
    char path[] = "/tmp/h3-motion-context-test.h3ctx";
    char error[256];
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, audio_t,
                                 frame_count, error, sizeof(error)));
    h3_motion_context context;
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.from_latents && context.frames == 22 && context.steps == 7);
    CHECK(context.audio_t == 37 && context.audio_offset == 0.0);
    CHECK(h3_motion_context_encode(&context, path, lw * 16, lh * 16, NULL,
                                   NULL, NULL, NULL, NULL, NULL, NULL,
                                   error, sizeof(error)));
    /* The rows are the patchified last 7 steps and the packed last 37 audio
     * steps, element for element. */
    size_t plane = (size_t)lh * lw;
    float *tail = malloc((size_t)24 * 7 * plane * sizeof(*tail));
    for (int channel = 0; channel < 24; channel++)
        memcpy(tail + (size_t)channel * 7 * plane,
               video + ((size_t)channel * video_t + 5) * plane,
               7 * plane * sizeof(*tail));
    size_t row_elements = 7 * plane / 4 * 96;
    float *rows = malloc(row_elements * sizeof(*rows));
    CHECK(h3_dit_patchify_video(tail, 24, 7, lh, lw, rows, row_elements));
    CHECK(context.video_elements == row_elements &&
          memcmp(context.video_rows, rows, row_elements * sizeof(*rows)) == 0);
    float *audio_tail = malloc((size_t)32 * 2 * 37 * sizeof(*audio_tail));
    for (int channel = 0; channel < 32; channel++)
        for (int stereo = 0; stereo < 2; stereo++)
            memcpy(audio_tail + ((size_t)channel * 2 + stereo) * 37,
                   audio + ((size_t)channel * 2 + stereo) * audio_t + 28,
                   37 * sizeof(*audio_tail));
    float *audio_rows = malloc((size_t)32 * 2 * 37 * sizeof(*audio_rows));
    CHECK(h3_dit_pack_audio(audio_tail, 32, 37, audio_rows, (size_t)32 * 2 * 37));
    CHECK(context.audio_elements == (size_t)32 * 2 * 37 &&
          memcmp(context.audio_rows, audio_rows,
                 context.audio_elements * sizeof(*audio_rows)) == 0);
    h3_motion_context_free(&context);
    /* A 107-frame clip allocates 178 audio steps for 178.33 frames' worth:
     * the window ends a third of a step early, so it starts one step before
     * the origin (round(36.33) - 37). */
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, 178,
                                 107, error, sizeof(error)));
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.audio_offset == -1.0);
    /* A 124-frame clip overshoots by a third: round(37.0) - 37 = 0. */
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, 207,
                                 124, error, sizeof(error)));
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.audio_offset == 0.0);
    /* Another canvas is refused; a tail off the cycle is refused. */
    CHECK(!h3_motion_context_plan(path, lh, lw + 2, &context, error, sizeof(error)));
    CHECK(h3_motion_context_save(path, video, 11, lh, lw, audio, audio_t,
                                 frame_count, error, sizeof(error)));
    CHECK(!h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    remove(path);
    /* A video file plans a 37-step window from the origin without reading it. */
    CHECK(!h3_motion_context_plan("/nonexistent/clip.mp4", lh, lw, &context,
                                  error, sizeof(error)));
    FILE *stub = fopen("/tmp/h3-motion-context-test.mp4", "wb");
    CHECK(stub && fputs("not a context file", stub) >= 0 && fclose(stub) == 0);
    CHECK(h3_motion_context_plan("/tmp/h3-motion-context-test.mp4", lh, lw,
                                 &context, error, sizeof(error)));
    CHECK(!context.from_latents && context.audio_t == 37 &&
          context.audio_offset == 0.0);
    remove("/tmp/h3-motion-context-test.mp4");
    /* Trim: 22 of 107 frames at 32 kHz cuts 29333 samples and sizes the rest
     * to 85 frames (113333 samples), zero-padding a short audio grid. */
    int samples = 178 * 800;
    float *pcm = malloc((size_t)2 * samples * sizeof(*pcm));
    for (int index = 0; index < 2 * samples; index++) pcm[index] = 1.0f;
    float *trimmed = NULL;
    int trimmed_samples = 0;
    CHECK(h3_motion_context_trim_pcm(pcm, 2, samples, 32000, 107, 22,
                                     &trimmed, &trimmed_samples));
    CHECK(trimmed_samples == 113333);
    CHECK(trimmed[0] == 1.0f && trimmed[113066] == 1.0f &&
          trimmed[113067] == 0.0f && trimmed[113333] == 1.0f &&
          trimmed[2 * 113333 - 1] == 0.0f);
    CHECK(!h3_motion_context_trim_pcm(pcm, 2, samples, 32000, 22, 22,
                                      &trimmed, &trimmed_samples));
    free(trimmed); free(pcm); free(audio_rows); free(audio_tail); free(rows);
    free(tail); free(video); free(audio);
}

static void test_layout_ref2va(void) {
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3, 0, 0, 0.0};
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

static void test_layout_motion_context_refs(void) {
    /* The pinned head and its audio window sit on the target timeline, which
     * starts after the reference spans (1 + 48 + 80 steps here), while the
     * references themselves keep their slots right after the text. */
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3, 7, 37, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 11196);
    const size_t bounds[][2] = {
        {0, 192}, {192, 3027}, {3027, 3101}, {3101, 3197}, {3197, 3293},
        {3293, 3965}, {3965, 4125}, {4125, 4311}, {4311, 11196}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_AUDIO,
        H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 9);
    CHECK(layout.img_cond_rows == 2835 + 768);
    CHECK(layout.audio_cond_rows == 74 + 256);
    CHECK(layout.img_target_rows == 6885 && layout.audio_target_rows == 186);
    CHECK(layout.positions[192].t == 321.0);
    CHECK(layout.positions[3101].t == 192.0);
    CHECK(layout.positions[4125].t == 321.0);
    int same = 1;
    for (size_t row = 0; row < 2835; row++) {
        const h3_position *pinned = &layout.positions[192 + row];
        const h3_position *target = &layout.positions[4311 + row];
        if (pinned->t != target->t || pinned->h != target->h ||
            pinned->w != target->w) same = 0;
    }
    CHECK(same);
    same = 1;
    for (size_t step = 0; step < 37; step++) {
        for (int stereo = 0; stereo < 2; stereo++) {
            const h3_position *pinned =
                &layout.positions[3027 + (size_t)stereo * 37 + step];
            const h3_position *target =
                &layout.positions[4125 + (size_t)stereo * 93 + step];
            if (pinned->t != target->t || pinned->w != target->w) same = 0;
        }
    }
    CHECK(same);
    h3_layout_free(&layout);
}

static void test_layout_hybrid(void) {
    /* Frame anchors with references: the anchors lead the sequence as in
     * the FL2VA layout but are timed from the target origin, after the
     * reference spans, where the target rows they stand in for sit. */
    int keyframes[] = {0, 55};
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           keyframes, 2, references, 3, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 9097);
    const size_t bounds[][2] = {
        {0, 192}, {192, 597}, {597, 1002}, {1002, 1098}, {1098, 1194},
        {1194, 1866}, {1866, 2026}, {2026, 2212}, {2212, 9097}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_AUDIO,
        H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 9);
    CHECK(layout.img_cond_rows == 810 + 768);
    CHECK(layout.audio_cond_rows == 256);
    CHECK(layout.positions[192].t == layout.positions[2212].t);
    CHECK(layout.positions[192].t == 321.0);
    CHECK(close_enough(layout.positions[597].t, 321.0 + 55.0 * 5.0 / 3.0, 1e-9));
    CHECK(layout.positions[1002].t == 192.0);
    h3_layout_free(&layout);
    /* A pinned head still replaces the first-frame anchor. */
    spec.head_steps = 7;
    spec.head_audio_t = 37;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    spec.keyframes = &keyframes[1];
    spec.keyframe_count = 1;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.img_cond_rows == 2835 + 405 + 768);
    CHECK(layout.segments[1].kind == H3_SEG_COND &&
          layout.segments[2].kind == H3_SEG_REF_AUDIO &&
          layout.segments[3].kind == H3_SEG_COND &&
          layout.segments[4].kind == H3_SEG_REF_IMAGE);
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

/* The tail reader keeps the last N decoded frames of a stream it reads
 * once; checked against the bounded reader on a synthetic 30-frame clip. */
static void test_ffmpeg_video_tail(void) {
    const char *ffmpeg = getenv("H3_FFMPEG");
    if (!ffmpeg || !*ffmpeg) {
        printf("skip: FFmpeg video tail test (set H3_FFMPEG)\n");
        return;
    }
    enum { FRAMES = 30, W = 64, H = 48, RATE = 32000, SAMPLES = RATE };
    size_t frame_bytes = (size_t)W * H * 3;
    uint8_t *frames = malloc(FRAMES * frame_bytes);
    float *pcm = calloc(2 * SAMPLES, sizeof(*pcm));
    CHECK(frames && pcm);
    /* Each frame a flat shade so lossy encoding keeps them apart. */
    for (int frame = 0; frame < FRAMES; frame++)
        memset(frames + (size_t)frame * frame_bytes, 16 + frame * 7, frame_bytes);
    char path[] = "/tmp/h3-ffmpeg-tail-test.mp4";
    char error[256];
    CHECK(h3_ffmpeg_write_av_rgb24_f32(path, frames, FRAMES, W, H, 24, pcm,
                                       SAMPLES, 2, RATE, error, sizeof(error)));
    float *whole = NULL, *tail = NULL;
    int whole_frames = 0, tail_frames = 0, total_frames = 0;
    CHECK(h3_ffmpeg_read_video_f32(path, W, H, FRAMES, &whole, &whole_frames,
                                   error, sizeof(error)));
    CHECK(h3_ffmpeg_read_video_tail_f32(path, W, H, 22, &tail, &tail_frames,
                                        &total_frames, error, sizeof(error)));
    /* The bounded reader snaps 30 frames down to 22 (frames 0..21); the
     * tail is frames 8..29, so the 14 shared frames must match exactly. */
    CHECK(tail_frames == 22 && whole_frames == 22 && total_frames == FRAMES);
    size_t area = (size_t)W * H;
    int same = whole && tail;
    for (int channel = 0; same && channel < 3; channel++)
        for (int t = 0; same && t < 14; t++)
            if (memcmp(tail + ((size_t)channel * 22 + t) * area,
                       whole + ((size_t)channel * 22 + 8 + t) * area,
                       area * sizeof(float)) != 0) same = 0;
    CHECK(same);
    /* Shades rise 7 levels a frame: the last tail frame is the brightest. */
    CHECK(tail[21 * area] > tail[0] + 100.0f / 255.0f);
    float *more = NULL;
    int more_frames = 0;
    CHECK(h3_ffmpeg_read_video_tail_f32(path, W, H, 40, &more, &more_frames,
                                        NULL, error, sizeof(error)));
    CHECK(more_frames == FRAMES);
    unlink(path);
    free(whole); free(tail); free(more); free(frames); free(pcm);
}

int main(void) {
    test_temporal_and_canvas();
    test_schedule();
    test_dit_reuse_schedule();
    test_layout_tiny();
    test_layout_fl2va();
    test_layout_motion_context();
    test_motion_context_file();
    test_layout_ref2va();
    test_layout_motion_context_refs();
    test_layout_hybrid();
    test_safetensors();
    test_rng_and_solver();
    test_rgb_resize();
    test_dit_row_conversions();
    test_metal_probe();
    test_terminal_zoom();
    test_ffmpeg_video_tail();
    printf("ok: %d checks\n", tests_run);
    return 0;
}
