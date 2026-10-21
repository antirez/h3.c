#include "h3_motion_context.h"

#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_host.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONTEXT_MAGIC "H3CTX001"
#define VIDEO_CHANNELS 24
#define AUDIO_CHANNELS 32
#define AUDIO_RATE 32000
#define AUDIO_SAMPLES_PER_STEP 800
/* The visual VAE encodes 17-frame clips (1 + 4 x 4 latent steps); a shorter
 * final clip keeps the released ceil(T/4) steps, which is the same grid. */
#define ENCODER_CLIP_FRAMES 17
/* The soundtrack is read through the reference-audio reader, whose bound is
 * 15 s; a longer previous clip keeps its first 15 s, so chain from the
 * context file for those. */
#define AUDIO_READ_SAMPLES (AUDIO_RATE * 15)

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

typedef struct {
    int video_t;
    int latent_h;
    int latent_w;
    int audio_t;
    int frame_count;
} context_header;

static int read_header(FILE *file, context_header *header) {
    char magic[8];
    int32_t values[5];
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, CONTEXT_MAGIC, sizeof(magic)) != 0 ||
        fread(values, sizeof(values[0]), 5, file) != 5) return 0;
    header->video_t = values[0];
    header->latent_h = values[1];
    header->latent_w = values[2];
    header->audio_t = values[3];
    header->frame_count = values[4];
    return header->video_t > 0 && header->latent_h > 0 &&
           header->latent_w > 0 && header->audio_t > 0 &&
           header->frame_count > 0;
}

static int is_context_file(const char *path, context_header *header) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    int ok = read_header(file, header);
    fclose(file);
    return ok;
}

/* The pinned audio window ends at the join. A previous clip's audio grid is
 * rounded to the nearest step, so it overshoots or stops short of its last
 * frame by up to a third of a step; the window moves with it. */
static double audio_offset(int frames, int audio_t, double overhang) {
    double end = (double)llround(H3_FRAME_RESCALE * (double)frames + overhang);
    return end - (double)audio_t;
}

int h3_motion_context_plan(const char *path, int latent_h, int latent_w,
                           h3_motion_context *context,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!context || !path || !*path || latent_h < 2 || latent_w < 2) {
        fail(error, error_size, "invalid motion context arguments");
        return 0;
    }
    memset(context, 0, sizeof(*context));
    context->frames = H3_MOTION_CONTEXT_FRAMES;
    context->steps = h3_video_head_steps(context->frames);
    context->latent_h = latent_h;
    context->latent_w = latent_w;
    if (!context->steps) {
        fail(error, error_size, "motion context frame count is off the latent grid");
        return 0;
    }
    FILE *probe = fopen(path, "rb");
    if (!probe) {
        fail(error, error_size, "cannot open motion context %s: %s", path,
             strerror(errno));
        return 0;
    }
    fclose(probe);
    context_header header;
    if (is_context_file(path, &header)) {
        context->from_latents = 1;
        if (header.latent_h != latent_h || header.latent_w != latent_w) {
            fail(error, error_size,
                 "motion context %s is %dx%d but this clip is %dx%d: a latent "
                 "cannot be resized, render the previous clip at this size",
                 path, header.latent_w * H3_VAE_SPATIAL_RATIO,
                 header.latent_h * H3_VAE_SPATIAL_RATIO,
                 latent_w * H3_VAE_SPATIAL_RATIO,
                 latent_h * H3_VAE_SPATIAL_RATIO);
            return 0;
        }
        int start = header.video_t - context->steps;
        if (start < 0 || start % 5) {
            fail(error, error_size,
                 "motion context %s has %d latent steps; the %d-step tail must "
                 "start on the 1,4,4,4,4 cycle", path, header.video_t,
                 context->steps);
            return 0;
        }
        context->audio_t = (int)llround((double)context->frames *
                                        H3_AUDIO_LATENT_FPS / H3_FPS);
        if (context->audio_t > header.audio_t) {
            fail(error, error_size, "motion context %s has too little audio", path);
            return 0;
        }
        double overhang = (double)header.audio_t -
                          H3_FRAME_RESCALE * (double)header.frame_count;
        if (!(overhang > -0.5 && overhang < 0.5)) {
            fprintf(stderr, "h3: motion context audio grid is unexpected "
                    "(%d steps for %d frames); assuming no overhang\n",
                    header.audio_t, header.frame_count);
            overhang = 0.0;
        }
        context->audio_offset = audio_offset(context->frames, context->audio_t,
                                             overhang);
        return 1;
    }
    /* A video: the audio tail is re-encoded, padded to whole steps. */
    long samples = lround((double)context->frames * AUDIO_RATE / H3_FPS);
    context->audio_t = (int)((samples + AUDIO_SAMPLES_PER_STEP - 1) /
                             AUDIO_SAMPLES_PER_STEP);
    context->audio_offset = audio_offset(context->frames, context->audio_t, 0.0);
    return 1;
}

static int slice_context_file(h3_motion_context *context, const char *path,
                              char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    context_header header;
    if (!file || !read_header(file, &header)) {
        if (file) fclose(file);
        fail(error, error_size, "cannot read motion context %s", path);
        return 0;
    }
    size_t plane = (size_t)header.latent_h * (size_t)header.latent_w;
    size_t video_count = (size_t)VIDEO_CHANNELS * (size_t)header.video_t * plane;
    size_t audio_count = (size_t)AUDIO_CHANNELS * 2 * (size_t)header.audio_t;
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    int ok = video && audio &&
             fread(video, sizeof(*video), video_count, file) == video_count &&
             fread(audio, sizeof(*audio), audio_count, file) == audio_count;
    fclose(file);
    if (!ok) {
        free(video);
        free(audio);
        fail(error, error_size, "motion context %s is truncated", path);
        return 0;
    }
    int steps = context->steps, start = header.video_t - steps;
    size_t tail_count = (size_t)VIDEO_CHANNELS * (size_t)steps * plane;
    float *tail = malloc(tail_count * sizeof(*tail));
    size_t row_elements = (size_t)steps * plane / 4 * 96;
    float *rows = malloc(row_elements * sizeof(*rows));
    int audio_t = context->audio_t;
    size_t audio_tail_count = (size_t)AUDIO_CHANNELS * 2 * (size_t)audio_t;
    float *audio_tail = malloc(audio_tail_count * sizeof(*audio_tail));
    size_t audio_row_elements = audio_tail_count;
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    ok = tail && rows && audio_tail && audio_rows;
    if (ok) {
        for (int channel = 0; channel < VIDEO_CHANNELS; channel++)
            memcpy(tail + (size_t)channel * steps * plane,
                   video + ((size_t)channel * header.video_t + start) * plane,
                   (size_t)steps * plane * sizeof(*tail));
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            for (int stereo = 0; stereo < 2; stereo++)
                memcpy(audio_tail + ((size_t)channel * 2 + stereo) * audio_t,
                       audio + ((size_t)channel * 2 + stereo) * header.audio_t +
                           (header.audio_t - audio_t),
                       (size_t)audio_t * sizeof(*audio_tail));
        ok = h3_dit_patchify_video(tail, VIDEO_CHANNELS, steps, header.latent_h,
                                   header.latent_w, rows, row_elements) &&
             h3_dit_pack_audio(audio_tail, AUDIO_CHANNELS, audio_t, audio_rows,
                               audio_row_elements);
    }
    free(video);
    free(audio);
    free(tail);
    free(audio_tail);
    if (!ok) {
        free(rows);
        free(audio_rows);
        fail(error, error_size, "cannot pack motion context rows");
        return 0;
    }
    context->video_rows = rows;
    context->video_elements = row_elements;
    context->audio_rows = audio_rows;
    context->audio_elements = audio_row_elements;
    return 1;
}

static int encode_video_file(h3_motion_context *context, const char *path,
                             int render_width, int render_height,
                             const char *vae_path, const char *audio_vae_path,
                             const char *shader_source_path,
                             h3_video_encoder_progress video_progress,
                             void *video_opaque,
                             h3_audio_vae_progress audio_progress,
                             void *audio_opaque,
                             char *error, size_t error_size) {
    float *pixels = NULL;
    int frames = 0, clip_frames = 0;
    if (!h3_ffmpeg_read_video_tail_f32(path, render_width, render_height,
                                       context->frames, &pixels, &frames,
                                       &clip_frames, error, error_size))
        return 0;
    if (frames != context->frames) {
        free(pixels);
        fail(error, error_size, "motion context %s has only %d frames, %d are pinned",
             path, frames, context->frames);
        return 0;
    }
    size_t area = (size_t)render_width * (size_t)render_height;
    size_t plane = (size_t)context->latent_h * (size_t)context->latent_w;
    size_t latent_count = (size_t)VIDEO_CHANNELS * (size_t)context->steps * plane;
    float *latent = malloc(latent_count * sizeof(*latent));
    float *clip = malloc((size_t)3 * ENCODER_CLIP_FRAMES * area * sizeof(*clip));
    int ok = latent && clip;
    if (!ok) fail(error, error_size, "out of memory for the motion context head");
    int step = 0;
    for (int first = 0; ok && first < frames; first += ENCODER_CLIP_FRAMES) {
        int count = frames - first < ENCODER_CLIP_FRAMES ?
                    frames - first : ENCODER_CLIP_FRAMES;
        for (int channel = 0; channel < 3; channel++)
            memcpy(clip + (size_t)channel * count * area,
                   pixels + ((size_t)channel * frames + first) * area,
                   (size_t)count * area * sizeof(*clip));
        h3_video_latent part;
        memset(&part, 0, sizeof(part));
        ok = h3_video_vae_encode(vae_path, shader_source_path, clip, count,
                                 render_height, render_width, video_progress,
                                 video_opaque, &part, error, error_size);
        if (ok && (part.time != h3_video_encoder_latent_t(count) ||
                   part.height != context->latent_h ||
                   part.width != context->latent_w ||
                   step + part.time > context->steps)) {
            fail(error, error_size, "motion context encode produced unexpected latent geometry");
            ok = 0;
        }
        if (ok) {
            for (int channel = 0; channel < VIDEO_CHANNELS; channel++)
                memcpy(latent + ((size_t)channel * context->steps + step) * plane,
                       part.values + (size_t)channel * part.time * plane,
                       (size_t)part.time * plane * sizeof(*latent));
            step += part.time;
        }
        h3_video_latent_free(&part);
    }
    if (ok && step != context->steps) {
        fail(error, error_size, "motion context encode covered %d steps, expected %d",
             step, context->steps);
        ok = 0;
    }
    free(clip);
    free(pixels);
    size_t row_elements = (size_t)context->steps * plane / 4 * 96;
    float *rows = ok ? malloc(row_elements * sizeof(*rows)) : NULL;
    if (ok && (!rows || !h3_dit_patchify_video(latent, VIDEO_CHANNELS,
                                               context->steps, context->latent_h,
                                               context->latent_w, rows,
                                               row_elements))) {
        fail(error, error_size, "cannot patchify the motion context head");
        ok = 0;
    }
    free(latent);
    if (!ok) {
        free(rows);
        return 0;
    }
    /* The pinned sound is the last audio_t steps of the clip's own audio
     * grid, round(frames x 5/3) steps of 800 samples, the window a context
     * file pins. The stream runs past that grid (h3 decodes a third of a
     * step beyond the last frame, and an AAC track decodes to whole codec
     * frames), so the stream's own tail sat up to a step off the grid and
     * its rows matched the true latents at cosine 0.71. */
    float *pcm = NULL;
    int samples = 0;
    if (!h3_ffmpeg_read_audio_f32(path, AUDIO_READ_SAMPLES, 1,
                                  &pcm, &samples, error, error_size)) {
        free(pcm);
        free(rows);
        return 0;
    }
    long grid_end = llround(H3_FRAME_RESCALE * (double)clip_frames) *
                    AUDIO_SAMPLES_PER_STEP;
    long want = (long)context->audio_t * AUDIO_SAMPLES_PER_STEP;
    long start = grid_end - want;
    if (grid_end > samples && samples >= AUDIO_READ_SAMPLES) {
        free(pcm);
        free(rows);
        fail(error, error_size, "motion context %s is longer than the %d s "
             "of sound a video can carry; chain from its context file", path,
             AUDIO_READ_SAMPLES / AUDIO_RATE);
        return 0;
    }
    if (start >= samples) {
        free(pcm);
        free(rows);
        fail(error, error_size, "motion context %s has %.3f s of audio, none "
             "under its last %d frames", path, (double)samples / AUDIO_RATE,
             context->frames);
        return 0;
    }
    long have = (samples < grid_end ? samples : grid_end) - start;
    if (have < want)
        fprintf(stderr, "h3: motion context %s has %.3f s of audio under its "
                "last %d frames, %.3f s are pinned; the rest is silence\n",
                path, (double)have / AUDIO_RATE, context->frames,
                (double)want / AUDIO_RATE);
    float *tail = calloc((size_t)2 * (size_t)want, sizeof(*tail));
    if (!tail) {
        free(pcm);
        free(rows);
        fail(error, error_size, "out of memory for the motion context audio");
        return 0;
    }
    for (int channel = 0; channel < 2; channel++)
        memcpy(tail + (size_t)channel * want,
               pcm + (size_t)channel * samples + start,
               (size_t)have * sizeof(*tail));
    free(pcm);
    h3_audio_latent audio;
    memset(&audio, 0, sizeof(audio));
    ok = h3_audio_vae_encode(audio_vae_path, shader_source_path, tail, (int)want,
                             audio_progress, audio_opaque, &audio,
                             error, error_size);
    free(tail);
    if (ok && (audio.channels != AUDIO_CHANNELS || audio.stereo != 2 ||
               audio.length != context->audio_t)) {
        fail(error, error_size, "motion context audio encode produced %d steps, expected %d",
             audio.length, context->audio_t);
        ok = 0;
    }
    size_t audio_row_elements = (size_t)AUDIO_CHANNELS * 2 * (size_t)context->audio_t;
    float *audio_rows = ok ? malloc(audio_row_elements * sizeof(*audio_rows)) : NULL;
    if (ok && (!audio_rows || !h3_dit_pack_audio(audio.values, AUDIO_CHANNELS,
                                                 context->audio_t, audio_rows,
                                                 audio_row_elements))) {
        fail(error, error_size, "cannot pack the motion context audio");
        ok = 0;
    }
    h3_audio_latent_free(&audio);
    if (!ok) {
        free(rows);
        free(audio_rows);
        return 0;
    }
    context->video_rows = rows;
    context->video_elements = row_elements;
    context->audio_rows = audio_rows;
    context->audio_elements = audio_row_elements;
    return 1;
}

int h3_motion_context_encode(h3_motion_context *context, const char *path,
                             int render_width, int render_height,
                             const char *vae_path, const char *audio_vae_path,
                             const char *shader_source_path,
                             h3_video_encoder_progress video_progress,
                             void *video_opaque,
                             h3_audio_vae_progress audio_progress,
                             void *audio_opaque,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!context || !path || !context->steps) {
        fail(error, error_size, "motion context was not planned");
        return 0;
    }
    if (context->from_latents)
        return slice_context_file(context, path, error, error_size);
    return encode_video_file(context, path, render_width, render_height,
                             vae_path, audio_vae_path, shader_source_path,
                             video_progress, video_opaque, audio_progress,
                             audio_opaque, error, error_size);
}

void h3_motion_context_free(h3_motion_context *context) {
    if (!context) return;
    free(context->video_rows);
    free(context->audio_rows);
    memset(context, 0, sizeof(*context));
}

int h3_motion_context_save(const char *path, const float *video, int video_t,
                           int latent_h, int latent_w, const float *audio,
                           int audio_t, int frame_count,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !video || !audio || video_t < 1 || latent_h < 1 ||
        latent_w < 1 || audio_t < 1 || frame_count < 1) {
        fail(error, error_size, "invalid motion context save arguments");
        return 0;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        fail(error, error_size, "cannot write %s: %s", path, strerror(errno));
        return 0;
    }
    int32_t values[5] = {video_t, latent_h, latent_w, audio_t, frame_count};
    size_t video_count = (size_t)VIDEO_CHANNELS * (size_t)video_t *
                         (size_t)latent_h * (size_t)latent_w;
    size_t audio_count = (size_t)AUDIO_CHANNELS * 2 * (size_t)audio_t;
    int ok = fwrite(CONTEXT_MAGIC, 1, 8, file) == 8 &&
             fwrite(values, sizeof(values[0]), 5, file) == 5 &&
             fwrite(video, sizeof(*video), video_count, file) == video_count &&
             fwrite(audio, sizeof(*audio), audio_count, file) == audio_count;
    if (fclose(file) != 0) ok = 0;
    if (!ok) fail(error, error_size, "cannot write motion context %s", path);
    return ok;
}

int h3_motion_context_trim_pcm(const float *pcm, int channels, int samples,
                               int sample_rate, int frame_count,
                               int trim_frames, float **output,
                               int *output_samples) {
    if (!pcm || !output || !output_samples || channels < 1 || samples < 1 ||
        sample_rate < 1 || trim_frames < 0 || trim_frames >= frame_count)
        return 0;
    long cut = lround((double)trim_frames * sample_rate / H3_FPS);
    long want = lround((double)(frame_count - trim_frames) * sample_rate / H3_FPS);
    if (cut >= samples || want < 1) return 0;
    long have = samples - cut;
    long keep = have < want ? have : want;
    float *trimmed = calloc((size_t)channels * (size_t)want, sizeof(*trimmed));
    if (!trimmed) return 0;
    for (int channel = 0; channel < channels; channel++)
        memcpy(trimmed + (size_t)channel * want,
               pcm + (size_t)channel * samples + cut,
               (size_t)keep * sizeof(*trimmed));
    *output = trimmed;
    *output_samples = (int)want;
    return 1;
}
