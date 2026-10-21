#ifndef H3_MOTION_CONTEXT_H
#define H3_MOTION_CONTEXT_H

#include "h3_audio_vae.h"
#include "h3_video_encoder.h"

#include <stddef.h>

/* Motion context chains one generation onto the previous one: the previous
 * clip's last H3_MOTION_CONTEXT_FRAMES pixel frames (7 latent steps on the
 * 1,4,4,4,4 grid) are pinned as visual condition rows on the target's own
 * first step times, and its last 0.917 s of sound as audio condition rows
 * that end at the join. The pinned head is trimmed from the delivered clip,
 * so clips concatenate. The previous clip is either the context file a run
 * wrote with save_context (its final latents, bit-exact) or a finished
 * video, whose tail is decoded and re-encoded with the released VAEs. */
#define H3_MOTION_CONTEXT_FRAMES 22

typedef struct {
    int frames;          /* pinned pixel frames */
    int steps;           /* pinned latent steps */
    int latent_h;
    int latent_w;
    int audio_t;         /* pinned audio latent steps */
    double audio_offset; /* first pinned audio step, relative to the target origin */
    int from_latents;    /* 1: context file, 0: video file */
    float *video_rows;   /* patchified visual condition rows, width 96 */
    size_t video_elements;
    float *audio_rows;   /* packed audio condition rows, width 32 */
    size_t audio_elements;
} h3_motion_context;

/* Geometry only, no GPU: how many rows the head will take and where the
 * audio window sits. latent_h/latent_w are the target canvas. */
int h3_motion_context_plan(const char *path, int latent_h, int latent_w,
                           h3_motion_context *context,
                           char *error, size_t error_size);
/* Fill the condition rows after a successful plan. A context file is sliced
 * on the CPU; a video file is decoded with FFmpeg and encoded with the
 * released visual and audio VAEs. */
int h3_motion_context_encode(h3_motion_context *context, const char *path,
                             int render_width, int render_height,
                             const char *vae_path, const char *audio_vae_path,
                             const char *shader_source_path,
                             h3_video_encoder_progress video_progress,
                             void *video_opaque,
                             h3_audio_vae_progress audio_progress,
                             void *audio_opaque,
                             char *error, size_t error_size);
void h3_motion_context_free(h3_motion_context *context);

/* Write a context file: the generation's final normalized video latent
 * [24,video_t,latent_h,latent_w] and audio latent [32,2,audio_t]. */
int h3_motion_context_save(const char *path, const float *video, int video_t,
                           int latent_h, int latent_w, const float *audio,
                           int audio_t, int frame_count,
                           char *error, size_t error_size);

/* Drop the pinned head from channel-major PCM: cut trim_frames worth of
 * samples at the front and size the rest to the remaining frames exactly
 * (zero-padded when the audio grid stopped short). The caller owns *output. */
int h3_motion_context_trim_pcm(const float *pcm, int channels, int samples,
                               int sample_rate, int frame_count,
                               int trim_frames, float **output,
                               int *output_samples);

#endif
