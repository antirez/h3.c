#ifndef H3_CHECKPOINT_H
#define H3_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t signature;
    uint64_t seed;
    uint32_t total_steps;
    uint32_t next_step;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t frame_count;
    uint64_t video_elements;
    uint64_t audio_elements;
    double denoise_seconds;
} h3_checkpoint_info;

/* Stable, non-secret compatibility fingerprint for the canonical run key. */
uint64_t h3_checkpoint_signature(const char *text);

/* Checkpoints are written atomically with mode 0600 and contain raw F32 video
 * and audio latents. The checksum detects corruption; it is not an
 * authenticity mechanism for untrusted files. */
int h3_checkpoint_save(const char *path, const h3_checkpoint_info *info,
                       const float *video, const float *audio,
                       char *error, size_t error_size);

/* expected describes every immutable run field. next_step and
 * denoise_seconds are returned from the file and ignored in expected. */
int h3_checkpoint_load(const char *path,
                       const h3_checkpoint_info *expected,
                       h3_checkpoint_info *actual,
                       float **video, float **audio,
                       char *error, size_t error_size);

#endif
