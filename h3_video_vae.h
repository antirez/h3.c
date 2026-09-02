#ifndef H3_VIDEO_VAE_H
#define H3_VIDEO_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int frames;
    int height;
    int width;
    /* Frame-major, row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_video_frames;

typedef void (*h3_video_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);

typedef struct h3_video_vae_decoder h3_video_vae_decoder;

/* Resident tiled decoder used by live denoising previews. It decodes one
 * representative middle frame per preview call and can then produce the final
 * complete video without loading the 9.7 GiB weight set again. */
h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size);
/* Reports block progress while decoding, which costs about as much as the
 * load did. Optional: a decoder without a sink decodes silently. */
void h3_video_vae_decoder_set_progress(h3_video_vae_decoder *decoder,
                                       h3_video_vae_progress progress,
                                       void *progress_opaque);

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size);
int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder);

/* Decode aligned H3 temporal chunks, using the released overlap/blend rules in
 * time and 256-pixel overlapping tiles in space. The two-token mode remains
 * available only for the diagnostic MLX fixture. */
int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_frames_free(h3_video_frames *frames);

#endif
