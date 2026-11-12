#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
/* The image VAE encoder (diffusers tensor names) on one frame: planar RGB
 * pixels [3,H,W] in [0,1], H and W multiples of 16, tiled as above. The output
 * holds the quant_conv moments [48,1,H/16,W/16], mean channels first,
 * unnormalized, with the tiles blended as moments. */
int h3_image_vae_encode_moments(const char *weight_directory,
                                const char *shader_source_path,
                                const float *pixels, int height, int width,
                                h3_video_encoder_progress progress,
                                void *progress_opaque, h3_video_latent *output,
                                char *error, size_t error_size);
/* The same encoder kept open across images: the weights load once. */
typedef struct h3_image_vae_encoder h3_image_vae_encoder;
h3_image_vae_encoder *h3_image_vae_encoder_open(const char *weight_directory,
                                                const char *shader_source_path,
                                                char *error, size_t error_size);
int h3_image_vae_encoder_moments(h3_image_vae_encoder *encoder,
                                 const float *pixels, int height, int width,
                                 h3_video_encoder_progress progress,
                                 void *progress_opaque, h3_video_latent *output,
                                 char *error, size_t error_size);
void h3_image_vae_encoder_close(h3_image_vae_encoder *encoder);
void h3_video_latent_free(h3_video_latent *latent);

#endif
