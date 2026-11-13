#ifndef H3_VAE_TILES_H
#define H3_VAE_TILES_H

#include "h3_host.h"
#include "h3_video_vae.h"

#include <stddef.h>

enum { H3_VAE_TILE_OVERLAP_MIN = 64 };

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} h3_vae_tile_axis;

/* Number of tiles of tile_pixels needed to cover extent with every overlap at
 * least H3_VAE_TILE_OVERLAP_MIN. */
int h3_vae_tile_count(int extent, int tile_pixels);
/* Lay tiles over one axis (extent a multiple of H3_VAE_SPATIAL_RATIO), with the
 * slack spread round-robin over the overlaps in whole 16 px units. */
int h3_vae_tile_axis_build(int extent, int tile_pixels,
                           h3_vae_tile_axis *axis, char *error,
                           size_t error_size);
void h3_vae_tile_axis_free(h3_vae_tile_axis *axis);
/* Blend row-major interleaved RGB tiles (frame, y, x, channel) linearly over
 * their overlaps into one frame_count x H x W image. */
int h3_vae_stitch_tiles(float **tiles, const h3_vae_tile_axis *y_axis,
                        const h3_vae_tile_axis *x_axis, int frame_count,
                        h3_video_frames *output,
                        char *error, size_t error_size);

#endif
