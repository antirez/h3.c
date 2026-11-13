#include "h3_dit_names.h"

#include <stdio.h>

static const h3_dit_names video_names = {
    .family = H3_DIT_FAMILY_VIDEO,
    .block_prefix = "blocks.%u.",
    .refiner_prefix = "token_refiner.blocks.%u.",
    .norm1 = "norm1.weight",
    .norm2 = "norm2.weight",
    .qkv = "attn.qkv_proj.weight",
    .q_norm = "attn.q_norm.weight",
    .k_norm = "attn.k_norm.weight",
    .out = "attn.out_proj.weight",
    .fc1 = "mlp.fc1.weight",
    .fc2 = "mlp.fc2.weight",
    .adaln_weight = "adaln_proj.linear.weight",
    .adaln_bias = "adaln_proj.linear.bias",
    .refiner_final_norm = "token_refiner.final_norm.weight",
    .condition_weight = "condition_proj.weight",
    .condition_bias = "condition_proj.bias",
    .time_in_weight = "time_embedder.proj_in.weight",
    .time_in_bias = "time_embedder.proj_in.bias",
    .time_out_weight = "time_embedder.proj_out.weight",
    .time_out_bias = "time_embedder.proj_out.bias",
    .patch_weight = "video_patch_proj.weight",
    .patch_bias = "video_patch_proj.bias",
    .audio_patch_weight = "audio_patch_proj.weight",
    .audio_patch_bias = "audio_patch_proj.bias",
    .final_norm = "final_layer.norm.weight",
    .final_adaln_weight = "final_layer.adaln_proj.linear.weight",
    .final_adaln_bias = "final_layer.adaln_proj.linear.bias",
    .head_weight = "final_layer.video_out.weight",
    .head_bias = "final_layer.video_out.bias",
    .audio_head_weight = "final_layer.audio_out.weight",
    .audio_head_bias = "final_layer.audio_out.bias",
    .rope_inv_freq = "rope.inv_freq",
};

static const h3_dit_names image_names = {
    .family = H3_DIT_FAMILY_IMAGE,
    .block_prefix = "transformer_blocks.%u.",
    .refiner_prefix = "token_refiner.refiner_blocks.%u.",
    .norm1 = "norm1.weight",
    .norm2 = "norm2.weight",
    .q = "attn.to_q.weight",
    .k = "attn.to_k.weight",
    .v = "attn.to_v.weight",
    .q_norm = "attn.norm_q.weight",
    .k_norm = "attn.norm_k.weight",
    .out = "attn.to_out.0.weight",
    .fc1 = "ff.net.0.proj.weight",
    .fc2 = "ff.net.2.weight",
    .adaln_weight = "adaln_proj.linear.weight",
    .adaln_bias = "adaln_proj.linear.bias",
    .refiner_final_norm = "token_refiner.final_norm.weight",
    .condition_weight = "context_embedder.weight",
    .condition_bias = "context_embedder.bias",
    .time_in_weight = "time_embedder.linear_1.weight",
    .time_in_bias = "time_embedder.linear_1.bias",
    .time_out_weight = "time_embedder.linear_2.weight",
    .time_out_bias = "time_embedder.linear_2.bias",
    .patch_weight = "proj_in.weight",
    .patch_bias = "proj_in.bias",
    .final_norm = "norm_out.norm.weight",
    .final_adaln_weight = "norm_out.linear.weight",
    .final_adaln_bias = "norm_out.linear.bias",
    .head_weight = "pdd_proj_out.%u.weight",
    .head_bias = "pdd_proj_out.%u.bias",
};

const h3_dit_names *h3_dit_names_for(h3_dit_family family) {
    switch (family) {
    case H3_DIT_FAMILY_VIDEO: return &video_names;
    case H3_DIT_FAMILY_IMAGE: return &image_names;
    }
    return NULL;
}

int h3_dit_name_format(char *out, size_t size, const char *pattern,
                       unsigned index) {
    if (!out || !size || !pattern) return 0;
    int written = snprintf(out, size, pattern, index);
    return written >= 0 && (size_t)written < size;
}

int h3_dit_block_name(char *out, size_t size, const h3_dit_names *names,
                      int refiner, unsigned block, const char *suffix) {
    if (!out || !size || !names || !suffix) return 0;
    char prefix[96];
    if (!h3_dit_name_format(prefix, sizeof(prefix),
                            refiner ? names->refiner_prefix :
                                      names->block_prefix, block)) return 0;
    int written = snprintf(out, size, "%s%s", prefix, suffix);
    return written >= 0 && (size_t)written < size;
}
