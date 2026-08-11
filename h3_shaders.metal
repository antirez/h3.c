#include <metal_stdlib>
#ifdef H3_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef H3_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

inline float h3_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort h3_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

inline float4 h3_bf16x4_to_f32(ushort4 value) {
    return as_type<float4>(uint4(value) << 16);
}

inline ushort4 h3_f32x4_to_bf16(float4 value) {
    uint4 bits = as_type<uint4>(value);
    bits += uint4(0x7fffu) + ((bits >> 16) & uint4(1u));
    return ushort4(bits >> 16);
}

struct linear_args {
    uint rows;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

struct int8_quant_args {
    uint rows;
    uint columns;
    float clip;
};

struct int8_head_major_quant_args {
    uint rows;
    uint padded_rows;
    uint heads;
    uint head_dim;
    float clip;
};

struct int8_group_quant_args {
    uint rows;
    uint columns;
    uint group_size;
    uint groups;
};

kernel void h3_linear_f32(device const float *input [[buffer(0)]],
                          device const float *weight [[buffer(1)]],
                          device const float *bias [[buffer(2)]],
                          device float *output [[buffer(3)]],
                          constant linear_args &args [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? bias[column] : 0.0f;
    device const float *x = input + row * args.input_dim;
    device const float *w = weight + column * args.input_dim;
    for (uint k = 0; k < args.input_dim; k++) sum = fma(x[k], w[k], sum);
    output[row * args.output_dim + column] = sum;
}

kernel void h3_linear_f32_tiled(device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device float *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = sum;
}

kernel void h3_linear_f32_tiled_bf16(
                                device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device ushort *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
}

kernel void h3_linear_f32_tiled_bf16_map(
                                device const float *input [[buffer(0)]],
                                device const float *weight [[buffer(1)]],
                                device const float *bias [[buffer(2)]],
                                device ushort *output [[buffer(3)]],
                                constant linear_args &args [[buffer(4)]],
                                device const uint *row_map [[buffer(5)]],
                                uint2 tid [[thread_position_in_threadgroup]],
                                uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    threadgroup uint output_rows[16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    if (tid.x == 0)
        output_rows[tid.y] = row < args.rows ? row_map[row] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[output_rows[tid.y] * args.output_dim + column] =
            h3_f32_to_bf16(sum);
}

kernel void h3_silu_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = input[gid];
    output[gid] = value / (1.0f + exp(-value));
}

kernel void h3_cast_f32_to_bf16(device const float *input [[buffer(0)]],
                                device ushort *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_f32_to_bf16(input[gid]);
}

kernel void h3_cast_bf16_to_f32(device const ushort *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = h3_bf16_to_f32(input[gid]);
}

struct norm_args {
    uint rows;
    uint width;
    float epsilon;
};

struct qkv_args {
    uint sequence;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint grouped;
    float epsilon;
};

kernel void h3_rms_norm_f32(device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant norm_args &args [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 thread_position [[thread_position_in_threadgroup]],
                            uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads)
        local_sum = fma(x[k], x[k], local_sum);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] = x[column] * inverse * weight[column];
}

struct scale_add_args { uint rows; uint width; };

kernel void h3_scale_add_f32(device const float *residual [[buffer(0)]],
                             device const float *branch [[buffer(1)]],
                             device const float *scale [[buffer(2)]],
                             device float *output [[buffer(3)]],
                             constant scale_add_args &args [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

kernel void h3_layer_norm_f32(device const float *input [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *output [[buffer(3)]],
                              constant norm_args &args [[buffer(4)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint3 thread_position [[thread_position_in_threadgroup]],
                              uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const float *x = input + row * args.width;
    float local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) local += x[k];
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float centered = x[k] - mean;
        local = fma(centered, centered, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads)
        output[row * args.width + column] =
            (x[column] - mean) * inverse * weight[column] + bias[column];
}

kernel void h3_video_qkv_rope_f32(
                            device const float *qkv [[buffer(0)]],
                            device const float *rope_cos [[buffer(1)]],
                            device const float *rope_sin [[buffer(2)]],
                            device float *query [[buffer(3)]],
                            device float *key [[buffer(4)]],
                            device float *value [[buffer(5)]],
                            constant qkv_args &args [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint base = (row * args.heads + head) * args.head_dim * 3;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[base + d];
        float k = qkv[base + args.head_dim + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float qi = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float ki = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[base + dimension] * qi;
    float k0 = qkv[base + args.head_dim + dimension] * ki;
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * qi;
        float k1 = qkv[base + args.head_dim + pair] * ki;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + args.head_dim * 2 + dimension];
}

struct adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

kernel void h3_adaln_f32(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device const float *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         constant adaln_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    device const float *x = input + row * args.width;
    float sum = 0.0f;
    for (uint k = 0; k < args.width; k++) sum = fma(x[k], x[k], sum);
    float normalized = x[column] * rsqrt(sum / float(args.width) + args.epsilon) * weight[column];
    uint modulation_row = row_map[row];
    uint base = modulation_row * args.slots * args.width;
    float shift = modulation[base + args.shift_slot * args.width + column];
    float scale = modulation[base + args.scale_slot * args.width + column];
    output[row * args.width + column] = normalized * (1.0f + scale) + shift;
}

struct gate_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
};

kernel void h3_gate_f32(device const float *residual [[buffer(0)]],
                        device const float *branch [[buffer(1)]],
                        device const float *modulation [[buffer(2)]],
                        device const uint *row_map [[buffer(3)]],
                        device float *output [[buffer(4)]],
                        constant gate_args &args [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = modulation[base + args.gate_slot * args.width + column];
    uint index = row * args.width + column;
    output[index] = residual[index] + branch[index] * gate;
}

kernel void h3_qkv_rope_f32(device const float *qkv [[buffer(0)]],
                            device const float *q_weight [[buffer(1)]],
                            device const float *k_weight [[buffer(2)]],
                            device const float *rope_cos [[buffer(3)]],
                            device const float *rope_sin [[buffer(4)]],
                            device float *query [[buffer(5)]],
                            device float *key [[buffer(6)]],
                            device float *value [[buffer(7)]],
                            constant qkv_args &args [[buffer(8)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = qkv[q_base + d];
        float k = qkv[k_base + d];
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[q_base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[k_base + dimension] * k_inverse * k_weight[dimension];
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[v_base + dimension];
}

struct swiglu_args {
    uint rows;
    uint width;
};

kernel void h3_swiglu_f32(device const float *fused [[buffer(0)]],
                          device float *output [[buffer(1)]],
                          constant swiglu_args &args [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = fused[base + column];
    float up = fused[base + args.width + column];
    output[row * args.width + column] = gate / (1.0f + exp(-gate)) * up;
}

struct vae_encoder_pad_args {
    uint batch;
    uint depth;
    uint height;
    uint width;
    uint channels;
    uint depth_front;
    uint height_before;
    uint height_after;
    uint width_before;
    uint width_after;
};

static int h3_reflect_coordinate(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

kernel void h3_vae_encoder_pad_f32(
                            device const float *input [[buffer(0)]],
                            device float *output [[buffer(1)]],
                            constant vae_encoder_pad_args &args [[buffer(2)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x;
    uint out_x = gid.y;
    uint out_height = args.height + args.height_before + args.height_after;
    uint out_width = args.width + args.width_before + args.width_after;
    uint out_depth = args.depth + args.depth_front;
    uint plane = gid.z;
    if (channel >= args.channels || out_x >= out_width ||
        plane >= args.batch * out_depth * out_height) return;
    uint out_y = plane % out_height;
    uint temporal_plane = plane / out_height;
    uint out_t = temporal_plane % out_depth;
    uint batch = temporal_plane / out_depth;
    size_t destination = ((((size_t)batch * out_depth + out_t) * out_height +
                           out_y) * out_width + out_x) * args.channels +
                         channel;
    if (out_t < args.depth_front) {
        output[destination] = 0.0f;
        return;
    }
    int source_y = h3_reflect_coordinate(
        int(out_y) - int(args.height_before), int(args.height));
    int source_x = h3_reflect_coordinate(
        int(out_x) - int(args.width_before), int(args.width));
    uint source_t = out_t - args.depth_front;
    size_t source = ((((size_t)batch * args.depth + source_t) * args.height +
                      uint(source_y)) * args.width + uint(source_x)) *
                    args.channels + channel;
    output[destination] = input[source];
}

struct vae_encoder_norm_args {
    uint batch;
    uint depth;
    uint height;
    uint width;
    uint channels;
    uint groups;
    float epsilon;
};

kernel void h3_vae_encoder_group_norm_silu_f32(
                            device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device const float *bias [[buffer(2)]],
                            device float *output [[buffer(3)]],
                            constant vae_encoder_norm_args &args [[buffer(4)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint3 thread_position [[thread_position_in_threadgroup]],
                            uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint rows = args.batch * args.depth * args.groups;
    if (row >= rows) return;
    uint channels_per_group = args.channels / args.groups;
    uint group_index = row % args.groups;
    uint temporal_plane = row / args.groups;
    uint elements = args.height * args.width * channels_per_group;
    threadgroup float reductions[256];
    float local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        local += input[source];
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroup_size.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(elements);
    local = 0.0f;
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        float centered = input[source] - mean;
        local = fma(centered, centered, local);
    }
    reductions[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadgroup_size.x / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(elements) + args.epsilon);
    for (uint index = tid; index < elements; index += threadgroup_size.x) {
        uint spatial = index / channels_per_group;
        uint channel = group_index * channels_per_group +
                       index % channels_per_group;
        size_t destination =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
            args.channels + channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + exp(-value));
    }
}

struct weight_norm_args {
    uint outer;
    uint inner;
};

kernel void h3_weight_norm_f32(device const float *vector [[buffer(0)]],
                               device const float *magnitude [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               constant weight_norm_args &args [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= args.outer) return;
    uint base = row * args.inner;
    float square_sum = 0.0f;
    for (uint index = 0; index < args.inner; index++) {
        float value = vector[base + index];
        square_sum = fma(value, value, square_sum);
    }
    float scale = magnitude[row] * rsqrt(square_sum);
    for (uint index = 0; index < args.inner; index++)
        output[base + index] = vector[base + index] * scale;
}

struct add_scaled_args {
    uint elements;
    float left_scale;
    float right_scale;
};

kernel void h3_add_scaled_f32(device const float *left [[buffer(0)]],
                              device const float *right [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant add_scaled_args &args [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = left[index] * args.left_scale +
                        right[index] * args.right_scale;
}

struct audio_activation_args {
    uint batch;
    uint length;
    uint channels;
};

/* BigVGAN's fixed 2x upsample -> SnakeBeta -> 2x low-pass downsample is
 * fused at the original rate. This avoids materializing the doubled waveform
 * for each of the decoder's 127 alias-free activations. */
kernel void h3_alias_free_snake_f32(
        device const float *input [[buffer(0)]],
        device const float *alpha_log [[buffer(1)]],
        device const float *beta_log [[buffer(2)]],
        device const float *upsample_filter [[buffer(3)]],
        device const float *downsample_filter [[buffer(4)]],
        device float *output [[buffer(5)]],
        constant audio_activation_args &args [[buffer(6)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint channel = gid.x;
    uint time = gid.y;
    uint batch = gid.z;
    if (channel >= args.channels || time >= args.length ||
        batch >= args.batch) return;
    float alpha = exp(alpha_log[channel]);
    float beta = exp(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = int(time * 2) + down_k - 5;
        up_time = clamp(up_time, 0, int(args.length * 2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = clamp(padded_time - 5, 0,
                                    int(args.length) - 1);
            uint source = (batch * args.length + uint(source_time)) *
                          args.channels + channel;
            upsampled = fma(input[source], 2.0f * upsample_filter[up_k],
                            upsampled);
        }
        float sine = sin(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fma(activated, downsample_filter[down_k], result);
    }
    uint destination = (batch * args.length + time) * args.channels + channel;
    output[destination] = result;
}

kernel void h3_snake1d_f32(device const float *input [[buffer(0)]],
                           device const float *alpha [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant audio_activation_args &args [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    uint count = args.batch * args.length * args.channels;
    if (gid >= count) return;
    float a = alpha[gid % args.channels];
    float x = input[gid];
    float wave = sin(a * x);
    output[gid] = x + wave * wave / (a + 1e-9f);
}

struct audio_qkv_args {
    uint batch;
    uint length;
    uint heads;
    uint head_dim;
};

kernel void h3_audio_qkv_split_f32(
                           device const float *qkv [[buffer(0)]],
                           device const float *q_bias [[buffer(1)]],
                           device const float *k_bias [[buffer(2)]],
                           device const float *v_bias [[buffer(3)]],
                           device float *query [[buffer(4)]],
                           device float *key [[buffer(5)]],
                           device float *value [[buffer(6)]],
                           constant audio_qkv_args &args [[buffer(7)]],
                           uint gid [[thread_position_in_grid]]) {
    uint width = args.heads * args.head_dim;
    uint count = args.batch * args.length * width;
    if (gid >= count) return;
    uint column = gid % width;
    uint row = gid / width;
    uint base = row * width * 3;
    query[gid] = qkv[base + column] + q_bias[column];
    key[gid] = qkv[base + width + column] + k_bias[column];
    value[gid] = qkv[base + width * 2 + column] + v_bias[column];
}

struct audio_pool_args {
    uint batch;
    uint length;
    uint heads;
    uint head_dim;
    uint output_dim;
};

kernel void h3_audio_attention_pool_f32(
                           device const float *attended [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant audio_pool_args &args [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint count = args.batch * args.length * args.output_dim;
    if (gid >= count) return;
    uint column = gid % args.output_dim;
    uint row = gid / args.output_dim;
    uint pool = args.head_dim / args.output_dim;
    float sum = 0.0f;
    for (uint head = 0; head < args.heads; head++) {
        uint base = (row * args.heads + head) * args.head_dim + column * pool;
        for (uint item = 0; item < pool; item++) sum += attended[base + item];
    }
    output[gid] = sum / float(args.heads * pool);
}

kernel void h3_geglu_f32(device const float *gate [[buffer(0)]],
                          device const float *linear [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant uint &count [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float x = gate[gid];
    float cube = x * x * x;
    float gelu = 0.5f * x *
        (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * cube)));
    output[gid] = gelu * linear[gid];
}

struct clip_args {
    uint elements;
    float minimum;
    float maximum;
};

kernel void h3_clip_f32(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        constant clip_args &args [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < args.elements)
        output[index] = clamp(input[index], args.minimum, args.maximum);
}

kernel void h3_linear_bf16(device const ushort *input [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]],
                           device const ushort *bias [[buffer(2)]],
                           device ushort *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    /* Adapted from Iris's short-sequence Qwen path. Each 16x16 weight/input
     * tile is loaded once per threadgroup instead of once per output scalar. */
    threadgroup float input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint tile_count = (args.input_dim + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        input_tile[tid.y][tid.x] =
            row < args.rows && input_k < args.input_dim ?
            h3_bf16_to_f32(input[row * args.input_dim + input_k]) : 0.0f;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            h3_bf16_to_f32(weight[column * args.input_dim + weight_k]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++) {
            sum = fma(input_tile[tid.y][k], weight_tile[k][tid.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
    }
}

/* Draw Things/ccv-style dynamic symmetric row reduction. This helper is also
 * used by portable fused epilogues, so keep it outside the Metal 4 guard. */
inline float h3_int8_reduce_max(float value, threadgroup float *scratch,
                                ushort simdgroup, ushort lane) {
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane == 0) scratch[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        value = lane < 8 ? scratch[lane] : 0.0f;
        value = max(value, simd_shuffle_xor(value, 16));
        value = max(value, simd_shuffle_xor(value, 8));
        value = max(value, simd_shuffle_xor(value, 4));
        value = max(value, simd_shuffle_xor(value, 2));
        value = max(value, simd_shuffle_xor(value, 1));
        if (lane == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

#ifdef H3_METAL_HAS_TENSOR
/* Draw Things' Metal 4 matmul schedules neighboring row/column tiles in
 * Morton order. The decoder is adapted from ccv's BSD-3-Clause NAMatMul;
 * see THIRD_PARTY_NOTICES.md. Keep it local so the ordinary Metal path stays
 * buildable on pre-Metal-4 systems. */
inline uint h3_compact_morton_even_bits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1)) & 0x33333333u;
    value = (value | (value >> 2)) & 0x0f0f0f0fu;
    value = (value | (value >> 4)) & 0x00ff00ffu;
    value = (value | (value >> 8)) & 0x0000ffffu;
    return value;
}

inline uint h3_lower_bits_mask(uint bits) {
    return bits ? (1u << bits) - 1u : 0u;
}

inline uint2 h3_morton_decode_rectangular(uint code, uint x_bits,
                                          uint y_bits) {
    uint paired_bits = min(x_bits, y_bits);
    uint paired_code = code & h3_lower_bits_mask(paired_bits * 2);
    uint2 tile = uint2(h3_compact_morton_even_bits(paired_code),
                       h3_compact_morton_even_bits(paired_code >> 1));
    uint tail = code >> (paired_bits * 2);
    if (x_bits > paired_bits) {
        uint extra = x_bits - paired_bits;
        tile.x |= (tail & h3_lower_bits_mask(extra)) << paired_bits;
        tail >>= extra;
    }
    if (y_bits > paired_bits) tile.y |= tail << paired_bits;
    return tile;
}

/* Compact four-column Morton blocks. Full 4x4 row blocks use a true Morton
 * code; the final 1-3 row strip is packed without launching empty groups. */
inline uint2 h3_morton_decode_compact4(uint code, uint row_tiles) {
    uint groups_per_column_block = row_tiles * 4;
    uint column_block = code / groups_per_column_block;
    uint local = code - column_block * groups_per_column_block;
    uint full_rows = row_tiles & ~3u;
    uint full_groups = full_rows * 4;
    if (local < full_groups) {
        uint row_block = local >> 4;
        uint2 tile = uint2(h3_compact_morton_even_bits(local & 15u),
                           h3_compact_morton_even_bits((local & 15u) >> 1));
        return uint2(row_block * 4 + tile.x,
                     column_block * 4 + tile.y);
    }
    uint tail_rows = row_tiles - full_rows;
    uint tail = local - full_groups;
    return uint2(full_rows + tail % tail_rows,
                 column_block * 4 + tail / tail_rows);
}

/* Exact compact Morton walk for a final 1-3 column strip as well. */
inline uint2 h3_morton_decode_compact(uint code, uint row_tiles,
                                      uint column_tiles) {
    uint full_columns = column_tiles & ~3u;
    uint full_groups = row_tiles * full_columns;
    if (code < full_groups)
        return h3_morton_decode_compact4(code, row_tiles);
    uint tail_columns = column_tiles - full_columns;
    uint tail = code - full_groups;
    return uint2(tail / tail_columns,
                 full_columns + tail % tail_columns);
}

/*
 * M5 Metal 4/TensorOps path for the large H3 matrices.  This follows the
 * retained direct-RHS design in ds4.c: the contiguous dimension comes first
 * in Metal tensor extents, checkpoint weights are transposed logically rather
 * than copied, and TensorOps handles the final partial row tile.  Native
 * bfloat inputs avoid both a format conversion and a duplicate weight arena.
 */
kernel void h3_linear_bf16_nax_r128(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

kernel void h3_linear_bf16_nax_r128_morton(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint code [[threadgroup_position_in_grid]]) {
    uint row_tiles = (args.rows + 127) / 128;
    uint column_tiles = (args.output_dim + 63) / 64;
    uint row_bits = row_tiles <= 1 ? 0 : 32 - clz(row_tiles - 1);
    uint column_bits = column_tiles <= 1 ? 0 : 32 - clz(column_tiles - 1);
    uint2 group = h3_morton_decode_rectangular(
        code, row_bits, column_bits);
    if (group.x >= row_tiles || group.y >= column_tiles) return;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

kernel void h3_linear_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(3)]],
                           constant linear_args &args [[buffer(4)]],
                           uint code [[threadgroup_position_in_grid]]) {
    uint row_tiles = (args.rows + 127) / 128;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim, (int)args.output_dim));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        output,
        dextents<int32_t, 2>((int)args.output_dim, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * 128);
    auto mw = w.slice(0, (int)group.y * 64);
    auto my = y.slice((int)group.y * 64, (int)group.x * 128);
    matmul2d<matmul2d_descriptor(128, 64, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

struct h3_qkv_project_rope_args {
    uint rows;
    uint input_dim;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint head_major;
    float epsilon;
};

/* Preserve the successful direct 128x64 TensorOps matmul, but route each
 * grouped checkpoint tile directly to its Q, K, or V destination. The six
 * 64-column tiles of every head align exactly with Q0/Q1/K0/K1/V0/V1. */
kernel void h3_qkv_project_split_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *query [[buffer(2)]],
                           device bfloat *key [[buffer(3)]],
                           device bfloat *value [[buffer(4)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    constexpr int HEAD_DIM = 128;
    constexpr int TILES_PER_HEAD = 6;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    uint head = group.y / TILES_PER_HEAD;
    uint head_tile = group.y - head * TILES_PER_HEAD;
    uint stream = head_tile >> 1;
    uint tile_half = head_tile & 1;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    if (args.head_major) destination += head * args.rows * HEAD_DIM;
    int output_width = args.head_major ? HEAD_DIM :
        (int)args.heads * HEAD_DIM;
    int output_column = (args.head_major ? 0 : (int)head * HEAD_DIM) +
        (int)tile_half * COLUMN_TILE;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * HEAD_DIM * 3));
    auto y = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        destination, dextents<int32_t, 2>(output_width, (int)args.rows));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw = w.slice(0, (int)group.y * COLUMN_TILE);
    auto my = y.slice(output_column, (int)group.x * ROW_TILE);
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    mm.run(mx, mw, my);
}

/* Q and K already have the requested attention layout. Cache both 128-wide
 * heads, retain the oracle's lane-zero scalar RMS order, and overwrite them
 * with the rounded normalized/RoPE result. V needs no second pass. */
kernel void h3_qk_rope_bf16_nax_inplace(
                           device bfloat *query [[buffer(0)]],
                           device bfloat *key [[buffer(1)]],
                           device bfloat *q_weight [[buffer(2)]],
                           device bfloat *k_weight [[buffer(3)]],
                           device bfloat *rope_cos [[buffer(4)]],
                           device bfloat *rope_sin [[buffer(5)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(6)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint HEAD_DIM = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.rows) return;
    uint base = args.head_major ?
        (head * args.rows + row) * HEAD_DIM :
        (row * args.heads + head) * HEAD_DIM;
    threadgroup bfloat q_values[HEADS_PER_GROUP * HEAD_DIM];
    threadgroup bfloat k_values[HEADS_PER_GROUP * HEAD_DIM];
    uint cache_base = simdgroup * HEAD_DIM;
    for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
        q_values[cache_base + dimension] = query[base + dimension];
        k_values[cache_base + dimension] = key[base + dimension];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint dimension = 0; dimension < HEAD_DIM; dimension++) {
            float q_element = (float)q_values[cache_base + dimension];
            float k_element = (float)k_values[cache_base + dimension];
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
        float q0 = (float)q_values[cache_base + dimension] *
                   q_inverse[simdgroup] * (float)q_weight[dimension];
        float k0 = (float)k_values[cache_base + dimension] *
                   k_inverse[simdgroup] * (float)k_weight[dimension];
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = (float)q_values[cache_base + pair] *
                       q_inverse[simdgroup] * (float)q_weight[pair];
            float k1 = (float)k_values[cache_base + pair] *
                       k_inverse[simdgroup] * (float)k_weight[pair];
            float c = (float)rope_cos[row * args.rope_half + dimension];
            float s = (float)rope_sin[row * args.rope_half + dimension];
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = (float)q_values[cache_base + pair] *
                       q_inverse[simdgroup] * (float)q_weight[pair];
            float k1 = (float)k_values[cache_base + pair] *
                       k_inverse[simdgroup] * (float)k_weight[pair];
            float c = (float)rope_cos[row * args.rope_half + pair];
            float s = (float)rope_sin[row * args.rope_half + pair];
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        query[base + dimension] = (bfloat)q0;
        key[base + dimension] = (bfloat)k0;
    }
}

/* Pair the two FC1 halves while the TensorOps accumulators are still local.
 * Only the post-SwiGLU [rows, hidden_dim] tensor reaches device memory. */
kernel void h3_fc1_swiglu_bf16_nax_r128(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

kernel void h3_fc1_swiglu_bf16_nax_r128_morton(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint column_tiles = (args.output_dim + COLUMN_TILE - 1) / COLUMN_TILE;
    uint row_bits = row_tiles <= 1 ? 0 : 32 - clz(row_tiles - 1);
    uint column_bits = column_tiles <= 1 ? 0 : 32 - clz(column_tiles - 1);
    uint2 group = h3_morton_decode_rectangular(
        code, row_bits, column_bits);
    if (group.x >= row_tiles || group.y >= column_tiles) return;
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

kernel void h3_fc1_swiglu_bf16_nax_r128_morton4(
                           device bfloat *input [[buffer(0)]],
                           device bfloat *weight [[buffer(1)]],
                           device bfloat *output [[buffer(2)]],
                           constant linear_args &args [[buffer(3)]],
                           threadgroup bfloat *tiles [[threadgroup(0)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int ROW_TILE = 128;
    constexpr int COLUMN_TILE = 64;
    uint row_tiles = (args.rows + ROW_TILE - 1) / ROW_TILE;
    uint2 group = h3_morton_decode_compact4(code, row_tiles);
    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim, (int)args.rows));
    auto w = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight,
        dextents<int32_t, 2>((int)args.input_dim,
                             (int)args.output_dim * 2));
    auto mx = x.slice(0, (int)group.x * ROW_TILE);
    auto mw_gate = w.slice(0, (int)group.y * COLUMN_TILE);
    auto mw_up = w.slice(0, (int)args.output_dim +
                            (int)group.y * COLUMN_TILE);
    threadgroup bfloat *gate_tile = tiles;
    threadgroup bfloat *up_tile = tiles + ROW_TILE * COLUMN_TILE;
    auto tg_gate = tensor(gate_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    auto tg_up = tensor(up_tile,
        dextents<int32_t, 2>(COLUMN_TILE, ROW_TILE));
    matmul2d<matmul2d_descriptor(ROW_TILE, COLUMN_TILE, dynamic_extent,
                                false, true, false),
              execution_simdgroups<4>> mm;
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_gate, accum);
        accum.store(tg_gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(x), decltype(w), bfloat>();
        mm.run(mx, mw_up, accum);
        accum.store(tg_up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row_base = group.x * ROW_TILE;
    uint column_base = group.y * COLUMN_TILE;
    for (uint index = tid; index < ROW_TILE * COLUMN_TILE; index += 128) {
        uint row = row_base + index / COLUMN_TILE;
        uint column = column_base + index % COLUMN_TILE;
        if (row < args.rows && column < args.output_dim) {
            float gate = (float)gate_tile[index];
            float up = (float)up_tile[index];
            float activated = gate / (1.0f + exp(-gate)) * up;
            output[row * args.output_dim + column] = (bfloat)activated;
        }
    }
}

/* Draw Things/ccv-style dynamic symmetric row quantization. Weight rows are
 * checkpoint output channels; activation rows include zero-padded M tails so
 * the int8 matmuls can keep a fully static 128x128x128 descriptor. */
kernel void h3_quantize_bf16_int8_rows(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint base = row * args.columns;
    if (row >= args.rows) {
        if ((args.columns & 3u) == 0) {
            device char4 *output4 =
                reinterpret_cast<device char4 *>(output);
            uint vector_base = row * (args.columns / 4);
            for (uint column = tid; column < args.columns / 4;
                 column += 256)
                output4[vector_base + column] = char4(0);
        } else {
            for (uint column = tid; column < args.columns; column += 256)
                output[base + column] = 0;
        }
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    float local_max = 0.0f;
    if ((args.columns & 3u) == 0) {
        device const bfloat4 *input4 =
            reinterpret_cast<device const bfloat4 *>(input);
        uint vectors_per_row = args.columns / 4;
        uint vector_base = row * vectors_per_row;
        for (uint column = tid; column < vectors_per_row; column += 256) {
            float4 value = float4(input4[vector_base + column]);
            local_max = max(local_max,
                max(max(fabs(value.x), fabs(value.y)),
                    max(fabs(value.z), fabs(value.w))));
        }
    } else {
        for (uint column = tid; column < args.columns; column += 256)
            local_max = max(local_max, fabs((float)input[base + column]));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    if ((args.columns & 3u) == 0) {
        device const bfloat4 *input4 =
            reinterpret_cast<device const bfloat4 *>(input);
        device char4 *output4 =
            reinterpret_cast<device char4 *>(output);
        uint vectors_per_row = args.columns / 4;
        uint vector_base = row * vectors_per_row;
        for (uint column = tid; column < vectors_per_row; column += 256) {
            int4 quantized = int4(rint(
                float4(input4[vector_base + column]) * inverse));
            output4[vector_base + column] =
                char4(clamp(quantized, int4(-127), int4(127)));
        }
    } else {
        for (uint column = tid; column < args.columns; column += 256) {
            int quantized = (int)rint((float)input[base + column] * inverse);
            output[base + column] =
                (int8_t)clamp(quantized, -127, 127);
        }
    }
}

/* Retained only for crossed kernel measurements against the vec4 path. */
kernel void h3_quantize_bf16_int8_rows_scalar(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += 256)
            output[base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    float local_max = 0.0f;
    for (uint column = tid; column < args.columns; column += 256)
        local_max = max(local_max, fabs((float)input[base + column]));
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    for (uint column = tid; column < args.columns; column += 256) {
        int quantized = (int)rint((float)input[base + column] * inverse);
        output[base + column] = (int8_t)clamp(quantized, -127, 127);
    }
}

/* Convert SDPA's head-major source directly to the row-major int8 layout used
 * by the projection. The gather replaces the removed BF16 transpose while
 * preserving fully coalesced int8 stores and the proven projection kernel. */
kernel void h3_quantize_bf16_int8_head_major_to_rows_cached(
                           device const bfloat4 *input [[buffer(0)]],
                           device char4 *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_head_major_quant_args &args
                               [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 256;
    constexpr uint VECTORS = 7;
    constexpr uint VECTORS_PER_HEAD = 32;
    constexpr uint VECTORS_PER_ROW = 56 * VECTORS_PER_HEAD;
    if (row >= args.rows) {
        for (uint slot = 0; slot < VECTORS; slot++) {
            uint linear = tid + slot * THREADS;
            output[row * VECTORS_PER_ROW + linear] = char4(0);
        }
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    thread bfloat4 values[VECTORS];
    threadgroup float scratch[8];
    float local_max = 0.0f;
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < VECTORS; slot++) {
        uint linear = tid + slot * THREADS;
        uint head = linear >> 5;
        uint vector = linear & (VECTORS_PER_HEAD - 1);
        bfloat4 value = input[
            (head * args.rows + row) * VECTORS_PER_HEAD + vector];
        values[slot] = value;
        float4 f = float4(value);
        local_max = max(local_max,
            max(max(fabs(f.x), fabs(f.y)), max(fabs(f.z), fabs(f.w))));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, scratch, simdgroup, lane);
    float clipped_max = max_abs * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < VECTORS; slot++) {
        uint linear = tid + slot * THREADS;
        int4 quantized = int4(rint(float4(values[slot]) * inverse));
        output[row * VECTORS_PER_ROW + linear] =
            char4(clamp(quantized, int4(-127), int4(127)));
    }
}

kernel void h3_quantize_bf16_int8_groups(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint vectors_per_row = args.columns / 4;
    uint vectors_per_group = args.group_size / 4;
    uint vector_base = row * vectors_per_row;
    device const bfloat4 *input4 =
        reinterpret_cast<device const bfloat4 *>(input);
    device char4 *output4 = reinterpret_cast<device char4 *>(output);
    if (row >= args.rows) {
        for (uint column = tid; column < vectors_per_row; column += 256)
            output4[vector_base + column] = char4(0);
        for (uint group = tid; group < args.groups; group += 256)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    for (uint group = 0; group < args.groups; group++) {
        uint start = vector_base + group * vectors_per_group;
        float local_max = 0.0f;
        for (uint local = tid; local < vectors_per_group; local += 256) {
            float4 value = float4(input4[start + local]);
            local_max = max(local_max,
                max(max(fabs(value.x), fabs(value.y)),
                    max(fabs(value.z), fabs(value.w))));
        }
        float max_abs = h3_int8_reduce_max(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < vectors_per_group; local += 256) {
            int4 quantized = int4(rint(float4(input4[start + local]) *
                                       inverse));
            output4[start + local] =
                char4(clamp(quantized, int4(-127), int4(127)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Retained only for crossed kernel measurements against the vec4 path. */
kernel void h3_quantize_bf16_int8_groups_scalar(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += 256)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += 256)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[8];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        float local_max = 0.0f;
        for (uint local = tid; local < args.group_size; local += 256)
            local_max = max(local_max,
                fabs((float)input[row_base + start + local]));
        float max_abs = h3_int8_reduce_max(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < args.group_size; local += 256) {
            int quantized = (int)rint(
                (float)input[row_base + start + local] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline float h3_int8_reduce_max4(float value, threadgroup float *scratch,
                                 ushort simdgroup, ushort lane) {
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane == 0) scratch[simdgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        value = lane < 4 ? scratch[lane] : 0.0f;
        value = max(value, simd_shuffle_xor(value, 16));
        value = max(value, simd_shuffle_xor(value, 8));
        value = max(value, simd_shuffle_xor(value, 4));
        value = max(value, simd_shuffle_xor(value, 2));
        value = max(value, simd_shuffle_xor(value, 1));
        if (lane == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

/* Exact 128-thread form of the selected scalar grouped quantizer. */
kernel void h3_quantize_bf16_int8_groups_scalar128(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 128;
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += THREADS)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += THREADS)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[4];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        float local_max = 0.0f;
        for (uint local = tid; local < args.group_size; local += THREADS)
            local_max = max(local_max,
                fabs((float)input[row_base + start + local]));
        float max_abs = h3_int8_reduce_max4(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint local = tid; local < args.group_size; local += THREADS) {
            int quantized = (int)rint(
                (float)input[row_base + start + local] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* The selected 1,024-wide grouped quantizer assigns exactly eight values to
 * each of 128 threads. Keep them private across max reduction and emission so
 * the quantization pass does not reread the BF16 activation row. */
kernel void h3_quantize_bf16_int8_groups_scalar128_cached(
                           device const bfloat *input [[buffer(0)]],
                           device int8_t *output [[buffer(1)]],
                           device float *scales [[buffer(2)]],
                           constant int8_group_quant_args &args [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort simdgroup
                               [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    constexpr uint THREADS = 128;
    constexpr uint VALUES = 8;
    uint row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint column = tid; column < args.columns; column += THREADS)
            output[row_base + column] = 0;
        for (uint group = tid; group < args.groups; group += THREADS)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    threadgroup float scratch[4];
    for (uint group = 0; group < args.groups; group++) {
        uint start = group * args.group_size;
        thread bfloat values[VALUES];
        float local_max = 0.0f;
        #pragma clang loop unroll(full)
        for (uint slot = 0; slot < VALUES; slot++) {
            uint local = tid + slot * THREADS;
            bfloat value = input[row_base + start + local];
            values[slot] = value;
            local_max = max(local_max, fabs((float)value));
        }
        float max_abs = h3_int8_reduce_max4(
            local_max, scratch, simdgroup, lane);
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        #pragma clang loop unroll(full)
        for (uint slot = 0; slot < VALUES; slot++) {
            uint local = tid + slot * THREADS;
            int quantized = (int)rint((float)values[slot] * inverse);
            output[row_base + start + local] =
                (int8_t)clamp(quantized, -127, 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Int8 H3 QKV projection. Checkpoint columns are grouped as
 * [head, q/k/v, dimension], so each 128-column TensorOps tile writes one
 * complete stream/head directly in head-major SDPA layout. */
kernel void h3_qkv_project_split_int8_nax_r128_morton4(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(7)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += TILE) {
        auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
        auto b = w.slice<TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[checkpoint_column + column]);
    }
}

/* Keep each projected Q/K 128x128 tile resident in its owning threadgroup.
 * After the TensorOps epilogue has rounded to BF16, one thread computes each
 * row's exact scalar RMS order; all threads then normalize and apply RoPE in
 * disjoint dimension pairs. The temporary device tile is the final head-major
 * output itself, so this adds only 512 bytes of threadgroup storage and removes
 * the separate per-row/per-four-head Q/K kernel dispatch. */
template<uint K_TILE>
kernel void h3_qkv_project_split_int8_rope_nax_r128_morton4_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           device const bfloat *q_weight [[buffer(7)]],
                           device const bfloat *k_weight [[buffer(8)]],
                           device const bfloat *rope_cos [[buffer(9)]],
                           device const bfloat *rope_sin [[buffer(10)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(11)]],
                           uint code [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    constexpr uint ROPE_PAIRS = 48;
    constexpr uint NORMALIZED_UNITS = ROPE_PAIRS +
        (TILE - ROPE_PAIRS * 2);
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
    auto first_b = w.slice<K_TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += K_TILE) {
        auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
        auto b = w.slice<K_TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[checkpoint_column + column]);
    }
    if (stream < 2) {
        threadgroup float inverse[TILE];
        threadgroup_barrier(mem_flags::mem_device);
        uint row = row_start + tid;
        if (tid < TILE && row < args.rows) {
            float sum = 0.0f;
            if (args.head_major & 2u) {
                device const bfloat4 *destination4 =
                    reinterpret_cast<device const bfloat4 *>(destination);
                uint vector_base = row * (TILE / 4);
                for (uint vector = 0; vector < TILE / 4; vector++) {
                    float4 elements = float4(destination4[vector_base + vector]);
                    sum = fma(elements.x, elements.x, sum);
                    sum = fma(elements.y, elements.y, sum);
                    sum = fma(elements.z, elements.z, sum);
                    sum = fma(elements.w, elements.w, sum);
                }
            } else {
                for (uint dimension = 0; dimension < TILE; dimension++) {
                    float element =
                        (float)destination[row * TILE + dimension];
                    sum = fma(element, element, sum);
                }
            }
            inverse[tid] = rsqrt(sum / float(TILE) + args.epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const bfloat *norm_weight = stream == 0 ? q_weight : k_weight;
        if (args.head_major & 4u) {
            constexpr uint PACKED_UNITS = NORMALIZED_UNITS / 4;
            device ushort4 *destination4 =
                reinterpret_cast<device ushort4 *>(destination);
            device const ushort4 *norm_weight4 =
                reinterpret_cast<device const ushort4 *>(norm_weight);
            device const ushort4 *rope_cos4 =
                reinterpret_cast<device const ushort4 *>(rope_cos);
            device const ushort4 *rope_sin4 =
                reinterpret_cast<device const ushort4 *>(rope_sin);
            for (uint unit = tid; unit < TILE * PACKED_UNITS; unit += 256) {
                uint local_row = unit / PACKED_UNITS;
                uint part = unit - local_row * PACKED_UNITS;
                uint global_row = row_start + local_row;
                if (global_row >= args.rows) continue;
                uint base4 = global_row * (TILE / 4);
                float inv = inverse[local_row];
                if (part < ROPE_PAIRS / 4) {
                    uint upper = part + ROPE_PAIRS / 4;
                    ushort4 lower_bits = destination4[base4 + part];
                    ushort4 upper_bits = destination4[base4 + upper];
                    ushort4 lower_norm = norm_weight4[part];
                    ushort4 upper_norm = norm_weight4[upper];
                    float4 lower_value = h3_bf16x4_to_f32(lower_bits) * inv *
                        h3_bf16x4_to_f32(lower_norm);
                    float4 upper_value = h3_bf16x4_to_f32(upper_bits) * inv *
                        h3_bf16x4_to_f32(upper_norm);
                    uint rope_base = global_row * (args.rope_half / 4);
                    ushort4 cosine_bits = rope_cos4[rope_base + part];
                    ushort4 sine_bits = rope_sin4[rope_base + part];
                    float4 cosine = h3_bf16x4_to_f32(cosine_bits);
                    float4 sine = h3_bf16x4_to_f32(sine_bits);
                    float4 lower_output = lower_value * cosine -
                        upper_value * sine;
                    float4 upper_output = upper_value * cosine +
                        lower_value * sine;
                    destination4[base4 + part] =
                        h3_f32x4_to_bf16(lower_output);
                    destination4[base4 + upper] =
                        h3_f32x4_to_bf16(upper_output);
                } else {
                    uint dimension = part + ROPE_PAIRS / 4;
                    ushort4 value_bits = destination4[base4 + dimension];
                    ushort4 norm_bits = norm_weight4[dimension];
                    destination4[base4 + dimension] = h3_f32x4_to_bf16(
                        h3_bf16x4_to_f32(value_bits) * inv *
                        h3_bf16x4_to_f32(norm_bits));
                }
            }
        } else for (uint unit = tid; unit < TILE * NORMALIZED_UNITS;
                    unit += 256) {
            uint local_row = unit / NORMALIZED_UNITS;
            uint part = unit - local_row * NORMALIZED_UNITS;
            uint global_row = row_start + local_row;
            if (global_row >= args.rows) continue;
            uint base = global_row * TILE;
            float inv = inverse[local_row];
            if (part < ROPE_PAIRS) {
                uint upper = part + ROPE_PAIRS;
                float lower_value = (float)destination[base + part] * inv *
                                    (float)norm_weight[part];
                float upper_value = (float)destination[base + upper] * inv *
                                    (float)norm_weight[upper];
                float c = (float)rope_cos[
                    global_row * args.rope_half + part];
                float s = (float)rope_sin[
                    global_row * args.rope_half + part];
                destination[base + part] =
                    (bfloat)(lower_value * c - upper_value * s);
                destination[base + upper] =
                    (bfloat)(upper_value * c + lower_value * s);
            } else {
                uint dimension = part + ROPE_PAIRS;
                destination[base + dimension] = (bfloat)(
                    (float)destination[base + dimension] * inv *
                    (float)norm_weight[dimension]);
            }
        }
    }
}

typedef decltype(h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<128>)
    h3_qkv_project_split_int8_rope_nax_r128_morton4_t;
template [[host_name("h3_qkv_project_split_int8_rope_nax_r128_morton4")]]
kernel h3_qkv_project_split_int8_rope_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<128>;
template [[host_name("h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4")]]
kernel h3_qkv_project_split_int8_rope_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_nax_r128_morton4_impl<5376>;

/* QKV variant that caches the row and checkpoint-column scale vectors. After
 * every cooperative fragment has consumed the scales, the first half is
 * recycled for inverse RMS and the second half for Q/K normalization weights. */
template<uint K_TILE>
kernel void h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *query [[buffer(4)]],
                           device bfloat *key [[buffer(5)]],
                           device bfloat *value [[buffer(6)]],
                           device const bfloat *q_weight [[buffer(7)]],
                           device const bfloat *k_weight [[buffer(8)]],
                           device const bfloat *rope_cos [[buffer(9)]],
                           device const bfloat *rope_sin [[buffer(10)]],
                           constant h3_qkv_project_rope_args &args
                               [[buffer(11)]],
                           uint code [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint STREAMS = 3;
    constexpr uint ROPE_PAIRS = 48;
    constexpr uint NORMALIZED_UNITS = ROPE_PAIRS +
        (TILE - ROPE_PAIRS * 2);
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.heads * STREAMS;
    uint2 group = h3_morton_decode_compact(code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint checkpoint_column = group.y * TILE;
    uint head = group.y / STREAMS;
    uint stream = group.y - head * STREAMS;
    device bfloat *destination = stream == 0 ? query :
        stream == 1 ? key : value;
    destination += head * args.rows * TILE;
    threadgroup float scratch[TILE * 2];
    if (tid < TILE) {
        scratch[tid] = input_scales[row_start + tid];
        scratch[TILE + tid] = weight_scales[checkpoint_column + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(
            (int)args.input_dim, (int)args.heads * (int)TILE * 3));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
    auto first_b = w.slice<K_TILE, TILE>(0, (int)checkpoint_column);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += K_TILE) {
        auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
        auto b = w.slice<K_TILE, TILE>((int)k, (int)checkpoint_column);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = (uint)index[0];
        if (row < args.rows)
            destination[row * TILE + column] =
                (bfloat)((float)accum[element] * scratch[(uint)index[1]] *
                         scratch[TILE + column]);
    }
    if (stream < 2) {
        device const bfloat *norm_weight = stream == 0 ? q_weight : k_weight;
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
        uint row = row_start + tid;
        if (tid < TILE) {
            scratch[TILE + tid] = (float)norm_weight[tid];
            if (row < args.rows) {
                float sum = 0.0f;
                if (args.head_major & 2u) {
                    device const bfloat4 *destination4 =
                        reinterpret_cast<device const bfloat4 *>(destination);
                    uint vector_base = row * (TILE / 4);
                    for (uint vector = 0; vector < TILE / 4; vector++) {
                        float4 elements =
                            float4(destination4[vector_base + vector]);
                        sum = fma(elements.x, elements.x, sum);
                        sum = fma(elements.y, elements.y, sum);
                        sum = fma(elements.z, elements.z, sum);
                        sum = fma(elements.w, elements.w, sum);
                    }
                } else {
                    for (uint dimension = 0; dimension < TILE; dimension++) {
                        float element =
                            (float)destination[row * TILE + dimension];
                        sum = fma(element, element, sum);
                    }
                }
                scratch[tid] = rsqrt(sum / float(TILE) + args.epsilon);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (args.head_major & 4u) {
            constexpr uint PACKED_UNITS = NORMALIZED_UNITS / 4;
            device ushort4 *destination4 =
                reinterpret_cast<device ushort4 *>(destination);
            device const ushort4 *rope_cos4 =
                reinterpret_cast<device const ushort4 *>(rope_cos);
            device const ushort4 *rope_sin4 =
                reinterpret_cast<device const ushort4 *>(rope_sin);
            threadgroup float4 *norm_weight4 =
                reinterpret_cast<threadgroup float4 *>(scratch + TILE);
            for (uint unit = tid; unit < TILE * PACKED_UNITS; unit += 256) {
                uint local_row = unit / PACKED_UNITS;
                uint part = unit - local_row * PACKED_UNITS;
                uint global_row = row_start + local_row;
                if (global_row >= args.rows) continue;
                uint base4 = global_row * (TILE / 4);
                float inv = scratch[local_row];
                if (part < ROPE_PAIRS / 4) {
                    uint upper = part + ROPE_PAIRS / 4;
                    ushort4 lower_bits = destination4[base4 + part];
                    ushort4 upper_bits = destination4[base4 + upper];
                    float4 lower_value = h3_bf16x4_to_f32(lower_bits) * inv *
                        norm_weight4[part];
                    float4 upper_value = h3_bf16x4_to_f32(upper_bits) * inv *
                        norm_weight4[upper];
                    uint rope_base = global_row * (args.rope_half / 4);
                    ushort4 cosine_bits = rope_cos4[rope_base + part];
                    ushort4 sine_bits = rope_sin4[rope_base + part];
                    float4 cosine = h3_bf16x4_to_f32(cosine_bits);
                    float4 sine = h3_bf16x4_to_f32(sine_bits);
                    float4 lower_output = lower_value * cosine -
                        upper_value * sine;
                    float4 upper_output = upper_value * cosine +
                        lower_value * sine;
                    destination4[base4 + part] =
                        h3_f32x4_to_bf16(lower_output);
                    destination4[base4 + upper] =
                        h3_f32x4_to_bf16(upper_output);
                } else {
                    uint dimension = part + ROPE_PAIRS / 4;
                    ushort4 value_bits = destination4[base4 + dimension];
                    destination4[base4 + dimension] = h3_f32x4_to_bf16(
                        h3_bf16x4_to_f32(value_bits) * inv *
                        norm_weight4[dimension]);
                }
            }
        } else for (uint unit = tid; unit < TILE * NORMALIZED_UNITS;
                    unit += 256) {
            uint local_row = unit / NORMALIZED_UNITS;
            uint part = unit - local_row * NORMALIZED_UNITS;
            uint global_row = row_start + local_row;
            if (global_row >= args.rows) continue;
            uint base = global_row * TILE;
            float inv = scratch[local_row];
            if (part < ROPE_PAIRS) {
                uint upper = part + ROPE_PAIRS;
                float lower_value = (float)destination[base + part] * inv *
                                    scratch[TILE + part];
                float upper_value = (float)destination[base + upper] * inv *
                                    scratch[TILE + upper];
                float c = (float)rope_cos[
                    global_row * args.rope_half + part];
                float s = (float)rope_sin[
                    global_row * args.rope_half + part];
                destination[base + part] =
                    (bfloat)(lower_value * c - upper_value * s);
                destination[base + upper] =
                    (bfloat)(upper_value * c + lower_value * s);
            } else {
                uint dimension = part + ROPE_PAIRS;
                destination[base + dimension] = (bfloat)(
                    (float)destination[base + dimension] * inv *
                    scratch[TILE + dimension]);
            }
        }
    }
}

typedef decltype(
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<128>)
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t;
template [[host_name(
    "h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4")]]
kernel h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<128>;
template [[host_name(
    "h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4")]]
kernel h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_t
    h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4_impl<5376>;

/* INPUT_DIM=5376 lets Metal schedule H3's fixed FC1 K loop without unrolling
 * its 42 TensorOps slices; zero retains the generic runtime-bound fallback. */
template<uint INPUT_DIM, uint K_TILE>
kernel void h3_fc1_swiglu_int8_nax_r128_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    uint input_dim = INPUT_DIM ? INPUT_DIM : args.input_dim;
    constexpr bool FULL_PRODUCT = INPUT_DIM != 0 && K_TILE == INPUT_DIM;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)input_dim,
                                     (int)args.output_dim * 2));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, K_TILE, false, true, true,
        FULL_PRODUCT ? matmul2d_descriptor::mode::multiply :
                       matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    threadgroup bfloat gate_tile[TILE * TILE];
    {
        auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
        auto first_b = w.slice<K_TILE, TILE>(0, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        if constexpr (!FULL_PRODUCT) {
            #pragma clang loop unroll(full)
            for (ushort element = 0; element < accum.get_capacity(); element++)
                if (accum.is_valid_element(element)) accum[element] = 0;
        }
        if (INPUT_DIM) {
            for (uint k = 0; k < INPUT_DIM; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>((int)k, (int)column_start);
                mm.run(a, b, accum);
            }
        } else {
            for (uint k = 0; k < input_dim; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>((int)k, (int)column_start);
                mm.run(a, b, accum);
            }
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] * input_scales[row] *
                          weight_scales[column];
            gate_tile[(uint)index[1] * TILE + (uint)index[0]] =
                (bfloat)value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        auto first_a = x.slice<TILE, K_TILE>(0, (int)row_start);
        auto first_b = w.slice<K_TILE, TILE>(
            0, (int)args.output_dim + (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        if constexpr (!FULL_PRODUCT) {
            #pragma clang loop unroll(full)
            for (ushort element = 0; element < accum.get_capacity(); element++)
                if (accum.is_valid_element(element)) accum[element] = 0;
        }
        if (INPUT_DIM) {
            for (uint k = 0; k < INPUT_DIM; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>(
                    (int)k, (int)args.output_dim + (int)column_start);
                mm.run(a, b, accum);
            }
        } else {
            for (uint k = 0; k < input_dim; k += K_TILE) {
                auto a = x.slice<TILE, K_TILE>((int)k, (int)row_start);
                auto b = w.slice<K_TILE, TILE>(
                    (int)k, (int)args.output_dim + (int)column_start);
                mm.run(a, b, accum);
            }
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint local = (uint)index[1] * TILE + (uint)index[0];
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            if (row >= args.rows) continue;
            float gate = (float)gate_tile[local];
            float up = (float)accum[element] * input_scales[row] *
                       weight_scales[args.output_dim + column];
            output[row * args.output_dim + column] =
                (bfloat)(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

typedef decltype(h3_fc1_swiglu_int8_nax_r128_impl<0, 128>)
    h3_fc1_swiglu_int8_nax_r128_t;
template [[host_name("h3_fc1_swiglu_int8_nax_r128")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<0, 128>;
template [[host_name("h3_fc1_swiglu_int8_nax_r128_k5376")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<5376, 128>;
template [[host_name("h3_fc1_swiglu_int8_nax_r128_full_k5376")]]
kernel h3_fc1_swiglu_int8_nax_r128_t
    h3_fc1_swiglu_int8_nax_r128_impl<5376, 5376>;

/* Gate and up use the same cooperative-fragment mapping. Preserve the gate's
 * existing BF16 rounding point in thread-private storage, then consume it
 * after the up projection without a 32 KiB threadgroup tile or barrier. */
kernel void h3_fc1_swiglu_int8_local_nax_r128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim * 2));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    thread bfloat gate_values[FRAGMENT_CAPACITY];
    {
        auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
        auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        for (uint k = 0; k < args.input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] * input_scales[row] *
                          weight_scales[column];
            gate_values[element] = (bfloat)value;
        }
    }
    {
        auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
        auto first_b = w.slice<TILE, TILE>(
            0, (int)args.output_dim + (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        for (uint k = 0; k < args.input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>(
                (int)k, (int)args.output_dim + (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            if (row >= args.rows) continue;
            float gate = (float)gate_values[element];
            float up = (float)accum[element] * input_scales[row] *
                       weight_scales[args.output_dim + column];
            output[row * args.output_dim + column] =
                (bfloat)(gate / (1.0f + exp(-gate)) * up);
        }
    }
}

kernel void h3_linear_int8_nax_r128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint column_tiles = args.output_dim / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++)
        if (accum.is_valid_element(element)) accum[element] = 0;
    for (uint k = 0; k < args.input_dim; k += TILE) {
        auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
        auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
        mm.run(a, b, accum);
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * args.output_dim + column] =
                (bfloat)((float)accum[element] * input_scales[row] *
                         weight_scales[column]);
    }
}

/* One-scale FC2 path. A static full-K product lets NAX own the
 * complete 14336-wide reduction; scale loads overlap that long operation. */
kernel void h3_linear_int8_nax_r128_full_k14336(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    constexpr uint INPUT_DIM = 14336;
    constexpr uint OUTPUT_DIM = 5376;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / TILE);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    auto a = x.slice<TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

kernel void h3_linear_int8_nax_r128x256_full_k14336(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 256;
    constexpr uint INPUT_DIM = 14336;
    constexpr uint OUTPUT_DIM = 5376;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, OUTPUT_DIM / COLUMN_TILE);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    threadgroup float local_input_scales[ROW_TILE];
    threadgroup float local_weight_scales[COLUMN_TILE];
    if (tid < COLUMN_TILE) {
        if (tid < ROW_TILE)
            local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)INPUT_DIM,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)INPUT_DIM,
                                     (int)OUTPUT_DIM));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, INPUT_DIM, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<16>> mm;
    auto a = x.slice<ROW_TILE, INPUT_DIM>(0, (int)row_start);
    auto b = w.slice<INPUT_DIM, COLUMN_TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), int32_t>();
    mm.run(a, b, accum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * OUTPUT_DIM + column] =
                (bfloat)((float)accum[element] *
                    local_input_scales[(uint)index[1]] *
                    local_weight_scales[(uint)index[0]]);
    }
}

/* Cache the two 128-value dequantization vectors once per projection tile.
 * The cooperative output fragment otherwise rereads them for every element. */
template<uint INPUT_DIM, uint OUTPUT_DIM>
kernel void h3_linear_int8_local_scales_nax_r128_impl(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 128;
    uint padded_rows = (args.rows + TILE - 1) & ~(TILE - 1);
    uint row_tiles = padded_rows / TILE;
    uint output_dim = OUTPUT_DIM ? OUTPUT_DIM : args.output_dim;
    uint column_tiles = output_dim / TILE;
    uint2 group = h3_morton_decode_compact(code, row_tiles, column_tiles);
    uint row_start = group.x * TILE;
    uint column_start = group.y * TILE;
    uint input_dim = INPUT_DIM ? INPUT_DIM : args.input_dim;
    threadgroup float local_input_scales[TILE];
    threadgroup float local_weight_scales[TILE];
    if (tid < TILE) {
        local_input_scales[tid] = input_scales[row_start + tid];
        local_weight_scales[tid] = weight_scales[column_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)input_dim,
                                     (int)output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto first_descriptor = matmul2d_descriptor(
        TILE, TILE, TILE, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    matmul2d<first_descriptor, execution_simdgroups<8>> first_mm;
    auto first_a = x.slice<TILE, TILE>(0, (int)row_start);
    auto first_b = w.slice<TILE, TILE>(0, (int)column_start);
    auto accum = mm.template get_destination_cooperative_tensor<
        decltype(first_a), decltype(first_b), int32_t>();
    first_mm.run(first_a, first_b, accum);
    if (INPUT_DIM) {
        for (uint k = TILE; k < INPUT_DIM; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
    } else {
        for (uint k = TILE; k < input_dim; k += TILE) {
            auto a = x.slice<TILE, TILE>((int)k, (int)row_start);
            auto b = w.slice<TILE, TILE>((int)k, (int)column_start);
            mm.run(a, b, accum);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort element = 0; element < accum.get_capacity(); element++) {
        if (!accum.is_valid_element(element)) continue;
        auto index = accum.get_multidimensional_index(element);
        uint row = row_start + (uint)index[1];
        uint column = column_start + (uint)index[0];
        if (row < args.rows)
            output[row * output_dim + column] =
                (bfloat)((float)accum[element] *
                         local_input_scales[(uint)index[1]] *
                         local_weight_scales[(uint)index[0]]);
    }
}

typedef decltype(h3_linear_int8_local_scales_nax_r128_impl<0, 0>)
    h3_linear_int8_local_scales_nax_r128_t;
template [[host_name("h3_linear_int8_local_scales_nax_r128")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<0, 0>;
template [[host_name("h3_linear_int8_local_scales_nax_r128_k7168")]]
kernel h3_linear_int8_local_scales_nax_r128_t
    h3_linear_int8_local_scales_nax_r128_impl<7168, 5376>;

/* FC2 is more sensitive to a single scale spanning all 14336 activated
 * channels.  Retain one activation scale per 1024-wide K group, accumulate
 * each group's exact int32 product separately, then apply its scale before
 * adding it to the FP32 tile.  The 128x64 tile keeps that FP32 tile within
 * the M5 threadgroup-memory limit. */
kernel void h3_linear_int8_grouped_nax_r128x64(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 64;
    constexpr uint K_TILE = 128;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> mm;
    threadgroup float totals[ROW_TILE * COLUMN_TILE];
    for (uint scale_group = 0; scale_group < scale_groups; scale_group++) {
        uint k_start = scale_group * SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)k_start, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)k_start, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        #pragma clang loop unroll(full)
        for (uint k_tile = 0; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = k_start + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint local = (uint)index[1] * COLUMN_TILE +
                         (uint)index[0];
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] *
                input_scales[row * scale_groups + scale_group] *
                weight_scales[column];
            if (scale_group == 0) totals[local] = value;
            else totals[local] += value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint local = tid; local < ROW_TILE * COLUMN_TILE; local += 128) {
        uint row = row_start + local / COLUMN_TILE;
        uint column = column_start + local % COLUMN_TILE;
        if (row < args.rows)
            output[row * args.output_dim + column] = (bfloat)totals[local];
    }
}

/* Same scaled K-group arithmetic as the threadgroup-tile version, but the
 * cooperative tensor assigns the same fragment element to the same thread
 * for every K slice. Keep those 64 FP32 totals private and issue two
 * 512-wide scale-group products before draining either fragment. */
kernel void h3_linear_int8_grouped_local_nax_r128x64(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 64;
    constexpr uint K_TILE = 512;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    constexpr uint SCALE_GROUPS = 14;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    threadgroup float local_input_scales[ROW_TILE * SCALE_GROUPS];
    threadgroup float local_weight_scales[COLUMN_TILE];
    for (uint local = tid; local < ROW_TILE * SCALE_GROUPS; local += 256) {
        uint row = row_start + local / SCALE_GROUPS;
        uint scale_group = local % SCALE_GROUPS;
        local_input_scales[local] =
            input_scales[row * SCALE_GROUPS + scale_group];
    }
    if (tid < COLUMN_TILE)
        local_weight_scales[tid] = weight_scales[column_start + tid];
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto first_descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    matmul2d<first_descriptor, execution_simdgroups<8>> first_mm;
    thread float totals[FRAGMENT_CAPACITY];
    for (uint scale_group = 0; scale_group < scale_groups;
         scale_group += 2) {
        uint first_k = scale_group * SCALE_GROUP;
        uint second_k = first_k + SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)first_k, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)first_k, (int)column_start);
        auto first_accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        first_mm.run(first_a, first_b, first_accum);
        #pragma clang loop unroll(full)
        for (uint k_tile = 1; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = first_k + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, first_accum);
        }
        auto second_a = x.slice<ROW_TILE, K_TILE>(
            (int)second_k, (int)row_start);
        auto second_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)second_k, (int)column_start);
        auto second_accum = mm.template get_destination_cooperative_tensor<
            decltype(second_a), decltype(second_b), int32_t>();
        first_mm.run(second_a, second_b, second_accum);
        #pragma clang loop unroll(full)
        for (uint k_tile = 1; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = second_k + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, second_accum);
        }
        if (scale_group == 0)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < first_accum.get_capacity();
             element++) {
            if (!first_accum.is_valid_element(element)) continue;
            auto index = first_accum.get_multidimensional_index(element);
            float value = (float)first_accum[element] *
                local_input_scales[(uint)index[1] * SCALE_GROUPS +
                                   scale_group] *
                local_weight_scales[(uint)index[0]];
            if (scale_group == 0) totals[element] = value;
            else totals[element] += value;
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < second_accum.get_capacity();
             element++) {
            if (!second_accum.is_valid_element(element)) continue;
            auto index = second_accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)second_accum[element] *
                local_input_scales[(uint)index[1] * SCALE_GROUPS +
                                   scale_group + 1] *
                local_weight_scales[(uint)index[0]];
            totals[element] += value;
            if (scale_group + 2 == scale_groups && row < args.rows)
                output[row * args.output_dim + column] =
                    (bfloat)totals[element];
        }
    }
}

/* Wider-output form: the doubled tile uses eight SIMD groups, so each thread
 * still owns 64 output elements while the grid launches half as many FC2
 * workgroups. */
kernel void h3_linear_int8_grouped_local_nax_r128x128(
                           device int8_t *input [[buffer(0)]],
                           device int8_t *weight [[buffer(1)]],
                           device const float *input_scales [[buffer(2)]],
                           device const float *weight_scales [[buffer(3)]],
                           device bfloat *output [[buffer(4)]],
                           constant linear_args &args [[buffer(5)]],
                           uint code [[threadgroup_position_in_grid]]) {
    constexpr uint ROW_TILE = 128;
    constexpr uint COLUMN_TILE = 128;
    constexpr uint K_TILE = 128;
    constexpr uint SCALE_GROUP = 1024;
    constexpr uint K_TILES_PER_GROUP = SCALE_GROUP / K_TILE;
    constexpr uint FRAGMENT_CAPACITY = 64;
    uint padded_rows = (args.rows + ROW_TILE - 1) & ~(ROW_TILE - 1);
    uint row_tiles = padded_rows / ROW_TILE;
    uint column_tiles = args.output_dim / COLUMN_TILE;
    uint2 group = h3_morton_decode_compact(
        code, row_tiles, column_tiles);
    uint row_start = group.x * ROW_TILE;
    uint column_start = group.y * COLUMN_TILE;
    uint scale_groups = args.input_dim / SCALE_GROUP;
    auto x = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>((int)args.input_dim,
                                    (int)padded_rows));
    auto w = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>((int)args.input_dim,
                                     (int)args.output_dim));
    constexpr auto descriptor = matmul2d_descriptor(
        ROW_TILE, COLUMN_TILE, K_TILE, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> mm;
    thread float totals[FRAGMENT_CAPACITY];
    for (uint scale_group = 0; scale_group < scale_groups; scale_group++) {
        uint k_start = scale_group * SCALE_GROUP;
        auto first_a = x.slice<ROW_TILE, K_TILE>(
            (int)k_start, (int)row_start);
        auto first_b = w.slice<K_TILE, COLUMN_TILE>(
            (int)k_start, (int)column_start);
        auto accum = mm.template get_destination_cooperative_tensor<
            decltype(first_a), decltype(first_b), int32_t>();
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++)
            if (accum.is_valid_element(element)) accum[element] = 0;
        #pragma clang loop unroll(full)
        for (uint k_tile = 0; k_tile < K_TILES_PER_GROUP; k_tile++) {
            uint k = k_start + k_tile * K_TILE;
            auto a = x.slice<ROW_TILE, K_TILE>((int)k, (int)row_start);
            auto b = w.slice<K_TILE, COLUMN_TILE>(
                (int)k, (int)column_start);
            mm.run(a, b, accum);
        }
        #pragma clang loop unroll(full)
        for (ushort element = 0; element < accum.get_capacity(); element++) {
            if (!accum.is_valid_element(element)) continue;
            auto index = accum.get_multidimensional_index(element);
            uint row = row_start + (uint)index[1];
            uint column = column_start + (uint)index[0];
            float value = (float)accum[element] *
                input_scales[row * scale_groups + scale_group] *
                weight_scales[column];
            if (scale_group == 0) totals[element] = value;
            else totals[element] += value;
            if (scale_group + 1 == scale_groups && row < args.rows)
                output[row * args.output_dim + column] =
                    (bfloat)totals[element];
        }
    }
}
#endif

kernel void h3_silu_bf16(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(input[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)));
}

kernel void h3_rms_norm_bf16(device const ushort *input [[buffer(0)]],
                             device const ushort *weight [[buffer(1)]],
                             device ushort *output [[buffer(2)]],
                             constant norm_args &args [[buffer(3)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 thread_position [[thread_position_in_threadgroup]],
                             uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse;
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}

kernel void h3_layer_norm_bf16(device const ushort *input [[buffer(0)]],
                               device const ushort *weight [[buffer(1)]],
                               device const ushort *bias [[buffer(2)]],
                               device ushort *output [[buffer(3)]],
                               constant norm_args &args [[buffer(4)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint3 thread_position [[thread_position_in_threadgroup]],
                               uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads)
        local_sum += h3_bf16_to_f32(x[column]);
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = reductions[0] / float(args.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_square = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        float centered = h3_bf16_to_f32(x[column]) - mean;
        local_square = fma(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = (h3_bf16_to_f32(x[column]) - mean) * inverse;
        float value = fma(normalized, h3_bf16_to_f32(weight[column]),
                          h3_bf16_to_f32(bias[column]));
        output[row * args.width + column] = h3_f32_to_bf16(value);
    }
}

struct gelu_bf16_args {
    uint elements;
    uint approximate;
};

inline float h3_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                           1.421413741f) * t - 0.284496736f) * t +
                           0.254829592f) * t;
    return sign * (1.0f - polynomial * exp(-x * x));
}

kernel void h3_gelu_bf16(device const ushort *input [[buffer(0)]],
                         device ushort *output [[buffer(1)]],
                         constant gelu_bf16_args &args [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= args.elements) return;
    float value = h3_bf16_to_f32(input[index]);
    float activated;
    if (args.approximate) {
        float inner = 0.7978845608028654f *
            (value + 0.044715f * value * value * value);
        /* Apple GPU tanh may return NaN for large finite arguments under the
         * runtime compiler's fast-math mode. Saturation is already exact at
         * BF16 precision outside this range. */
        activated = inner <= -10.0f ? 0.0f :
                    inner >= 10.0f ? value :
                    0.5f * value * (1.0f + tanh(inner));
    } else {
        activated = value <= -10.0f ? 0.0f :
                    value >= 10.0f ? value :
                    0.5f * value *
                    (1.0f + h3_erf_approx(value * 0.7071067811865475f));
    }
    output[index] = h3_f32_to_bf16(activated);
}

kernel void h3_vision_qkv_rope_bf16(
                            device const ushort *qkv [[buffer(0)]],
                            device const ushort *rope_cos [[buffer(1)]],
                            device const ushort *rope_sin [[buffer(2)]],
                            device ushort *query [[buffer(3)]],
                            device ushort *key [[buffer(4)]],
                            device ushort *value [[buffer(5)]],
                            constant qkv_args &args [[buffer(6)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = row_base + inner + head * args.head_dim;
    uint v_base = row_base + inner * 2 + head * args.head_dim;
    uint half_dim = args.rope_half;
    uint rope_index = row * half_dim + dimension % half_dim;
    float c = h3_bf16_to_f32(rope_cos[rope_index]);
    float s = h3_bf16_to_f32(rope_sin[rope_index]);
    uint pair = dimension < half_dim ? dimension + half_dim :
                dimension - half_dim;
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]);
    float q1 = h3_bf16_to_f32(qkv[q_base + pair]);
    float k1 = h3_bf16_to_f32(qkv[k_base + pair]);
    float qr = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    float kr = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(qr);
    key[output_index] = h3_f32_to_bf16(kr);
    value[output_index] = qkv[v_base + dimension];
}

kernel void h3_adaln_bf16(device const ushort *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device const ushort *modulation [[buffer(2)]],
                          device const uint *row_map [[buffer(3)]],
                          device ushort *output [[buffer(4)]],
                          constant adaln_args &args [[buffer(5)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 thread_position [[thread_position_in_threadgroup]],
                          uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Preserve the standalone AdaLN reduction tree while emitting only the one
 * F32 inverse RMS scalar needed by the fused final projection. */
kernel void h3_rms_inverse_bf16(
                          device const ushort *input [[buffer(0)]],
                          device float *inverse [[buffer(1)]],
                          constant norm_args &args [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint3 thread_position
                              [[thread_position_in_threadgroup]],
                          uint3 threadgroup_size
                              [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    device const ushort *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0)
        inverse[row] = rsqrt(reductions[0] / float(args.width) +
                             args.epsilon);
}

struct h3_adaln_linear_args {
    uint rows;
    uint width;
    uint output_dim;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    uint has_bias;
};

/* Apply final AdaLN while loading the existing 16x16 linear tiles. The
 * normalized value is rounded to BF16 before the dot product, retaining the
 * exact arithmetic boundary and accumulation order of AdaLN + linear. */
kernel void h3_adaln_linear_bf16(
                          device const ushort *input [[buffer(0)]],
                          device const float *inverse [[buffer(1)]],
                          device const ushort *norm_weight [[buffer(2)]],
                          device const ushort *modulation [[buffer(3)]],
                          device const uint *row_map [[buffer(4)]],
                          device const ushort *weight [[buffer(5)]],
                          device const ushort *bias [[buffer(6)]],
                          device ushort *output [[buffer(7)]],
                          constant h3_adaln_linear_args &args [[buffer(8)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup ushort input_tile[16][16];
    threadgroup float weight_tile[16][16];
    uint row = group.y * 16 + tid.y;
    uint column = group.x * 16 + tid.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint base = row < args.rows ?
        row_map[row] * args.slots * args.width : 0;
    uint tile_count = (args.width + 15) / 16;
    for (uint tile = 0; tile < tile_count; tile++) {
        uint input_k = tile * 16 + tid.x;
        ushort normalized = 0;
        if (row < args.rows && input_k < args.width) {
            float value = h3_bf16_to_f32(
                input[row * args.width + input_k]);
            float shift = h3_bf16_to_f32(
                modulation[base + args.shift_slot * args.width + input_k]);
            float scale = h3_bf16_to_f32(
                modulation[base + args.scale_slot * args.width + input_k]);
            float normed = value * inverse[row] *
                h3_bf16_to_f32(norm_weight[input_k]);
            normalized = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
        }
        input_tile[tid.y][tid.x] = normalized;
        uint weight_k = tile * 16 + tid.y;
        weight_tile[tid.y][tid.x] =
            column < args.output_dim && weight_k < args.width ?
            h3_bf16_to_f32(weight[column * args.width + weight_k]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++)
            sum = fma(h3_bf16_to_f32(input_tile[tid.y][k]),
                      weight_tile[k][tid.x], sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.output_dim)
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
}

kernel void h3_gate_bf16(device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device ushort *output [[buffer(4)]],
                         constant gate_args &args [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row_map[row] * args.slots * args.width;
    float gate = h3_bf16_to_f32(modulation[base + args.gate_slot * args.width + column]);
    uint index = row * args.width + column;
    float value = h3_bf16_to_f32(residual[index]) +
                  h3_bf16_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16(value);
}

struct h3_gate_adaln_args {
    uint rows;
    uint width;
    uint slots;
    uint gate_slot;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

/* Preserve the gate's BF16 rounding boundary while retaining that rounded row
 * in threadgroup memory for the immediately following AdaLN. */
kernel void h3_gate_adaln_bf16(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device ushort *output [[buffer(7)]],
                         constant h3_gate_adaln_args &args [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    uint base = row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint index = row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* Preserve the exact 256-way reduction tree, but perform its final five
 * stages with SIMD shuffles after the cross-SIMD stages have completed. */
kernel void h3_gate_adaln_bf16_exact_simd(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device ushort *output [[buffer(7)]],
                         constant h3_gate_adaln_args &args [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    uint base = row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint index = row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = reductions[0];
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

/* M5 int8 projections consume AdaLN through a dynamic per-row quantizer. Fold
 * that quantizer into the preceding gated-residual/AdaLN pass. The rounded
 * gated row is overwritten in-place by the rounded AdaLN row once its RMS is
 * known, so exact standalone-quantizer bytes need only one 10.5 KiB row in
 * threadgroup memory and never round-trip through device BF16 memory. */
kernel void h3_gate_adaln_quantize_int8_scalar(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device int8_t *quantized [[buffer(7)]],
                         device float *quantized_scales [[buffer(8)]],
                         constant h3_gate_adaln_args &args [[buffer(9)]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort simdgroup [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         uint row [[threadgroup_position_in_grid]]) {
    if (row >= args.rows) {
        uint quantized_base = row * args.width;
        for (uint column = tid; column < args.width; column += 256)
            quantized[quantized_base + column] = 0;
        if (tid == 0) quantized_scales[row] = 1.0f;
        return;
    }
    threadgroup float reductions[256];
    threadgroup ushort gated_values[5376];
    threadgroup float max_scratch[8];
    uint base = row_map[row] * args.slots * args.width;
    uint row_base = row * args.width;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        uint index = row_base + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        ushort gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = reductions[0];
    float local_max = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        ushort value = h3_f32_to_bf16(
            normalized * (1.0f + scale) + shift);
        gated_values[column] = value;
        local_max = max(local_max, fabs(h3_bf16_to_f32(value)));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, max_scratch, simdgroup, lane);
    float output_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float quantize_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) quantized_scales[row] = output_scale;
    for (uint column = tid; column < args.width; column += 256) {
        int value = (int)rint(
            h3_bf16_to_f32(gated_values[column]) * quantize_scale);
        quantized[row_base + column] = (int8_t)clamp(value, -127, 127);
    }
}

/* The full H3 width is a multiple of four. Load every row stream as four
 * adjacent BF16 values so each pass issues a quarter as many memory
 * instructions while retaining the same BF16 rounding boundaries. */
kernel void h3_gate_adaln_quantize_int8(
                         device const ushort *residual [[buffer(0)]],
                         device const ushort *branch [[buffer(1)]],
                         device const ushort *gate_modulation [[buffer(2)]],
                         device const uint *row_map [[buffer(3)]],
                         device const ushort *weight [[buffer(4)]],
                         device const ushort *norm_modulation [[buffer(5)]],
                         device ushort *gated_residual [[buffer(6)]],
                         device int8_t *quantized [[buffer(7)]],
                         device float *quantized_scales [[buffer(8)]],
                         constant h3_gate_adaln_args &args [[buffer(9)]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort simdgroup [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         uint row [[threadgroup_position_in_grid]]) {
    constexpr uint VECTOR_WIDTH = 5376 / 4;
    if (row >= args.rows) {
        device char4 *quantized4 =
            reinterpret_cast<device char4 *>(quantized);
        uint quantized_base = row * VECTOR_WIDTH;
        for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256)
            quantized4[quantized_base + vector] = char4(0);
        if (tid == 0) quantized_scales[row] = 1.0f;
        return;
    }
    threadgroup float reductions[256];
    threadgroup ushort4 gated_values[VECTOR_WIDTH];
    threadgroup float max_scratch[8];
    uint base = row_map[row] * args.slots * args.width;
    uint vector_base = row * VECTOR_WIDTH;
    uint modulation_base = base / 4;
    device const ushort4 *residual4 =
        reinterpret_cast<device const ushort4 *>(residual);
    device const ushort4 *branch4 =
        reinterpret_cast<device const ushort4 *>(branch);
    device const ushort4 *gate4 =
        reinterpret_cast<device const ushort4 *>(gate_modulation);
    device ushort4 *gated_residual4 =
        reinterpret_cast<device ushort4 *>(gated_residual);
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 residual_bits = residual4[vector_base + vector];
        ushort4 branch_bits = branch4[vector_base + vector];
        ushort4 gate_bits = gate4[modulation_base +
            args.gate_slot * VECTOR_WIDTH + vector];
        float4 gated_float = h3_bf16x4_to_f32(residual_bits) +
            h3_bf16x4_to_f32(branch_bits) * h3_bf16x4_to_f32(gate_bits);
        ushort4 gated = h3_f32x4_to_bf16(gated_float);
        gated_residual4[vector_base + vector] = gated;
        gated_values[vector] = gated;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup ushort *gated_scalars =
        reinterpret_cast<threadgroup ushort *>(gated_values);
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += 256) {
        float value = h3_bf16_to_f32(gated_scalars[column]);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride >= 32; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float sum = reductions[tid];
        for (uint stride = 16; stride; stride >>= 1) {
            float partner = simd_shuffle_down(sum, stride);
            if (tid < stride) sum += partner;
        }
        if (tid == 0)
            reductions[0] = rsqrt(sum / float(args.width) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const ushort4 *weight4 =
        reinterpret_cast<device const ushort4 *>(weight);
    device const ushort4 *norm_modulation4 =
        reinterpret_cast<device const ushort4 *>(norm_modulation);
    float inverse = reductions[0];
    float local_max = 0.0f;
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 gated = gated_values[vector];
        ushort4 norm_bits = weight4[vector];
        ushort4 shift_bits = norm_modulation4[modulation_base +
            args.shift_slot * VECTOR_WIDTH + vector];
        ushort4 scale_bits = norm_modulation4[modulation_base +
            args.scale_slot * VECTOR_WIDTH + vector];
        float4 normalized = h3_bf16x4_to_f32(gated) * inverse *
            h3_bf16x4_to_f32(norm_bits);
        float4 value_float = normalized *
            (float4(1.0f) + h3_bf16x4_to_f32(scale_bits)) +
            h3_bf16x4_to_f32(shift_bits);
        ushort4 value = h3_f32x4_to_bf16(value_float);
        gated_values[vector] = value;
        float4 rounded = h3_bf16x4_to_f32(value);
        local_max = max(local_max,
            max(max(fabs(rounded.x), fabs(rounded.y)),
                max(fabs(rounded.z), fabs(rounded.w))));
    }
    float max_abs = h3_int8_reduce_max(
        local_max, max_scratch, simdgroup, lane);
    float output_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    float quantize_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
    if (tid == 0) quantized_scales[row] = output_scale;
    device char4 *quantized4 = reinterpret_cast<device char4 *>(quantized);
    for (uint vector = tid; vector < VECTOR_WIDTH; vector += 256) {
        ushort4 bits = gated_values[vector];
        int4 value = int4(rint(
            h3_bf16x4_to_f32(bits) * quantize_scale));
        quantized4[vector_base + vector] =
            char4(clamp(value, int4(-127), int4(127)));
    }
}

kernel void h3_qkv_rope_bf16(device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[q_base + d]);
        float k = h3_bf16_to_f32(qkv[k_base + d]);
        q_sum = fma(q, q, q_sum);
        k_sum = fma(k, k, k_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint pair = dimension + args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint pair = dimension - args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[v_base + dimension];
}

/* One SIMD group owns one (row, head), so a 128-thread threadgroup processes
 * four heads.  Lane zero forms its head's norm once in the oracle's original
 * scalar order.  Each lane then owns four dimensions; the +/-64 RoPE partner
 * stays in that same lane.  This removes duplicate reductions while keeping
 * every floating-point expression and rounding boundary unchanged. */
kernel void h3_qkv_rope_bf16_coop_uncached(
                             device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint WIDTH = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }

    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint d = 0; d < WIDTH; d++) {
            float q_element = h3_bf16_to_f32(qkv[q_base + d]);
            float k_element = h3_bf16_to_f32(qkv[k_base + d]);
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) *
                   q_inverse[simdgroup] *
                   h3_bf16_to_f32(q_weight[dimension]);
        float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) *
                   k_inverse[simdgroup] *
                   h3_bf16_to_f32(k_weight[dimension]);
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = h3_bf16_to_f32(qkv[q_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(qkv[k_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + dimension]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + dimension]);
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = h3_bf16_to_f32(qkv[q_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(qkv[k_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + pair]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + pair]);
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        uint output_index =
            (row * args.heads + head) * args.head_dim + dimension;
        query[output_index] = h3_f32_to_bf16(q0);
        key[output_index] = h3_f32_to_bf16(k0);
        value[output_index] = qkv[v_base + dimension];
    }
}

/* Exact cooperative kernel with a per-threadgroup Q/K cache. Coalesced lanes
 * load every projected value once; lane zero retains the oracle's scalar RMS
 * order, and the RoPE epilogue reuses the cached values. Each SIMD group owns
 * disjoint storage, so only SIMD-local barriers are required. */
kernel void h3_qkv_rope_bf16_coop(
                             device const ushort *qkv [[buffer(0)]],
                             device const ushort *q_weight [[buffer(1)]],
                             device const ushort *k_weight [[buffer(2)]],
                             device const ushort *rope_cos [[buffer(3)]],
                             device const ushort *rope_sin [[buffer(4)]],
                             device ushort *query [[buffer(5)]],
                             device ushort *key [[buffer(6)]],
                             device ushort *value [[buffer(7)]],
                             constant qkv_args &args [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint WIDTH = 128;
    constexpr uint HEADS_PER_GROUP = 4;
    uint head = group.x * HEADS_PER_GROUP + simdgroup;
    uint row = group.y;
    if (head >= args.heads || row >= args.sequence) return;
    uint inner = args.heads * args.head_dim;
    uint row_base = row * inner * 3;
    uint q_base = row_base + head * args.head_dim;
    uint k_base = q_base + inner;
    uint v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }

    threadgroup ushort q_values[HEADS_PER_GROUP * WIDTH];
    threadgroup ushort k_values[HEADS_PER_GROUP * WIDTH];
    uint cache_base = simdgroup * WIDTH;
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        q_values[cache_base + dimension] = qkv[q_base + dimension];
        k_values[cache_base + dimension] = qkv[k_base + dimension];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float q_inverse[HEADS_PER_GROUP];
    threadgroup float k_inverse[HEADS_PER_GROUP];
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint dimension = 0; dimension < WIDTH; dimension++) {
            float q_element =
                h3_bf16_to_f32(q_values[cache_base + dimension]);
            float k_element =
                h3_bf16_to_f32(k_values[cache_base + dimension]);
            q_sum = fma(q_element, q_element, q_sum);
            k_sum = fma(k_element, k_element, k_sum);
        }
        q_inverse[simdgroup] =
            rsqrt(q_sum / float(args.head_dim) + args.epsilon);
        k_inverse[simdgroup] =
            rsqrt(k_sum / float(args.head_dim) + args.epsilon);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < WIDTH; dimension += 32) {
        float q0 = h3_bf16_to_f32(q_values[cache_base + dimension]) *
                   q_inverse[simdgroup] *
                   h3_bf16_to_f32(q_weight[dimension]);
        float k0 = h3_bf16_to_f32(k_values[cache_base + dimension]) *
                   k_inverse[simdgroup] *
                   h3_bf16_to_f32(k_weight[dimension]);
        if (dimension < args.rope_half) {
            uint pair = dimension + args.rope_half;
            float q1 = h3_bf16_to_f32(q_values[cache_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(k_values[cache_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + dimension]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + dimension]);
            q0 = q0 * c - q1 * s;
            k0 = k0 * c - k1 * s;
        } else if (dimension < args.rope_half * 2) {
            uint pair = dimension - args.rope_half;
            float q1 = h3_bf16_to_f32(q_values[cache_base + pair]) *
                       q_inverse[simdgroup] *
                       h3_bf16_to_f32(q_weight[pair]);
            float k1 = h3_bf16_to_f32(k_values[cache_base + pair]) *
                       k_inverse[simdgroup] *
                       h3_bf16_to_f32(k_weight[pair]);
            float c = h3_bf16_to_f32(
                rope_cos[row * args.rope_half + pair]);
            float s = h3_bf16_to_f32(
                rope_sin[row * args.rope_half + pair]);
            q0 = q0 * c + q1 * s;
            k0 = k0 * c + k1 * s;
        }
        uint output_index =
            (row * args.heads + head) * args.head_dim + dimension;
        query[output_index] = h3_f32_to_bf16(q0);
        key[output_index] = h3_f32_to_bf16(k0);
        value[output_index] = qkv[v_base + dimension];
    }
}

kernel void h3_swiglu_bf16(device const ushort *fused [[buffer(0)]],
                           device ushort *output [[buffer(1)]],
                           constant swiglu_args &args [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint base = row * args.width * 2;
    float gate = h3_bf16_to_f32(fused[base + column]);
    float up = h3_bf16_to_f32(fused[base + args.width + column]);
    output[row * args.width + column] =
        h3_f32_to_bf16(gate / (1.0f + exp(-gate)) * up);
}

struct embedding_args {
    uint tokens;
    uint vocab_size;
    uint width;
};

kernel void h3_embedding_bf16(device const ushort *weight [[buffer(0)]],
                              device const uint *token_ids [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant embedding_args &args [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint token = gid.y;
    if (token >= args.tokens || column >= args.width) return;
    uint identifier = token_ids[token];
    output[token * args.width + column] = identifier < args.vocab_size ?
        weight[identifier * args.width + column] : ushort(0);
}

struct text_rope_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float epsilon;
};

kernel void h3_text_qk_rope_bf16(
        device const ushort *query_input [[buffer(0)]],
        device const ushort *key_input [[buffer(1)]],
        device const ushort *q_weight [[buffer(2)]],
        device const ushort *k_weight [[buffer(3)]],
        device const ushort *rope_cos [[buffer(4)]],
        device const ushort *rope_sin [[buffer(5)]],
        device ushort *query_output [[buffer(6)]],
        device ushort *key_output [[buffer(7)]],
        constant text_rope_args &args [[buffer(8)]],
        uint3 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (dimension >= args.head_dim || head >= args.query_heads || row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    uint pair = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    float c = h3_bf16_to_f32(rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_to_f32(rope_sin[row * half_dim + (dimension % half_dim)]);

    uint q_base = (row * args.query_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(query_input[q_base + d]);
        q_sum = fma(value, value, q_sum);
    }
    float q_inverse = rsqrt(q_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(query_input[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_to_f32(q_weight[pair]);
    float q_rotated = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16(q_rotated);

    if (head < args.kv_heads) {
        uint k_base = (row * args.kv_heads + head) * args.head_dim;
        float k_sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            float value = h3_bf16_to_f32(key_input[k_base + d]);
            k_sum = fma(value, value, k_sum);
        }
        float k_inverse = rsqrt(k_sum / float(args.head_dim) + args.epsilon);
        float k0 = h3_bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                   h3_bf16_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float k_rotated = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16(k_rotated);
    }
}

struct gqa_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float scale;
};

struct head_norm_args {
    uint sequence;
    uint heads;
    uint head_dim;
    float epsilon;
};

/* Adapted from Iris's Qwen3 BF16 per-head norm: a single thread owns a
 * complete head, making in-place normalization race-free. */
kernel void h3_head_rms_norm_bf16(
        device ushort *tensor [[buffer(0)]],
        device const ushort *weight [[buffer(1)]],
        constant head_norm_args &args [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence || head >= args.heads) return;
    uint base = (row * args.heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        sum = fma(value, value, sum);
    }
    float inverse = rsqrt(sum / float(args.head_dim) + args.epsilon);
    for (uint d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16(
            value * inverse * h3_bf16_to_f32(weight[d]));
    }
}

struct text_rope_inplace_args {
    uint sequence;
    uint query_heads;
    uint kv_heads;
    uint head_dim;
};

/* Iris-style text RoPE. F32 tables avoid compounding table quantization while
 * Q/K remain BF16 at the operation boundary. */
kernel void h3_rope_text_bf16(
        device ushort *query [[buffer(0)]],
        device ushort *key [[buffer(1)]],
        device const float *rope_cos [[buffer(2)]],
        device const float *rope_sin [[buffer(3)]],
        constant text_rope_inplace_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    uint head = gid.y;
    if (row >= args.sequence) return;
    uint half_dim = args.head_dim / 2;
    if (head < args.query_heads) {
        uint base = (row * args.query_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(query[base + d]);
            float second = h3_bf16_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16(first * c - second * s);
            query[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
    if (head < args.kv_heads) {
        uint base = (row * args.kv_heads + head) * args.head_dim;
        for (uint d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(key[base + d]);
            float second = h3_bf16_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16(first * c - second * s);
            key[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
}

kernel void h3_gqa_causal_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *output [[buffer(3)]],
        constant gqa_args &args [[buffer(4)]],
        threadgroup float *scores [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 thread_position [[thread_position_in_threadgroup]],
        uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    uint query_row = group.x;
    uint query_head = group.y;
    if (query_row >= args.sequence || query_head >= args.query_heads) return;
    uint kv_head = query_head / (args.query_heads / args.kv_heads);
    uint q_base = (query_row * args.query_heads + query_head) * args.head_dim;
    uint key_count = query_row + 1;
    threadgroup float reductions[128];
    threadgroup float shared_query[128];

    for (uint d = tid; d < args.head_dim; d += threads) {
        /* Keep Q scaling in F32 through the QK contraction. Rounding the
         * product back to BF16 discards precision without reducing storage. */
        shared_query[d] =
            h3_bf16_to_f32(query[q_base + d]) * args.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        uint k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            dot = fma(shared_query[d], h3_bf16_to_f32(key[k_base + d]), dot);
        }
        float score = dot;
        scores[key_row] = score;
        local_max = max(local_max, score);
    }
    reductions[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] = max(reductions[tid], reductions[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reductions[0];
    float local_sum = 0.0f;
    for (uint key_row = tid; key_row < key_count; key_row += threads) {
        float probability = exp(scores[key_row] - maximum);
        scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint d = tid; d < args.head_dim; d += threads) {
        float sum = 0.0f;
        for (uint key_row = 0; key_row < key_count; key_row++) {
            uint v_index = (key_row * args.kv_heads + kv_head) * args.head_dim + d;
            sum = fma(scores[key_row] * inverse_sum,
                      h3_bf16_to_f32(value[v_index]), sum);
        }
        output[q_base + d] = h3_f32_to_bf16(sum);
    }
}

kernel void h3_add_bf16(device const ushort *left [[buffer(0)]],
                         device const ushort *right [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) +
                                  h3_bf16_to_f32(right[gid]));
}

kernel void h3_sub_bf16(device const ushort *left [[buffer(0)]],
                         device const ushort *right [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         constant uint &count [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) -
                                  h3_bf16_to_f32(right[gid]));
}

struct h3_token_pool_args {
    uint input_offset;
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
};

/* The pair table is [output row, {first, second}]. Singleton rows repeat their
 * source index so non-video prefixes remain bit exact. */
kernel void h3_token_pool_bf16(
                         device const ushort *input [[buffer(0)]],
                         device const uint2 *pairs [[buffer(1)]],
                         device ushort *output [[buffer(2)]],
                         device ushort *baseline [[buffer(3)]],
                         device const uint *baseline_indices [[buffer(4)]],
                         device ushort *original [[buffer(5)]],
                         constant h3_token_pool_args &args [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint2 pair = pairs[row];
    ushort first = input[args.input_offset + pair.x * args.width + column];
    original[args.original_offset + pair.x * args.width + column] = first;
    ushort pooled = first;
    if (pair.x != pair.y) {
        ushort second = input[
            args.input_offset + pair.y * args.width + column];
        original[args.original_offset + pair.y * args.width + column] = second;
        float average = (h3_bf16_to_f32(first) +
                         h3_bf16_to_f32(second)) * 0.5f;
        pooled = h3_f32_to_bf16(average);
    }
    output[row * args.width + column] = pooled;
    uint baseline_index = baseline_indices[row];
    if (baseline_index != 0xffffffffu)
        baseline[args.baseline_offset + baseline_index * args.width + column] =
            pooled;
}

struct h3_token_pool_adaln_args {
    uint input_offset;
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float epsilon;
};

/* Pool and snapshot the full token grid while immediately producing the first
 * reduced block's attention AdaLN. Keeping the pooled BF16 row in threadgroup
 * memory avoids rereading the residual from global memory. */
kernel void h3_token_pool_adaln_bf16(
                         device const ushort *input [[buffer(0)]],
                         device const uint2 *pairs [[buffer(1)]],
                         device ushort *residual [[buffer(2)]],
                         device ushort *baseline [[buffer(3)]],
                         device const uint *baseline_indices [[buffer(4)]],
                         device ushort *original [[buffer(5)]],
                         device const ushort *weight [[buffer(6)]],
                         device const ushort *modulation [[buffer(7)]],
                         device const uint *row_map [[buffer(8)]],
                         device ushort *output [[buffer(9)]],
                         constant h3_token_pool_adaln_args &args
                             [[buffer(10)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort pooled_values[5376];
    uint2 pair = pairs[row];
    uint baseline_index = baseline_indices[row];
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint first_index = args.input_offset +
            pair.x * args.width + column;
        ushort first = input[first_index];
        original[args.original_offset + pair.x * args.width + column] = first;
        ushort pooled = first;
        if (pair.x != pair.y) {
            ushort second = input[args.input_offset +
                pair.y * args.width + column];
            original[args.original_offset +
                     pair.y * args.width + column] = second;
            pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) +
                                     h3_bf16_to_f32(second)) * 0.5f);
        }
        uint destination = row * args.width + column;
        residual[destination] = pooled;
        pooled_values[column] = pooled;
        if (baseline_index != 0xffffffffu)
            baseline[args.baseline_offset +
                     baseline_index * args.width + column] = pooled;
        float value = h3_bf16_to_f32(pooled);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(pooled_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

struct h3_token_expand_args {
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint exact_prefix_rows;
    float update_scale;
};

/* Restore the full grid while retaining the original within-pair detail. The
 * reduced stack contributes only its update relative to the pooled baseline. */
kernel void h3_token_expand_delta_bf16(
                         device const ushort *original [[buffer(0)]],
                         device const ushort *reduced [[buffer(1)]],
                         device const ushort *baseline [[buffer(2)]],
                         device const uint *baseline_indices [[buffer(3)]],
                         device const uint *parents [[buffer(4)]],
                         device ushort *output [[buffer(5)]],
                         constant h3_token_expand_args &args [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (row >= args.rows || column >= args.width) return;
    uint parent = parents[row];
    uint destination = row * args.width + column;
    uint reduced_index = parent * args.width + column;
    if (row < args.exact_prefix_rows) {
        output[destination] = reduced[reduced_index];
    } else {
        uint baseline_row = baseline_indices[parent];
        if (baseline_row == 0xffffffffu) {
            output[destination] = reduced[reduced_index];
            return;
        }
        uint baseline_index = args.baseline_offset +
            baseline_row * args.width + column;
        float update = h3_bf16_to_f32(reduced[reduced_index]) -
                       h3_bf16_to_f32(baseline[baseline_index]);
        output[destination] = h3_f32_to_bf16(
            h3_bf16_to_f32(original[args.original_offset + destination]) +
            args.update_scale * update);
    }
}

struct h3_token_expand_adaln_args {
    uint original_offset;
    uint baseline_offset;
    uint rows;
    uint width;
    uint exact_prefix_rows;
    uint slots;
    uint shift_slot;
    uint scale_slot;
    float update_scale;
    float epsilon;
};

/* H3's hidden width is 5376. Keep the restored BF16 row in 10.5 KiB of
 * threadgroup memory across the reduction instead of writing and then
 * rereading the residual globally. The host caps width at 5376. */
kernel void h3_token_expand_adaln_bf16(
                         device const ushort *original [[buffer(0)]],
                         device const ushort *reduced [[buffer(1)]],
                         device const ushort *baseline [[buffer(2)]],
                         device const uint *baseline_indices [[buffer(3)]],
                         device const uint *parents [[buffer(4)]],
                         device ushort *residual [[buffer(5)]],
                         device const ushort *weight [[buffer(6)]],
                         device const ushort *modulation [[buffer(7)]],
                         device const uint *row_map [[buffer(8)]],
                         device ushort *output [[buffer(9)]],
                         constant h3_token_expand_adaln_args &args
                             [[buffer(10)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 thread_position
                             [[thread_position_in_threadgroup]],
                         uint3 threadgroup_size [[threads_per_threadgroup]]) {
    uint row = group.x;
    uint tid = thread_position.x;
    uint threads = threadgroup_size.x;
    if (row >= args.rows) return;
    threadgroup float reductions[256];
    threadgroup ushort restored_values[5376];
    uint parent = parents[row];
    uint baseline_row = baseline_indices[parent];
    bool direct = row < args.exact_prefix_rows ||
                  baseline_row == 0xffffffffu;
    float local_sum = 0.0f;
    for (uint column = tid; column < args.width; column += threads) {
        uint destination = row * args.width + column;
        uint reduced_index = parent * args.width + column;
        ushort restored = reduced[reduced_index];
        if (!direct) {
            uint baseline_index = args.baseline_offset +
                baseline_row * args.width + column;
            float update = h3_bf16_to_f32(restored) -
                           h3_bf16_to_f32(baseline[baseline_index]);
            restored = h3_f32_to_bf16(
                h3_bf16_to_f32(
                    original[args.original_offset + destination]) +
                args.update_scale * update);
        }
        restored_values[column] = restored;
        residual[destination] = restored;
        float value = h3_bf16_to_f32(restored);
        local_sum = fma(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(reductions[0] / float(args.width) + args.epsilon);
    uint base = row_map[row] * args.slots * args.width;
    for (uint column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(restored_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

struct h3_euler_args {
    uint sample_offset;
    uint elements;
    float delta;
    float ratio;
};

kernel void h3_euler_bf16(device float *sample [[buffer(0)]],
                           device const ushort *last [[buffer(1)]],
                           device const ushort *previous [[buffer(2)]],
                           constant h3_euler_args &args [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= args.elements) return;
    float last_value = h3_bf16_to_f32(last[gid]);
    float velocity = fma(args.ratio,
                         last_value - h3_bf16_to_f32(previous[gid]),
                         last_value);
    uint sample_index = args.sample_offset + gid;
    sample[sample_index] = fma(args.delta, velocity, sample[sample_index]);
}

kernel void h3_silu_mul_bf16(device const ushort *gate [[buffer(0)]],
                              device const ushort *up [[buffer(1)]],
                              device ushort *output [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = h3_bf16_to_f32(gate[gid]);
    float other = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + exp(-value)) * other);
}
