#include "h3.h"
#include "h3_cli.h"
#include "h3_host.h"
#include "h3_terminal.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s -d MODEL_DIR [options]              # interactive (video only)\n"
        "       %s -d MODEL_DIR -p PROMPT [-o OUTPUT] [options]\n"
        "       %s -d MODEL_DIR --info\n\n"
        "A MiniMax-H3-Image directory (transformer_diffusers/) selects text-to-image:\n"
        "  %s -d MiniMax-H3-Image -p PROMPT [--size WxH] [--min-generation-area N]\n"
        "      [--seed N] [--ssd-streaming] [-o out.png]\n"
        "  %s -d MiniMax-H3-Image -p PROMPT --ref-image PATH [--ref-image PATH ...]\n"
        "      edits: the prompt calls the references <Picture 1>, <Picture 2>, ...\n"
        "The interactive session and the video-only options below (--steps, --reuse,\n"
        "--core-reuse, --token-reduction, --frames, --seconds, --first-frame,\n"
        "--last-frame, --ref-image-size, --ref-video and the other --ref-* options,\n"
        "--render-*, --layers other than 50) do not apply to image directories.\n"
        "Image weight modes: resident BF16 by default (about 36 GiB of GPU buffers,\n"
        "fast); --int8 (about 18.6 GiB, a little faster, less exact); --ssd-streaming\n"
        "(exact BF16, about 1.5 GiB, slower). --int8 and --ssd-streaming exclude\n"
        "each other.\n\n",
        program, program, program, program, program);
    fprintf(stderr,
        "Options:\n"
        "  -d, --model-dir PATH   MiniMax-H3 or MiniMax-H3-Image local directory\n"
        "  -p, --prompt TEXT      Raw H3 prompt\n"
        "  -o, --output PATH      Output MP4, or PNG for images (default:\n"
        "                         outputs/h3.mp4, outputs/h3.png)\n"
        "      --image            Require an image directory (error otherwise)\n"
        "      --size WxH         Output size (image default: 1024x1024, or the size of\n"
        "                         <Picture 1> when editing; the generation canvas is\n"
        "                         limited to 33554432 pixels, 8192x4096)\n"
        "      --min-generation-area N  Image canvas area floor (default: 4194304,\n"
        "                         range 1024..16777216; 1024 disables enlarging)\n"
        "      --width N          Output width (default: 864)\n"
        "      --height N         Output height (default: 480)\n"
        "      --render-width N   Lower internal model width (optional)\n"
        "      --render-height N  Lower internal model height (optional)\n"
        "      --frames N         Requested frames (default: 56)\n"
        "      --seconds N        Requested duration at 24 fps (instead of --frames)\n"
        "      --steps N          Denoising passes (default: 20)\n"
        "      --reuse N          Denoiser reuse: 1 close, 2 fast, 3 aggressive\n"
        "      --layers N         DiT blocks: 50 exact, 45 fast, 40 aggressive\n"
        "      --core-reuse N     Core refresh: 1 exact, 4 fast, 6 aggressive\n"
        "      --token-reduction  Pair video tokens in middle DiT blocks\n"
        "      --ssd-streaming    Stream original BF16 DiT layers from SSD\n"
        "      --no-cache         Images: do not read or write the text embedding\n"
        "                         cache ($H3_CACHE_DIR, default ~/.h3_cache/edit;\n"
        "                         $H3_CACHE_BYTES caps it, default 2 GiB)\n"
        "      --int8             Images: resident int8 DiT (about 18.6 GiB GPU buffers,\n"
        "                         about twice the per-step error of BF16)\n"
        "      --use-int8-row-fc2 Faster one-scale int8 FC2 (M5)\n"
        "      --use-reference-rope  Disable native 256 RoPE adaptation\n"
        "      --use-slower-bf16-mlp  Force close-reference BF16/MPS MLP\n"
        "      --use-slower-bf16-qkv  Force close-reference BF16 QKV\n"
        "      --use-slower-bf16-attention-output  Force BF16 attention output\n"
        "      --use-slower-row-major-attention-output  Restore SDPA transpose\n"
        "      --use-slower-unfused-int8-inputs  Keep standalone quantizers\n"
        "      --use-slower-unfused-qkv-rope  Keep separate Q/K norm/RoPE\n"
        "      --use-slower-scalar-qkv-rms  Force scalar Q/K RMS loads\n"
        "      --use-slower-uncached-int8-scales  Reread projection scales\n"
        "      --use-slower-dynamic-fc1-k  Use runtime-bound FC1 K loop\n"
        "      --use-slower-grouped-quantizer  Force 256-thread FC2 quantizer\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame conditioning image\n"
        "      --last-frame PATH  Last-frame conditioning image\n"
        "      --ref-image PATH    Video: append an ordered Ref2VA image. Images: edit\n"
        "                         with a reference, repeatable up to 8 (each side at\n"
        "                         least 32 pixels; the prompt names it <Picture N>)\n"
        "      --ref-image-size S  Image sizing: match (default) or max\n"
        "      --ref-video PATH    Append video, including embedded audio\n"
        "      --ref-silent-video PATH  Append video without its audio\n"
        "      --ref-video-audio VIDEO AUDIO  Append video + soundtrack\n"
        "      --ref-audio PATH    Append an ordered standalone audio clip\n"
        "      --frames-dir PATH  Write generated frames as PPM files\n"
        "      --show             Display a frame after every denoising step (M5)\n"
        "      --zoom N           Terminal image zoom (default: 2 for Retina)\n"
        "      --profile          Print per-phase timing, allocation and memory data\n"
        "      --info             Inspect model/device without mapping weights\n"
        "  -h, --help             Show this help\n");
}

static int parse_int(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (int)parsed;
}

static int frames_from_seconds(const char *value) {
    char *end = NULL;
    errno = 0;
    double seconds = strtod(value, &end);
    double frames = seconds * (double)H3_FPS;
    if (errno || !end || *end || !isfinite(seconds) || seconds <= 0.0 ||
        !isfinite(frames) || frames > (double)INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    long long rounded = llround(frames);
    if (rounded < 1 || rounded > INT32_MAX) {
        fprintf(stderr, "h3: invalid seconds: %s\n", value);
        exit(2);
    }
    return (int)rounded;
}

static uint64_t parse_u64(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "h3: invalid %s: %s\n", label, value);
        exit(2);
    }
    return (uint64_t)parsed;
}

#define MAX_CLI_REFERENCES 12

static h3_reference *append_reference(
        h3_reference references[MAX_CLI_REFERENCES], size_t *count) {
    if (*count >= MAX_CLI_REFERENCES) {
        fprintf(stderr, "h3: Ref2VA supports at most 12 references\n");
        exit(2);
    }
    h3_reference *reference = &references[(*count)++];
    memset(reference, 0, sizeof(*reference));
    return reference;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_component(const char *label, const h3_component_info *item) {
    printf("  %-18s %2zu files  %4zu tensors  %7.3f GiB\n",
           label, item->files, item->tensors, gib(item->tensor_bytes));
}

static void print_info(const h3_ctx *ctx) {
    const h3_device_info *device = h3_device(ctx);
    const h3_model_info *model = h3_model(ctx);
    printf("h3-metal %s\n", H3_VERSION);
    printf("Device: %s (%s)\n", device->name, device->architecture);
    printf("  physical memory       %.1f GiB\n", gib(device->physical_memory));
    printf("  recommended GPU set   %.1f GiB\n", gib(device->recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(device->max_buffer_length));
    printf("  Apple GPU family      %d\n", device->apple_gpu_family);
    printf("  Metal 4               %s\n", device->metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", device->unified_memory ? "yes" : "no");
    printf("Native checkpoint inventory (header-only):\n");
    print_component("Qwen3-VL encoder", &model->text_encoder);
    if (h3_is_image_model(ctx)) {
        print_component("image DiT", &model->image_transformer);
        print_component("image VAE", &model->image_vae);
        return;
    }
    print_component("FL2VA DiT", &model->fl2va_transformer);
    print_component("Ref2VA DiT", &model->ref2va_transformer);
    print_component("video VAE", &model->video_vae);
    print_component("audio VAE", &model->audio_vae);
}

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
    h3_terminal_protocol terminal;
    int display_failed;
    const char *frames_dir;
    int frame_write_failed;
} cli_state;

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    cli_state *state = opaque;
    if (!strcmp(state->phase, phase) && state->completed == completed &&
        state->total == total) return 0;
    if (strcmp(state->phase, phase)) {
        if (state->active) fputc('\n', stderr);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    state->completed = completed;
    state->total = total;
    state->active = completed < total;
    fprintf(stderr, "\r%-25s %4d/%-4d", phase, completed, total);
    if (!state->active) fputc('\n', stderr);
    fflush(stderr);
    return 0;
}

typedef struct {
    cli_state base;
    struct timespec started;
    struct timespec phase_started;
    char phase[64];
} image_cli_state;

static double seconds_between(const struct timespec *from,
                              const struct timespec *to) {
    return (double)(to->tv_sec - from->tv_sec) +
           (double)(to->tv_nsec - from->tv_nsec) * 1e-9;
}

static void image_phase_end(image_cli_state *state) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (state->phase[0])
        fprintf(stderr, "h3: %-22s %7.1f s\n", state->phase,
                seconds_between(&state->phase_started, &now));
    state->phase[0] = '\0';
}

static int image_progress(const char *phase, int completed, int total,
                          void *opaque) {
    image_cli_state *state = opaque;
    if (strcmp(state->phase, phase)) {
        if (state->base.active) {
            fputc('\n', stderr);
            state->base.active = 0;
        }
        image_phase_end(state);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
        clock_gettime(CLOCK_MONOTONIC, &state->phase_started);
    }
    return cli_progress(phase, completed, total, &state->base);
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    cli_state *state = opaque;
    int preview = frame->denoise_step >= 0;
    if (!preview && state->frames_dir && !state->frame_write_failed) {
        char path[1024];
        int length = snprintf(path, sizeof(path), "%s/frame-%04d.ppm",
                              state->frames_dir, frame->frame_index);
        FILE *output = length > 0 && (size_t)length < sizeof(path) ?
            fopen(path, "wb") : NULL;
        if (!output ||
            fprintf(output, "P6\n%d %d\n255\n", frame->width,
                    frame->height) < 0) {
            fprintf(stderr, "h3: cannot write frame %d to %s\n",
                    frame->frame_index, state->frames_dir);
            if (output) fclose(output);
            state->frame_write_failed = 1;
        } else {
            size_t row_bytes = (size_t)frame->width * 3;
            for (int row = 0; row < frame->height; row++) {
                if (fwrite(frame->rgb + (size_t)row * frame->stride, 1,
                           row_bytes, output) != row_bytes) {
                    state->frame_write_failed = 1;
                    break;
                }
            }
            if (fclose(output) != 0) state->frame_write_failed = 1;
            if (state->frame_write_failed)
                fprintf(stderr, "h3: incomplete frame %d in %s\n",
                        frame->frame_index, state->frames_dir);
        }
    }
    if (state->frame_write_failed) return 1;
    if (state->display_failed || state->terminal == H3_TERM_NONE) return 0;
    if (state->active) {
        fputc('\n', stderr);
        state->active = 0;
    }
    if (preview)
        fprintf(stderr,
                "h3: denoise preview %d/%d, video frame %d/%d via %s\n",
                frame->denoise_step + 1, frame->denoise_steps,
                frame->frame_index + 1, frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    else
        fprintf(stderr, "h3: frame %d/%d via %s\n", frame->frame_index + 1,
                frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3: terminal display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    enum { OPT_WIDTH = 1000, OPT_HEIGHT, OPT_RENDER_WIDTH, OPT_RENDER_HEIGHT,
           OPT_FRAMES, OPT_SECONDS, OPT_STEPS, OPT_REUSE,
           OPT_LAYERS,
           OPT_CORE_REUSE,
           OPT_TOKEN_REDUCTION,
           OPT_SSD_STREAMING,
           OPT_USE_INT8_ROW_FC2,
           OPT_USE_REFERENCE_ROPE,
           OPT_USE_SLOWER_BF16_MLP,
           OPT_USE_SLOWER_BF16_QKV,
           OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT,
           OPT_USE_SLOWER_UNFUSED_INT8_INPUTS,
           OPT_USE_SLOWER_UNFUSED_QKV_ROPE,
           OPT_USE_SLOWER_SCALAR_QKV_RMS,
           OPT_USE_SLOWER_UNCACHED_INT8_SCALES,
           OPT_USE_SLOWER_DYNAMIC_FC1_K,
           OPT_USE_SLOWER_GROUPED_QUANTIZER,
           OPT_SEED,
           OPT_FIRST, OPT_LAST, OPT_REF_IMAGE, OPT_REF_IMAGE_SIZE,
           OPT_REF_VIDEO, OPT_REF_SILENT_VIDEO, OPT_REF_VIDEO_AUDIO,
           OPT_REF_AUDIO, OPT_FRAMES_DIR, OPT_SHOW, OPT_ZOOM,
           OPT_PROFILE, OPT_INFO, OPT_IMAGE, OPT_SIZE, OPT_MIN_AREA, OPT_INT8,
           OPT_NO_CACHE };
    static const struct option options[] = {
        {"model-dir", required_argument, NULL, 'd'},
        {"prompt", required_argument, NULL, 'p'},
        {"output", required_argument, NULL, 'o'},
        {"width", required_argument, NULL, OPT_WIDTH},
        {"height", required_argument, NULL, OPT_HEIGHT},
        {"render-width", required_argument, NULL, OPT_RENDER_WIDTH},
        {"render-height", required_argument, NULL, OPT_RENDER_HEIGHT},
        {"frames", required_argument, NULL, OPT_FRAMES},
        {"seconds", required_argument, NULL, OPT_SECONDS},
        {"steps", required_argument, NULL, OPT_STEPS},
        {"reuse", required_argument, NULL, OPT_REUSE},
        {"layers", required_argument, NULL, OPT_LAYERS},
        {"core-reuse", required_argument, NULL, OPT_CORE_REUSE},
        {"token-reduction", no_argument, NULL, OPT_TOKEN_REDUCTION},
        {"ssd-streaming", no_argument, NULL, OPT_SSD_STREAMING},
        {"use-int8-row-fc2", no_argument, NULL, OPT_USE_INT8_ROW_FC2},
        {"use-reference-rope", no_argument, NULL, OPT_USE_REFERENCE_ROPE},
        {"use-slower-bf16-mlp", no_argument, NULL,
         OPT_USE_SLOWER_BF16_MLP},
        {"use-slower-bf16-qkv", no_argument, NULL,
         OPT_USE_SLOWER_BF16_QKV},
        {"use-slower-bf16-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT},
        {"use-slower-row-major-attention-output", no_argument, NULL,
         OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT},
        {"use-slower-unfused-int8-inputs", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_INT8_INPUTS},
        {"use-slower-unfused-qkv-rope", no_argument, NULL,
         OPT_USE_SLOWER_UNFUSED_QKV_ROPE},
        {"use-slower-scalar-qkv-rms", no_argument, NULL,
         OPT_USE_SLOWER_SCALAR_QKV_RMS},
        {"use-slower-uncached-int8-scales", no_argument, NULL,
         OPT_USE_SLOWER_UNCACHED_INT8_SCALES},
        {"use-slower-dynamic-fc1-k", no_argument, NULL,
         OPT_USE_SLOWER_DYNAMIC_FC1_K},
        {"use-slower-grouped-quantizer", no_argument, NULL,
         OPT_USE_SLOWER_GROUPED_QUANTIZER},
        {"seed", required_argument, NULL, OPT_SEED},
        {"first-frame", required_argument, NULL, OPT_FIRST},
        {"last-frame", required_argument, NULL, OPT_LAST},
        {"ref-image", required_argument, NULL, OPT_REF_IMAGE},
        {"ref-image-size", required_argument, NULL, OPT_REF_IMAGE_SIZE},
        {"ref-video", required_argument, NULL, OPT_REF_VIDEO},
        {"ref-silent-video", required_argument, NULL, OPT_REF_SILENT_VIDEO},
        {"ref-video-audio", required_argument, NULL, OPT_REF_VIDEO_AUDIO},
        {"ref-audio", required_argument, NULL, OPT_REF_AUDIO},
        {"frames-dir", required_argument, NULL, OPT_FRAMES_DIR},
        {"show", no_argument, NULL, OPT_SHOW},
        {"zoom", required_argument, NULL, OPT_ZOOM},
        {"profile", no_argument, NULL, OPT_PROFILE},
        {"info", no_argument, NULL, OPT_INFO},
        {"image", no_argument, NULL, OPT_IMAGE},
        {"int8", no_argument, NULL, OPT_INT8},
        {"no-cache", no_argument, NULL, OPT_NO_CACHE},
        {"size", required_argument, NULL, OPT_SIZE},
        {"min-generation-area", required_argument, NULL, OPT_MIN_AREA},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };
    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = NULL;
    const char *video_only = NULL;
    int image_flag = 0;
    int int8_flag = 0;
    int no_cache_flag = 0;
    int width_given = 0, height_given = 0;
    int min_area = H3_IMAGE_DEFAULT_MIN_AREA;
    int min_area_given = 0;
    h3_params params = H3_PARAMS_DEFAULT;
    h3_reference references[MAX_CLI_REFERENCES];
    size_t reference_count = 0;
    cli_state cli = {{0}, 0, -1, -1, H3_TERM_NONE, 0, NULL, 0};
    int show = 0;
    int profile = 0;
    int info = 0;
    int frames_given = 0;
    int seconds_given = 0;
    int seed_given = 0;
    int option;
    while ((option = getopt_long(argc, argv, "d:p:o:h", options, NULL)) != -1) {
        switch (option) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'h': usage(argv[0]); return 0;
            case OPT_WIDTH:
                params.width = parse_int(optarg, "width");
                width_given = 1;
                break;
            case OPT_HEIGHT:
                params.height = parse_int(optarg, "height");
                height_given = 1;
                break;
            case OPT_SIZE:
                if (!h3_parse_size(optarg, &params.width,
                                         &params.height)) {
                    fprintf(stderr, "h3: --size must be WxH: %s\n", optarg);
                    return 2;
                }
                width_given = height_given = 1;
                break;
            case OPT_IMAGE: image_flag = 1; break;
            case OPT_INT8: int8_flag = 1; break;
            case OPT_NO_CACHE: no_cache_flag = 1; break;
            case OPT_MIN_AREA:
                min_area = parse_int(optarg, "min generation area");
                min_area_given = 1;
                break;
            case OPT_RENDER_WIDTH: video_only = video_only ? video_only : "--render-width";
                params.render_width = parse_int(optarg, "render width");
                break;
            case OPT_RENDER_HEIGHT: video_only = video_only ? video_only : "--render-height";
                params.render_height = parse_int(optarg, "render height");
                break;
            case OPT_FRAMES: video_only = video_only ? video_only : "--frames";
                params.frames = parse_int(optarg, "frames");
                frames_given = 1;
                break;
            case OPT_SECONDS: video_only = video_only ? video_only : "--seconds";
                params.frames = frames_from_seconds(optarg);
                seconds_given = 1;
                break;
            case OPT_STEPS: video_only = video_only ? video_only : "--steps"; params.steps = parse_int(optarg, "steps"); break;
            case OPT_REUSE: video_only = video_only ? video_only : "--reuse";
                params.denoise_reuse = parse_int(optarg, "reuse");
                break;
            case OPT_LAYERS:
                params.dit_layers = parse_int(optarg, "layers");
                break;
            case OPT_CORE_REUSE: video_only = video_only ? video_only : "--core-reuse";
                params.core_reuse = parse_int(optarg, "core reuse");
                break;
            case OPT_TOKEN_REDUCTION:
                params.token_reduction = 1;
                video_only = video_only ? video_only : "--token-reduction";
                break;
            case OPT_SSD_STREAMING: params.ssd_streaming = 1; break;
            case OPT_USE_INT8_ROW_FC2:
                params.use_int8_row_fc2 = 1;
                break;
            case OPT_USE_REFERENCE_ROPE: video_only = video_only ? video_only : "--use-reference-rope";
                params.use_reference_rope = 1;
                break;
            case OPT_USE_SLOWER_BF16_MLP:
                params.use_slower_bf16_mlp = 1;
                break;
            case OPT_USE_SLOWER_BF16_QKV:
                params.use_slower_bf16_qkv = 1;
                break;
            case OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT:
                params.use_slower_bf16_attention_output = 1;
                break;
            case OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT:
                params.use_slower_row_major_attention_output = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_INT8_INPUTS:
                params.use_slower_unfused_int8_inputs = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_QKV_ROPE:
                params.use_slower_unfused_qkv_rope = 1;
                break;
            case OPT_USE_SLOWER_SCALAR_QKV_RMS:
                params.use_slower_scalar_qkv_rms = 1;
                break;
            case OPT_USE_SLOWER_UNCACHED_INT8_SCALES:
                params.use_slower_uncached_int8_scales = 1;
                break;
            case OPT_USE_SLOWER_DYNAMIC_FC1_K:
                params.use_slower_dynamic_fc1_k = 1;
                break;
            case OPT_USE_SLOWER_GROUPED_QUANTIZER:
                params.use_slower_grouped_quantizer = 1;
                break;
            case OPT_SEED:
                params.seed = parse_u64(optarg, "seed");
                seed_given = 1;
                break;
            case OPT_FIRST: video_only = video_only ? video_only : "--first-frame"; params.first_frame = optarg; break;
            case OPT_LAST: video_only = video_only ? video_only : "--last-frame"; params.last_frame = optarg; break;
            case OPT_REF_IMAGE: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_IMAGE;
                reference->path = optarg;
                break;
            }
            case OPT_REF_IMAGE_SIZE: video_only = video_only ? video_only : "--ref-image-size";
                if (!strcmp(optarg, "match"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
                else if (!strcmp(optarg, "max"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MAX;
                else {
                    fprintf(stderr,
                        "h3: --ref-image-size must be match or max\n");
                    return 2;
                }
                break;
            case OPT_REF_VIDEO: video_only = video_only ? video_only : "--ref-video"; {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 1;
                break;
            }
            case OPT_REF_SILENT_VIDEO: video_only = video_only ? video_only : "--ref-silent-video"; {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 0;
                break;
            }
            case OPT_REF_VIDEO_AUDIO: video_only = video_only ? video_only : "--ref-video-audio"; {
                if (optind >= argc) {
                    fprintf(stderr,
                        "h3: --ref-video-audio requires VIDEO and AUDIO\n");
                    return 2;
                }
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO_AUDIO;
                reference->path = optarg;
                reference->audio_path = argv[optind++];
                break;
            }
            case OPT_REF_AUDIO: video_only = video_only ? video_only : "--ref-audio"; {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_AUDIO;
                reference->path = optarg;
                break;
            }
            case OPT_FRAMES_DIR: video_only = video_only ? video_only : "--frames-dir"; cli.frames_dir = optarg; break;
            case OPT_SHOW: show = 1; break;
            case OPT_ZOOM:
                if (!h3_terminal_set_zoom(parse_int(optarg, "zoom"))) {
                    fprintf(stderr, "h3: --zoom must be at least 1\n");
                    return 2;
                }
                break;
            case OPT_PROFILE: profile = 1; break;
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    if (!model_dir) {
        usage(argv[0]);
        return 2;
    }
    if (frames_given && seconds_given) {
        fprintf(stderr, "h3: --seconds and --frames are mutually exclusive\n");
        return 2;
    }
    if (prompt && params.steps >= 2 && params.steps <= 7 &&
        params.denoise_reuse > 1) {
        fprintf(stderr,
            "h3: warning: --reuse with only %d denoising steps leaves very "
            "few fresh model evaluations\n", params.steps);
    }
    params.references = references;
    params.reference_count = reference_count;
    if (output && !*output) {
        fprintf(stderr, "h3: -o needs a path\n");
        return 2;
    }
    if (profile) setenv("H3_PROFILE", "1", 1);
    h3_ctx *ctx = h3_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "h3: %s\n", h3_last_error(NULL));
        return 1;
    }
    int image = h3_is_image_model(ctx);
    const char *image_error = NULL;
    char image_error_text[128];
    if (image_flag && !image)
        image_error = "--image requires a MiniMax-H3-Image model directory";
    else if (!image && min_area_given)
        image_error = "--min-generation-area applies to image models only";
    else if (!image && int8_flag)
        image_error = "--int8 applies to image models only";
    else if (!image && no_cache_flag)
        image_error = "--no-cache applies to image models only";
    else if (image && int8_flag && params.ssd_streaming)
        image_error = "--int8 and --ssd-streaming exclude each other";
    else if (image && reference_count > H3_IMAGE_MAX_REFERENCES) {
        snprintf(image_error_text, sizeof(image_error_text),
                 "at most %d --ref-image options are accepted for image "
                 "editing, got %zu", H3_IMAGE_MAX_REFERENCES, reference_count);
        image_error = image_error_text;
    } else if (image && video_only) {
        snprintf(image_error_text, sizeof(image_error_text),
                 "%s does not apply to image models", video_only);
        image_error = image_error_text;
    } else if (image && params.dit_layers != H3_DEFAULT_DIT_LAYERS)
        image_error = "--layers does not apply to image models (all 50 "
                      "blocks run)";
    else if (image && width_given != height_given)
        image_error = "give both --width and --height, or --size WxH";
    else if (image && !prompt && !info)
        image_error = "the interactive session is video-only; pass -p "
                      "PROMPT to generate an image";
    if (image_error) {
        fprintf(stderr, "h3: %s\n", image_error);
        h3_free(ctx);
        return 2;
    }
    if (cli.frames_dir && mkdir(cli.frames_dir, 0755) != 0 &&
        errno != EEXIST) {
        fprintf(stderr, "h3: cannot create frames directory %s: %s\n",
                cli.frames_dir, strerror(errno));
        h3_free(ctx);
        return 1;
    }
    if (info) print_info(ctx);
    if (image && prompt) {
        h3_image_params image_params = H3_IMAGE_PARAMS_DEFAULT;
        h3_image_reference edit_references[MAX_CLI_REFERENCES];
        for (size_t index = 0; index < reference_count; index++)
            edit_references[index].path = references[index].path;
        image_params.references = edit_references;
        image_params.reference_count = reference_count;
        if (width_given) {
            image_params.width = params.width;
            image_params.height = params.height;
        } else if (reference_count) {
            image_params.width = image_params.height = 0;
        }
        image_params.min_generation_area = min_area;
        if (!no_cache_flag) {
            const char *home = getenv("HOME");
            const char *configured = getenv("H3_CACHE_DIR");
            char fallback[PATH_MAX];
            if (!configured && home && *home &&
                snprintf(fallback, sizeof(fallback), "%s/.h3_cache/edit",
                         home) < (int)sizeof(fallback))
                configured = fallback;
            h3_cache_set_directory(ctx, configured);
            if (!configured)
                fprintf(stderr, "h3: %s disabled: no cache directory\n",
                        reference_count ? "edit text cache" : "text cache");
        } else {
            fprintf(stderr, "h3: %s disabled\n",
                    reference_count ? "edit text cache" : "text cache");
        }
        image_params.seed = params.seed;
        image_params.output_path = output ? output : "outputs/h3.png";
        image_params.dit_layers = params.dit_layers;
        image_params.use_int8 = int8_flag;
        image_params.use_int8_row_fc2 = params.use_int8_row_fc2;
        image_params.ssd_streaming = params.ssd_streaming;
        image_params.use_slower_bf16_mlp = params.use_slower_bf16_mlp;
        image_params.use_slower_bf16_qkv = params.use_slower_bf16_qkv;
        image_params.use_slower_bf16_attention_output =
            params.use_slower_bf16_attention_output;
        image_params.use_slower_row_major_attention_output =
            params.use_slower_row_major_attention_output;
        image_params.use_slower_unfused_int8_inputs =
            params.use_slower_unfused_int8_inputs;
        image_params.use_slower_unfused_qkv_rope =
            params.use_slower_unfused_qkv_rope;
        image_params.use_slower_scalar_qkv_rms =
            params.use_slower_scalar_qkv_rms;
        image_params.use_slower_uncached_int8_scales =
            params.use_slower_uncached_int8_scales;
        image_params.use_slower_dynamic_fc1_k =
            params.use_slower_dynamic_fc1_k;
        image_params.use_slower_grouped_quantizer =
            params.use_slower_grouped_quantizer;
        image_cli_state image_cli;
        memset(&image_cli, 0, sizeof(image_cli));
        image_cli.base = cli;
        image_cli.base.completed = image_cli.base.total = -1;
        image_params.on_progress = image_progress;
        image_params.callback_opaque = &image_cli;
        if (show) {
            image_cli.base.terminal = h3_terminal_detect();
            if (image_cli.base.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3: graphical output uses %s\n",
                        h3_terminal_protocol_name(image_cli.base.terminal));
                image_params.on_frame = cli_frame;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &image_cli.started);
        h3_result *result = h3_generate_image(ctx, prompt, &image_params);
        if (image_cli.base.active) fputc('\n', stderr);
        image_phase_end(&image_cli);
        if (!result) {
            fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            return 1;
        }
        struct timespec finished;
        clock_gettime(CLOCK_MONOTONIC, &finished);
        fprintf(stderr, "h3: %-22s %7.1f s\n", "total",
                seconds_between(&image_cli.started, &finished));
        h3_result_free(result);
        fprintf(stderr, "h3: wrote %s\n", image_params.output_path);
    } else if (prompt) {
        params.output_path = output ? output : "outputs/h3.mp4";
        params.on_progress = cli_progress;
        params.callback_opaque = &cli;
        if (cli.frames_dir) params.on_frame = cli_frame;
        if (show) {
            cli.terminal = h3_terminal_detect();
            if (cli.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3: graphical output uses %s\n",
                        h3_terminal_protocol_name(cli.terminal));
                params.on_frame = cli_frame;
                params.preview_denoise = 1;
            }
        }
        h3_result *result = h3_generate(ctx, prompt, &params);
        if (!result) {
            if (cli.active) fputc('\n', stderr);
            fprintf(stderr, "h3: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            return 1;
        }
        h3_result_free(result);
        fprintf(stderr, "h3: wrote %s\n", params.output_path);
        if (cli.frames_dir)
            fprintf(stderr, "h3: wrote frames to %s\n", cli.frames_dir);
    } else if (!info) {
        int cli_status = h3_cli_run(ctx, model_dir, &params, show, seed_given);
        h3_free(ctx);
        return cli_status;
    }
    h3_free(ctx);
    return 0;
}
