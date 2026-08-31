PLATFORM ?= $(shell uname -s)
.DEFAULT_GOAL := all
AR ?= ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion

ifeq ($(PLATFORM),Darwin)
CC ?= clang
OBJCFLAGS := $(CFLAGS) -D_DARWIN_C_SOURCE -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm
DEVICE_LDLIBS := $(FRAMEWORKS)
DEVICE_SRC := h3_metal.m
GPU_SRC := h3_gpu.m
TOKENIZER_SRC := h3_tokenizer.m
TOKENIZER_OBJ := h3_tokenizer_metal.o
BACKEND := metal
PLATFORM_LD := $(CC)
else ifeq ($(PLATFORM),Linux)
CC ?= cc
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
NVCC_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(NVCC_ARCH) -Xcompiler=-Wall,-Wextra,-Wshadow
CPPFLAGS += -D_POSIX_C_SOURCE=200809L -I$(CUDA_HOME)/include
CUDA_LDLIBS := -L$(CUDA_HOME)/lib64 -lcudart -lcublasLt
CUDNN_ROOT ?=
CUDNN_FRONTEND_ROOT ?=
ifneq ($(strip $(CUDNN_ROOT)$(CUDNN_FRONTEND_ROOT)),)
ifeq ($(strip $(CUDNN_ROOT)),)
$(error CUDNN_ROOT is required when enabling cuDNN attention)
endif
ifeq ($(strip $(CUDNN_FRONTEND_ROOT)),)
$(error CUDNN_FRONTEND_ROOT is required when enabling cuDNN attention)
endif
CPPFLAGS += -DH3_USE_CUDNN -isystem $(CUDNN_ROOT)/include \
	-isystem $(CUDNN_FRONTEND_ROOT)/include
CUDNN_LDLIBS := -L$(CUDNN_ROOT)/lib \
	-Xlinker -rpath -Xlinker $(CUDNN_ROOT)/lib -l:libcudnn.so.9 \
	-lnvrtc -lcuda
CUDA_LDLIBS += $(CUDNN_LDLIBS)
endif
LDLIBS := $(CUDA_LDLIBS) -lstdc++ -licui18n -licuuc -lm
DEVICE_LDLIBS := -L$(CUDA_HOME)/lib64 -lcudart
DEVICE_SRC := h3_device_cuda.cu
GPU_SRC := h3_gpu_cuda.cu
TOKENIZER_SRC := h3_tokenizer.c
TOKENIZER_OBJ := h3_tokenizer.o
BACKEND := cuda
PLATFORM_LD := $(NVCC)
else
$(error unsupported PLATFORM '$(PLATFORM)'; expected Darwin or Linux)
endif

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c
LIB_PLATFORM := $(DEVICE_SRC) $(GPU_SRC) $(TOKENIZER_SRC)
DEVICE_OBJ := $(DEVICE_SRC:.m=.o)
DEVICE_OBJ := $(DEVICE_OBJ:.cu=.o)
GPU_OBJ := $(GPU_SRC:.m=.o)
GPU_OBJ := $(GPU_OBJ:.cu=.o)
LIB_OBJ := $(LIB_C:.c=.o) $(DEVICE_OBJ) $(GPU_OBJ) $(TOKENIZER_OBJ)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test host-portable-test tokenizer-portable-test checkpoint-schema-test cuda-runtime-test cuda-primitives-test cuda-rope-tokens-test cuda-linear-test cuda-attention-test cuda-ops-test parity real-parity print-build-config clean

print-build-config:
	@echo "platform=$(PLATFORM) backend=$(BACKEND) cc=$(CC) sources=$(LIB_PLATFORM)"

h3_host_portable_test: tests/test_host_portable.o h3_host.o
	$(CC) -o $@ $^ -lm

host-portable-test: h3_host_portable_test
	./h3_host_portable_test

h3_tokenizer_portable_test: tests/test_tokenizer_portable.o h3_tokenizer.o
	$(CC) -o $@ $^ -licui18n -licuuc

tokenizer-portable-test: h3_tokenizer_portable_test
	./h3_tokenizer_portable_test

h3_checkpoint_schema_test: tests/test_checkpoint_schema.o h3_safetensors.o h3_weights.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

checkpoint-schema-test: h3_checkpoint_schema_test
	./h3_checkpoint_schema_test MiniMax-H3

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_device_test: tests/test_device.o $(DEVICE_OBJ)
	$(PLATFORM_LD) -o $@ $^ $(DEVICE_LDLIBS)

h3_cuda_runtime_test: tests/test_cuda_runtime.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-runtime-test: h3_cuda_runtime_test
	./h3_cuda_runtime_test

h3_cuda_primitives_test: tests/test_cuda_primitives.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-primitives-test: h3_cuda_primitives_test
	./h3_cuda_primitives_test

h3_cuda_rope_tokens_test: tests/test_cuda_rope_tokens.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-rope-tokens-test: h3_cuda_rope_tokens_test
	./h3_cuda_rope_tokens_test

h3_cuda_linear_test: tests/test_cuda_linear.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-linear-test: h3_cuda_linear_test
	./h3_cuda_linear_test

h3_cuda_attention_test: tests/test_cuda_attention.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-attention-test: h3_cuda_attention_test
	./h3_cuda_attention_test

h3_cuda_ops_test: tests/test_cuda_ops.o h3_gpu_cuda.o
	$(NVCC) -o $@ $^ $(CUDA_LDLIBS)

cuda-ops-test: h3_cuda_ops_test
	./h3_cuda_ops_test

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(TOKENIZER_OBJ)
	$(CC) -o $@ $^ $(if $(filter Darwin,$(PLATFORM)),-licucore,-licui18n -licuuc)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_dit_bench_quality: tests/bench_dit_quality.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_quality.o: tests/bench_dit.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=36 \
		-DH3_BENCH_LATENT_W=64 -DH3_BENCH_LATENT_T=32 \
		-DH3_BENCH_AUDIO_T=178 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_vae_bench_quality: tests/bench_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_vae_bench_smoke: tests/bench_video_vae_smoke.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_video_vae_smoke.o: tests/bench_video_vae.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -DH3_BENCH_VAE_LATENT_T=7 \
		-DH3_BENCH_VAE_LATENT_H=4 -DH3_BENCH_VAE_LATENT_W=4 -c $< -o $@

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test

	./h3_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(CPPFLAGS) $(OBJCFLAGS) -I. -c $< -o $@

h3_tokenizer_metal.o: h3_tokenizer.m
	$(CC) $(CPPFLAGS) $(OBJCFLAGS) -I. -c $< -o $@

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_device_test h3_host_portable_test h3_tokenizer_portable_test h3_checkpoint_schema_test h3_cuda_runtime_test h3_cuda_primitives_test h3_cuda_rope_tokens_test h3_cuda_linear_test h3_cuda_attention_test h3_cuda_ops_test h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 h3_dit_bench_quality h3_vae_bench_quality \
		h3_vae_bench_smoke \
	libh3.a *.o *.d tests/*.o tests/*.d
