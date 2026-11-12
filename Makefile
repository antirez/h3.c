CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm

H3_IMAGE_DIR ?= MiniMax-H3-Image

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_names.c h3_dit_schedule.c h3_dit.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c h3_image.c \
	h3_vae_tiles.c h3_image_vae.c h3_image_vae_encode.c h3_image_pipeline.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test parity real-parity image-check image-edit-fixtures clean

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_edit_presentation_tests: tests/test_edit_presentation.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_image_gpu_tests: tests/test_image_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_gpu_determinism_tests: tests/test_gpu_determinism.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_png_test: tests/test_png.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_image_png_test: tests/test_image_png.o $(LIB_OBJ)
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

h3_real_image_vae_test: tests/test_real_image_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_vae_encode_test: tests/test_real_image_vae_encode.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_dit_block_test: tests/test_real_image_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_dit_test: tests/test_real_image_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_dit_large_test: tests/test_real_image_dit_large.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_text_test: tests/test_real_image_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_e2e_test: tests/test_real_image_e2e.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_edit_text_test: tests/test_real_image_edit_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_image_edit_dit_test: tests/test_real_image_edit_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
	h3_edit_presentation_tests h3_text_tests \
	h3_audio_gpu_tests h3_image_gpu_tests h3_gpu_determinism_tests \
	h3_real_audio_vae_test \
	h3_real_audio_encoder_test h3_av_mux_test h3_png_test \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test \
	h3_real_image_vae_test h3_real_image_vae_encode_test \
	h3_real_image_dit_block_test h3_real_image_dit_test h3_real_image_dit_large_test \
	h3_real_image_text_test h3_real_image_e2e_test \
	h3_real_image_edit_text_test h3_real_image_edit_dit_test

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
	./h3_edit_presentation_tests $(H3_IMAGE_DIR)/tokenizer/tokenizer.json
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	./h3_image_gpu_tests
	./h3_gpu_determinism_tests
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
		./h3_png_test; \
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
	@if test -f $(H3_IMAGE_DIR)/text_encoder/model-00014-of-00014.safetensors && \
	         test -f $(H3_IMAGE_DIR)/tokenizer/tokenizer.json && \
	         test -f misc/fixtures/h3_image_text_256.safetensors; then \
		./h3_real_image_text_test $(H3_IMAGE_DIR) \
			misc/fixtures/h3_image_text_256.safetensors; \
	else \
		echo "skip: image text encoder weights/fixture are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors && \
	         test -f $(H3_IMAGE_DIR)/text_encoder/model-00014-of-00014.safetensors && \
	         test -f $(H3_IMAGE_DIR)/vae_diffusers/diffusion_pytorch_model.safetensors && \
	         test -f misc/fixtures/h3_image_e2e_512.safetensors && \
	         test -f misc/fixtures/h3_image_denoise.safetensors; then \
		rm -f h3_image_e2e_latent.bin; \
		for mode in stream bf16 int8; do \
			./h3_real_image_e2e_test $(H3_IMAGE_DIR) \
				misc/fixtures/h3_image_e2e_512.safetensors $$mode \
				misc/fixtures/h3_image_denoise.safetensors \
				h3_image_e2e_latent.bin || exit 1; \
		done; \
		rm -f h3_image_e2e_latent.bin; \
	else \
		echo "skip: image end-to-end weights/fixture are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/vae_diffusers/diffusion_pytorch_model.safetensors && \
	         test -f misc/fixtures/h3_image_vae_decode.safetensors; then \
		./h3_real_image_vae_test $(H3_IMAGE_DIR)/vae_diffusers \
			misc/fixtures/h3_image_vae_decode.safetensors; \
	else \
		echo "skip: image VAE weights/fixture are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/vae_diffusers/diffusion_pytorch_model.safetensors && \
	         test -f misc/fixtures/h3_image_edit_vae.safetensors; then \
		./h3_real_image_vae_encode_test $(H3_IMAGE_DIR)/vae_diffusers \
			misc/fixtures/h3_image_edit_vae.safetensors; \
	else \
		echo "skip: image VAE encoder weights/fixture are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors && \
	         test -f misc/fixtures/h3_image_dit_block0.safetensors; then \
		./h3_real_image_dit_block_test $(H3_IMAGE_DIR)/transformer_diffusers \
			misc/fixtures/h3_image_dit_block0.safetensors; \
	else \
		echo "skip: image DiT weights/block-0 fixture are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors && \
	         test -f misc/fixtures/h3_image_dit_step.safetensors && \
	         test -f misc/fixtures/h3_image_text_256.safetensors; then \
		rm -f h3_image_dit_velocity.bin; \
		for mode in stream bf16 int8; do \
			./h3_real_image_dit_test $(H3_IMAGE_DIR)/transformer_diffusers \
				misc/fixtures/h3_image_dit_step.safetensors \
				misc/fixtures/h3_image_text_256.safetensors $$mode \
				h3_image_dit_velocity.bin \
				misc/fixtures/h3_image_dit_step_f32.safetensors || exit 1; \
		done; \
		rm -f h3_image_dit_velocity.bin; \
	else \
		echo "skip: image DiT weights/step fixtures are not installed"; \
	fi

	@for latent in 128 192; do \
		fixture=misc/fixtures/h3_image_dit_step_large_$${latent}x$${latent}.safetensors; \
		if test -f $(H3_IMAGE_DIR)/transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors && \
		   test -f $$fixture && \
		   test -f misc/fixtures/h3_image_text_256.safetensors; then \
			rm -f h3_image_dit_large_velocity.bin; \
			for mode in bf16 stream int8; do \
				./h3_real_image_dit_large_test \
					$(H3_IMAGE_DIR)/transformer_diffusers \
					misc/fixtures/h3_image_text_256.safetensors $$fixture \
					$$mode h3_image_dit_large_velocity.bin || exit 1; \
			done; \
			rm -f h3_image_dit_large_velocity.bin; \
		else \
			echo "skip: large-canvas DiT fixture $$fixture is not installed"; \
		fi; \
	done
	@if test -f $(H3_IMAGE_DIR)/text_encoder/model-00014-of-00014.safetensors && \
	         test -f $(H3_IMAGE_DIR)/tokenizer/tokenizer.json && \
	         test -f misc/fixtures/h3_image_edit_text.safetensors && \
	         test -f misc/fixtures/h3_image_edit_text_small.safetensors; then \
		./h3_real_image_edit_text_test $(H3_IMAGE_DIR) \
			misc/fixtures/h3_image_edit_text.safetensors \
			misc/fixtures/h3_image_edit_text_small.safetensors; \
	else \
		echo "skip: image edit text weights/fixtures are not installed"; \
	fi
	@if test -f $(H3_IMAGE_DIR)/transformer_diffusers/diffusion_pytorch_model-00001-of-00012.safetensors && \
	         test -f misc/fixtures/h3_image_edit_step.safetensors && \
	         test -f misc/fixtures/h3_image_edit_text_small.safetensors; then \
		rm -f h3_image_edit_dit_velocity.bin; \
		for mode in stream bf16 int8; do \
			./h3_real_image_edit_dit_test $(H3_IMAGE_DIR)/transformer_diffusers \
				misc/fixtures/h3_image_edit_step.safetensors \
				misc/fixtures/h3_image_edit_text_small.safetensors $$mode \
				h3_image_edit_dit_velocity.bin \
				misc/fixtures/h3_image_edit_step_f32.safetensors || exit 1; \
		done; \
		rm -f h3_image_edit_dit_velocity.bin; \
	else \
		echo "skip: image edit DiT weights/step fixtures are not installed"; \
	fi

# Two default-mode runs of the fixture prompt must give a valid 512x512 PNG
# and identical bytes. It runs the 62 GB text encoder and transformer, so on
# the development Mac run it through the guard wrapper (gated_exec.py), not
# bare: python3 gated_exec.py --swap-mb 14000 --timeout 1500 -- make image-check
image-check: h3 h3_image_png_test
	@dir=$$(mktemp -d) && \
	for run in a b; do \
		./h3 -d $(H3_IMAGE_DIR) -p "a red fox in snow" --size 512x512 \
			--min-generation-area 1024 --seed 42 \
			-o $$dir/$$run.png || { rm -rf $$dir; exit 1; }; \
	done && \
	./h3_image_png_test 512 512 $$dir/a.png $$dir/b.png; \
	status=$$?; rm -rf $$dir; exit $$status

# make image-edit-fixtures prints, and runs nothing, the commands of the gated
# window for reference-image editing (M5b.2, M5b.3, M5c.2) in the order they
# depend on each other. Times and peak memory are estimates from the t2i runs,
# not measurements; fill them in from the first run. PY is any Python with
# diffusers (MiniMax-H3-Image classes), transformers, torch and safetensors;
# G is the guard wrapper. The harness runs on the CPU from memory-mapped
# weights, so its peak is evictable file pages plus a few GB. Every command
# can first be run with --plan (harness) to print shapes without a 62 GB model.
define IMAGE_EDIT_FIXTURES
GATED=/private/tmp/claude-501/-Users-ronica-projects/6bc281dc-2caa-4d62-b4ea-14cf2c92d0cd/scratchpad/gated_exec.py
G="python3 $$GATED --swap-mb 14000 --timeout 7200 --"   PY="python -I"
H="$$PY tools/h3_image_reference.py --device cpu --out-dir misc/fixtures"
 1. $$H --stage edit_vae                        only for the M5c reference-encoder test, not for steps 2-8; light, about 1 min
 2. $$G $$H --stage edit_text_small             62 GB text encoder, N=84: 5-15 min, 4 GB anon + page cache
 3. $$G $$H --stage edit_step                   BF16 DiT x 6 steps, S=404: about 30 min, 6 GB anon + page cache
 4. $$G $$H --stage edit_f32truth               F32, each block converted once for all six steps: about 5 min, 4 GB anon + page cache
 5. $$G $$H --stage edit_text                   62 GB text encoder, N=1044: 10-30 min, 6 GB anon + page cache
 6. $$G $$H --stage edit_e2e                    F32 steps-outer, 300 block conversions, S=2324: 25-35 min, 8 GB anon + page cache
    ($$H --stage edit_e2e --bf16-reference adds the BF16 image, hours; optional)
 7. make h3_real_image_edit_text_test && H3_MEASURE=1 $$G ./h3_real_image_edit_text_test $(H3_IMAGE_DIR) \
      misc/fixtures/h3_image_edit_text.safetensors misc/fixtures/h3_image_edit_text_small.safetensors
      2 vision-tower and 2 text passes: 10-20 min, about 8 GB (ring of layers) + page cache
 8. make h3_real_image_edit_dit_test && rm -f h3_image_edit_dit_velocity.bin; for mode in stream bf16 int8; do \
      H3_MEASURE=1 $$G ./h3_real_image_edit_dit_test \
      $(H3_IMAGE_DIR)/transformer_diffusers misc/fixtures/h3_image_edit_step.safetensors \
      misc/fixtures/h3_image_edit_text_small.safetensors $$mode h3_image_edit_dit_velocity.bin \
      misc/fixtures/h3_image_edit_step_f32.safetensors; done
      two loads per mode: stream 5-10 min and 4 GiB, bf16 5-10 min and 38 GiB, int8 5-10 min and 20 GiB
    H3_MEASURE=1 makes every placeholder bound only print PASS or OVER and the run continue; exact checks still fail.
 9. M5c.1/M5c.2: make image-edit-check, then the edit end-to-end test on misc/fixtures/h3_image_edit_e2e_512.safetensors
    After steps 7-8 replace the placeholders marked "measured in the gated window" in
    tests/test_real_image_edit_text.c and tests/test_real_image_edit_dit.c by 1.4x the printed values,
    then run both without H3_MEASURE (make test does).
endef
export IMAGE_EDIT_FIXTURES
image-edit-fixtures:
	@printf '%s\n' "$$IMAGE_EDIT_FIXTURES"

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_edit_presentation_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_image_gpu_tests h3_gpu_determinism_tests \
		h3_real_audio_vae_test \
		h3_real_audio_encoder_test h3_av_mux_test h3_png_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test h3_real_image_vae_test \
		h3_real_image_vae_encode_test \
		h3_real_image_dit_block_test h3_real_image_dit_test \
		h3_real_image_dit_large_test h3_real_image_text_test h3_real_image_e2e_test h3_image_png_test \
		h3_real_image_edit_text_test h3_real_image_edit_dit_test \
	h3_dit_bench h3_dit_bench_864 \
	libh3.a *.o *.d tests/*.o tests/*.d
