# Inventario API GPU per il port CUDA

Fonte di verità: `h3_gpu.h` al commit `8974cc0`. Le shape sono espresse in
ordine di memoria row-major; `N` indica il numero di elementi. `custom+BLAS`
indica un wrapper che usa cuBLASLt per GEMM e kernel CUDA per layout/fusione.

## Runtime, tensori e profiling

| API | Backend CUDA | Dtype e shape | Verifica CUDA prevista |
|---|---|---|---|
| `h3_gpu_create` | host + CUDA runtime/NVRTC | contesto; sorgente shader ignorata su CUDA | create su device 0, errore leggibile senza device |
| `h3_gpu_free` | host + CUDA runtime | contesto | teardown senza leak (Compute Sanitizer) |
| `h3_gpu_is_m5` | host | N/A; sempre 0 su CUDA | capability test |
| `h3_gpu_has_nax_mlp` | host | N/A; sempre 0 su CUDA | capability test |
| `h3_gpu_has_int8_mlp` | host | N/A; 1 se cuBLASLt INT8 disponibile | capability test |
| `h3_gpu_tensor_new_f32` | CUDA runtime | F32 `[N]` | allocazione, dtype, contatori |
| `h3_gpu_tensor_new_bf16` | CUDA runtime | BF16 `[N]` | allocazione, dtype, contatori |
| `h3_gpu_tensor_new_i8` | CUDA runtime | I8 `[N]` | allocazione, dtype, contatori |
| `h3_gpu_tensor_from_f32` | host + CUDA runtime | F32 host `[N]` -> device `[N]` | round-trip esatto |
| `h3_gpu_tensor_from_bf16` | host + CUDA runtime | BF16 host `[N]` -> device `[N]` | round-trip bit-esatto |
| `h3_gpu_tensor_from_u32` | host + CUDA runtime | U32 host `[N]` -> device `[N]` | round-trip bit-esatto |
| `h3_gpu_tensor_load_bf16` | host + CUDA runtime | file BF16 `[N]` -> device `[N]` | fixture safetensors, offset e short read |
| `h3_gpu_tensor_load_f32` | host + CUDA runtime | file F32 `[N]` -> device `[N]` | fixture safetensors, offset e short read |
| `h3_gpu_tensor_read_file_bf16` | host + CUDA runtime | file BF16 `[N]` -> tensor BF16 `[N]` | reload in-place e bounds |
| `h3_gpu_tensor_stream_file_bf16` | host + CUDA runtime | file BF16 `[N]` -> tensor BF16 `[N]` | stesso golden del read; hint cache best-effort |
| `h3_gpu_tensor_free` | CUDA runtime | tensor | double ownership escluso; contatori e leak |
| `h3_gpu_tensor_elements` | host | metadata `N` | tutti i costruttori |
| `h3_gpu_tensor_dtype` | host | metadata F32/BF16/I8/U32 | tutti i costruttori |
| `h3_gpu_tensor_read_f32` | CUDA runtime + custom | tensor F32/BF16 `[N]` -> host F32 `[N]` | F32 esatto; BF16 conversione esatta |
| `h3_gpu_tensor_read_f32_range` | CUDA runtime + custom | tensor F32/BF16 `[N]`, slice `[offset, n]` -> host F32 | offset/bounds e conversione |
| `h3_gpu_tensor_read_bf16` | CUDA runtime | BF16 `[N]` -> host BF16 `[N]` | bit-esatto |
| `h3_gpu_tensor_write_f32` | CUDA runtime + custom | host F32 `[N]` -> tensor F32/BF16 `[N]` | F32 esatto; BF16 round-to-nearest-even |
| `h3_gpu_tensor_write_f32_range` | CUDA runtime + custom | host F32 `[n]` -> slice tensor `[offset,n]` | offset/bounds, sentinelle adiacenti |
| `h3_gpu_tensor_write_bf16` | CUDA runtime | host BF16 `[N]` -> tensor BF16 `[N]` | bit-esatto |
| `h3_gpu_tensor_write_bf16_range` | CUDA runtime | host BF16 `[n]` -> slice `[offset,n]` | bit-esatto e bounds |
| `h3_gpu_begin` | CUDA runtime | stream/graph corrente | state-machine test |
| `h3_gpu_continue` | CUDA runtime | event + stream ordinato | due tranche, risultato e ordine |
| `h3_gpu_submit` | CUDA runtime | sincronizzazione stream | error propagation e contatori |
| `h3_gpu_error` | host | stringa | errore sintetico non vuoto |
| `h3_gpu_get_stats` | host + CUDA events | `h3_gpu_stats` | allocazioni, dispatch, submission, tempi non negativi |
| `h3_gpu_profile_set_label` | host | stringa | smoke con `H3_PROFILE` |
| `h3_gpu_profile_mark` | host + CUDA events | fase | smoke con `H3_PROFILE`, stream ordinato |

## Algebra lineare, quantizzazione e fusioni

| API | Backend CUDA | Dtype e shape | Verifica CUDA prevista |
|---|---|---|---|
| `h3_gpu_linear_f32` | cuBLASLt | X F32 `[R,K]`, W F32 `[O,K]`, bias `[O]?` -> Y `[R,O]` | fixture `test_metal`; CPU piccoli |
| `h3_gpu_patch_linear_bf16` | cuBLASLt | X/W/B/Y BF16, `[R,K]·[O,K] -> [R,O]` | BF16 fixture + CPU piccoli |
| `h3_gpu_patch_linear_bf16_offset` | cuBLASLt | come sopra con offset elemento input/output | sentinelle e parità patch-linear |
| `h3_gpu_patch_linear_bf16_map` | custom+BLAS | X `[R,K]`, map U32 `[R]`, Y `[output_rows,O]` | map sparsa, duplicati, bounds |
| `h3_gpu_linear_bf16` | cuBLASLt | X/W/B/Y BF16, `[R,K]·[O,K] -> [R,O]`, accumulo F32 | `test_bf16`, `test_text_metal`, fixture reali |
| `h3_gpu_mlp_bf16` | custom+BLAS | X `[R,K]`, W1 `[2H,K]`, W2 `[O,H]` -> Y `[R,O]` | `test_bf16`, confronto pipeline non fusa |
| `h3_gpu_mlp_nax_bf16` | host fallback | stesso contratto MLP; alias della pipeline portabile su CUDA | risultato identico a `mlp_bf16`, capability false |
| `h3_gpu_quantize_weight_int8` | CUDA custom | BF16/F32 `[R,C]` -> I8 `[R,C]` + scale F32 `[R]` | dequant CPU, errore per-canale |
| `h3_gpu_linear_int8_bf16` | custom+cuBLASLt | X BF16 `[R,K]`, W I8 `[O,K]`, scale X `[R]`, W `[O]` -> BF16 `[R,O]` | CPU INT8 e confronto BF16 |
| `h3_gpu_linear_int8_head_major_bf16` | custom+cuBLASLt | X BF16 `[heads,R,D]`, W I8 `[O,heads*D]` -> BF16 `[R,O]` | confronto transpose+linear |
| `h3_gpu_mlp_int8_bf16` | custom+cuBLASLt | X `[R,K]`, W1 I8 `[2H,K]`, W2 I8 `[O,H]` -> BF16 `[R,O]` | confronto MLP BF16 e flag-paths |
| `h3_gpu_adaln_linear_bf16` | custom+cuBLASLt | AdaLN `[R,W]` + Wgt `[O,W]` -> BF16 `[R,O]`, inverse `[R]` | `test_bf16`, confronto due chiamate |
| `h3_gpu_grouped_qkv_linear_rope_bf16` | custom+cuBLASLt | X `[R,K]`, W `[3*H*D,K]` -> Q/K/V `[H,R,D]` | `test_bf16`, confronto linear+grouped QKV |
| `h3_gpu_grouped_qkv_linear_rope_int8` | custom+cuBLASLt | X BF16/I8 `[R,K]`, W I8 `[3HD,K]` -> Q/K/V BF16 `[H,R,D]` | confronto variante BF16 e flag-paths |

## Elementwise, norm, embedding, RoPE e token transforms

| API | Backend CUDA | Dtype e shape | Verifica CUDA prevista |
|---|---|---|---|
| `h3_gpu_silu_f32` | CUDA custom | F32 `[N] -> [N]` | CPU edge values |
| `h3_gpu_cast_f32_to_bf16` | CUDA custom | F32 `[N] -> BF16 [N]` | bit oracle inclusi NaN/Inf |
| `h3_gpu_cast_bf16_to_f32` | CUDA custom | BF16 `[N] -> F32 [N]` | bit oracle |
| `h3_gpu_copy_bf16` | CUDA runtime | BF16 slice `[src,n] -> [dst,n]` | sentinelle, overlap policy |
| `h3_gpu_copy_f32` | CUDA runtime | F32 slice `[src,n] -> [dst,n]` | sentinelle, overlap policy |
| `h3_gpu_rms_norm_f32` | CUDA custom | X F32 `[R,W]`, weight `[W]` -> F32 `[R,W]` | `test_metal`/CPU |
| `h3_gpu_adaln_f32` | CUDA custom | X `[R,W]`, norm `[W]`, mod `[M,S,W]`, map U32 `[R]` -> `[R,W]` | `test_metal`/CPU |
| `h3_gpu_gate_f32` | CUDA custom | residual/branch `[R,W]`, mod `[M,S,W]`, map `[R]` -> `[R,W]` | `test_metal`/CPU |
| `h3_gpu_swiglu_f32` | CUDA custom | fused F32 `[R,2W] -> [R,W]` | `test_metal`/CPU |
| `h3_gpu_scale_add_f32` | CUDA custom | residual/branch `[R,W]`, scale `[R]` o `[1]` -> `[R,W]` | CPU broadcast cases |
| `h3_gpu_layer_norm_f32` | CUDA custom | X `[R,W]`, weight/bias `[W]` -> `[R,W]` | CPU, constant row |
| `h3_gpu_weight_norm_f32` | CUDA custom | vector `[outer,inner]`, magnitude `[outer]` -> `[outer,inner]` | `test_audio_gpu`, abs `2e-6` |
| `h3_gpu_add_scaled_f32` | CUDA custom | left/right F32 `[N] -> [N]` | `test_audio_gpu`, abs `1e-7` |
| `h3_gpu_alias_free_snake_f32` | CUDA custom | X F32 `[B,L,C]`, params `[C]`, filters `[12]` -> `[B,L,C]` | `test_audio_gpu`, abs `2e-5` |
| `h3_gpu_snake1d_f32` | CUDA custom | X F32 `[B,L,C]`, alpha `[C]` -> `[B,L,C]` | CPU edge values |
| `h3_gpu_geglu_f32` | CUDA custom | gate/linear F32 `[N] -> [N]` | CPU GELU oracle |
| `h3_gpu_clip_f32` | CUDA custom | F32 `[N] -> [N]` | `test_audio_gpu`, abs `1e-7` |
| `h3_gpu_silu_bf16` | CUDA custom | BF16 `[N] -> [N]`, accumulo F32 | `test_bf16`, BF16 boundary |
| `h3_gpu_rms_norm_bf16` | CUDA custom | X BF16 `[R,W]`, weight `[W]` -> BF16 `[R,W]` | `test_bf16`, fixture reali |
| `h3_gpu_layer_norm_bf16` | CUDA custom | X BF16 `[R,W]`, weight/bias `[W]` -> BF16 `[R,W]` | vision fixture + CPU |
| `h3_gpu_gelu_bf16` | CUDA custom | BF16 `[N] -> [N]`, exact/approx flag | entrambe le modalità contro CPU |
| `h3_gpu_vision_qkv_rope_bf16` | CUDA custom | QKV `[S,3,H,D]` -> Q/K/V `[H,S,D]` BF16 | vision fixture |
| `h3_gpu_adaln_bf16` | CUDA custom | BF16 equivalente AdaLN F32 `[R,W]` | `test_bf16` |
| `h3_gpu_adaln_bf16_offset` | CUDA custom | come AdaLN, input slice da offset | `test_bf16`, sentinelle |
| `h3_gpu_gate_bf16` | CUDA custom | BF16 equivalente gate `[R,W]` | `test_bf16` |
| `h3_gpu_gate_adaln_bf16` | CUDA custom | gate residual + AdaLN, due output `[R,W]` | `test_bf16`, confronto non fuso |
| `h3_gpu_gate_adaln_quantize_int8` | CUDA custom | gate/AdaLN BF16 `[R,W]` -> residual BF16 + I8 `[padded_R,W]`, scale `[padded_R]` | confronto non fuso + dequant |
| `h3_gpu_qkv_rope_bf16` | CUDA custom | QKV BF16 `[S,3,H,D]` -> Q/K/V `[H,S,D]` | `test_bf16` |
| `h3_gpu_grouped_qkv_rope_bf16` | CUDA custom | QKV BF16 `[S,H,3,D]` -> Q/K/V `[H,S,D]` | `test_bf16`, bit-identico alla permutazione |
| `h3_gpu_swiglu_bf16` | CUDA custom | fused BF16 `[R,2W] -> [R,W]` | `test_bf16`, fixture MLX |
| `h3_gpu_embedding_bf16` | CUDA custom | weight BF16 `[V,W]`, ids U32 `[T]` -> BF16 `[T,W]` | `test_text_metal`, ids limite/invalidi |
| `h3_gpu_text_qk_rope_bf16` | CUDA custom | Q `[S,QH,D]`, K `[S,KH,D]` -> head-major BF16 | confronto head-norm + rope separati |
| `h3_gpu_head_rms_norm_bf16` | CUDA custom | tensor BF16 `[S,H,D]` in-place, weight `[D]` | `test_text_metal` |
| `h3_gpu_rope_text_bf16` | CUDA custom | Q `[S,QH,D]`, K `[S,KH,D]`, cos/sin F32 `[S,D/2]` | `test_text_metal` |
| `h3_gpu_add_bf16` | CUDA custom | BF16 `[N] + [N] -> [N]` | `test_bf16`, `test_text_metal` |
| `h3_gpu_sub_bf16` | CUDA custom | BF16 `[N] - [N] -> [N]` | CPU e round boundary |
| `h3_gpu_token_pool_bf16` | CUDA custom | BF16 input `[input_R,W]` -> pooled `[R,W]`; indices U32 | `test_bf16` synthetic exact |
| `h3_gpu_token_pool_adaln_bf16` | CUDA custom | pool + AdaLN, residual/output `[R,W]` | `test_bf16`, confronto non fuso |
| `h3_gpu_token_expand_delta_bf16` | CUDA custom | reduced `[reduced_R,W]` + maps U32 -> output `[R,W]` | `test_bf16` synthetic exact |
| `h3_gpu_token_expand_adaln_bf16` | CUDA custom | expand + AdaLN, residual/output `[R,W]` | `test_bf16`, confronto non fuso |
| `h3_gpu_euler_bf16` | CUDA custom | sample F32 slice `[N]`, last/previous BF16 `[N]` -> F32 | `test_bf16`, CPU formula |
| `h3_gpu_silu_mul_bf16` | CUDA custom | gate/up BF16 `[N] -> [N]` | `test_text_metal`, CPU |

## Attention

| API | Backend CUDA | Dtype e shape | Verifica CUDA prevista |
|---|---|---|---|
| `h3_gpu_qkv_rope_f32` | CUDA custom | QKV F32 `[S,3,H,D]` -> Q/K/V `[H,S,D]`, norm/RoPE | `test_metal` fixture |
| `h3_gpu_sdpa_f32` | CUDA custom + cuBLASLt | Q/K/V F32 `[H,S,D]` -> `[S,H,D]`, non causale | `test_metal`, CPU piccoli |
| `h3_gpu_video_qkv_rope_f32` | CUDA custom | QKV F32 `[S,3,H,D]` -> `[H,S,D]`, video RoPE | video encoder fixture |
| `h3_gpu_audio_qkv_split_f32` | CUDA custom | QKV F32 `[B,L,3,H,D]` + bias -> Q/K/V `[B,H,L,D]` | audio encoder fixture + CPU |
| `h3_gpu_sdpa_causal_f32` | CUDA custom + cuBLASLt | Q/K/V F32 `[B,H,S,D]` -> `[B,S,H,D]`, causal | CPU mask test |
| `h3_gpu_audio_attention_pool_f32` | CUDA custom | attended `[B,L,H,D]` -> `[B,output_dim]` | audio encoder fixture |
| `h3_gpu_sdpa_bf16` | CUDA custom + cuBLASLt | Q/K/V BF16 `[H,S,D]` -> `[S,H,D]` | `test_bf16`, real DiT fixture |
| `h3_gpu_sdpa_bf16_head_major_output` | CUDA custom + cuBLASLt | Q/K/V BF16 `[H,S,D]` -> `[H,S,D]` | confronto `sdpa_bf16` + transpose |
| `h3_gpu_gqa_causal_bf16` | CUDA custom + cuBLASLt | Q `[QH,S,D]`, K/V `[KH,S,D]` -> `[S,QH,D]`, causal | `test_text_metal`, CPU mask/GQA |

## Convoluzioni e VAE

| API | Backend CUDA | Dtype e shape | Verifica CUDA prevista |
|---|---|---|---|
| `h3_gpu_conv1d_f32` | CUDA custom + cuBLASLt | X `[B,L,Cin]`, W `[Cout,Cin,K]` -> `[B,Lout,Cout]` | `test_audio_gpu`, abs `2e-5` |
| `h3_gpu_conv1d_stride_f32` | CUDA custom + cuBLASLt | come Conv1d con stride, `Lout=floor((L+2P-D(K-1)-1)/S)+1` | CPU stride/dilation |
| `h3_gpu_conv_transpose1d_f32` | CUDA custom + cuBLASLt | X `[B,L,Cin]`, W `[Cin,Cout,K]` -> `[(L-1)S+K-2P]` | `test_audio_gpu`, abs `2e-5` |
| `h3_gpu_vae_encoder_pad_f32` | CUDA custom | X `[B,T,H,W,C]` -> padded channels-last tensor | semantic VAE encoder, border oracle |
| `h3_gpu_conv3d_f32` | CUDA custom + cuBLASLt | X `[B,T,H,W,Cin]`, W `[Cout,Cin,Kt,Kh,Kw]` -> channels-last output | semantic/real video encoder fixtures |
| `h3_gpu_vae_encoder_group_norm_silu_f32` | CUDA custom | X F32 `[B,T,H,W,C]`, weight/bias `[C]` -> stessa shape | semantic VAE encoder + CPU |

## Soglie di parità

- Copie, indici, metadata, layout puri e round-trip BF16/U32/I8: confronto
  bit-esatto. Le fusioni devono coincidere bit-per-bit con la pipeline CUDA non
  fusa quando condividono gli stessi confini di arrotondamento BF16.
- Primitive F32 con oracle CPU: `max_abs <= 2e-5`, salvo weight norm
  `2e-6` e add/clip `1e-7`, mantenendo le soglie già usate da
  `tests/test_audio_gpu.c`.
- Blocchi F32 contro fixture MLX: `max_rel < 5e-3`, come
  `tests/test_metal.c`.
- Primitive e blocchi BF16 contro fixture MLX: `max_rel < 1e-2`, come
  `tests/test_bf16.c` e `tests/test_text_metal.c`. Per una singola primitiva si
  registra anche `max_abs`; NaN/Inf o mismatch di shape sono sempre FAIL.
- INT8: oltre alla soglia finale BF16 `max_rel < 2e-2`, la quantizzazione deve
  rispettare `max_abs(dequant-input) <= scale/2 + 1e-6` per riga/canale.

## Equivalenza della suite Metal

Ogni eseguibile GPU esistente viene compilato una seconda volta contro il
backend CUDA, senza cambiare fixture né assertions:

| Suite esistente | Equivalente CUDA | Copertura principale |
|---|---|---|
| `tests/test_metal.c` | `h3_cuda_tests` | blocco DiT F32, statistiche |
| `tests/test_bf16.c` | `h3_cuda_bf16_tests` | DiT BF16, fusioni, token reduction, Euler, INT8 fallback |
| `tests/test_text_metal.c` | `h3_cuda_text_tests` | embedding, Qwen norm/RoPE/GQA/MLP |
| `tests/test_audio_gpu.c` | `h3_cuda_audio_gpu_tests` | Conv1d, transpose, weight norm, Snake, elementwise |
| `tests/test_real_dit_block.c` | stesso target con backend CUDA | primitive reali e confini BF16 |
| `tests/test_real_dit.c`, `test_real_dit_schedule.c`, `test_semantic_dit.c` | stessi target con backend CUDA | integrazione DiT e scheduling |
| test real/semantic audio e video già elencati nel `Makefile` | stessi target con backend CUDA | operatori encoder/VAE e fixture checkpoint |

Le API di lifecycle/I/O non coperte direttamente dalle suite Metal ricevono un
test CUDA dedicato. Il test genera un file temporaneo controllato per coprire
load, reload, streaming, range, errori e contatori.

## Rischi specifici GB10

1. La memoria unificata CPU/GPU non rende automaticamente conveniente
   `cudaMallocManaged`: page migration e fault su stream di pesi possono
   serializzare I/O e compute. Il default resta device memory con staging
   pinned; managed+prefetch richiede benchmark del Task 11.
2. cuBLASLt su ARM64/Blackwell può scegliere workspace e algoritmi diversi tra
   shape; ogni matmul deve avere fallback deterministico e workspace limitato.
3. BF16 Tensor Core cambia l’ordine delle riduzioni rispetto a MPSGraph. Le
   soglie sopra verificano il risultato, mentre i test bit-esatti sono limitati
   a layout, copie e fusioni con uguali confini BF16.
4. Gli operatori conv/attention senza cuDNN devono evitare materializzazioni
   `im2col` o score `[S,S]` non limitate: sul modello reale possono consumare
   decine di GiB nonostante i 121 GiB disponibili.
5. Lo streaming SSD deve mantenere vivi staging buffer e CUDA event fino alla
   copia completata; riuso anticipato produce corruzioni intermittenti che i
   soli test piccoli non rilevano.
6. Le opzioni Metal 4/NAX non hanno equivalente diretto. Su CUDA usano la
   pipeline portabile; INT8 è dichiarato disponibile solo dopo un probe reale
   cuBLASLt, mai in base al solo compute capability.
