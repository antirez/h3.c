# h3.c 运行时 LoRA 设计方案

状态：拟实施

日期：2026-08-23

目标平台：Apple Silicon，重点为 M5 Pro/Max

## 1. 决策摘要

采用“加载时内存折叠”，不在每次线性层前向中动态执行 LoRA 分支。

用户接口表现为运行时 LoRA：

```sh
./h3 -d ./MiniMax-H3/turbo-v4-step600-ema \
  --lora style.safetensors --lora-scale 0.8 \
  -p "A person walks" -o outputs/walk.mp4
```

加载 DiT 权重时按行分块计算：

```text
W_next = BF16(W_current + user_scale * alpha/rank * (B @ A))
```

随后继续使用现有 BF16、M5 INT8、QKV、attention-output 和 FC2 路径。因此：

- 不产生新的约 62 GiB transformer 副本；
- LoRA 路径和强度可在启动参数中选择；
- 生成阶段不增加 LoRA GEMM；
- 多 LoRA 的结果与当前离线顺序 folding 保持一致；
- 使用 LoRA 时不使用与其不匹配的预生成 INT8 sidecar。

第一版不实现“每层前向时动态 LoRA”。切换 LoRA 集合或 scale 会重建 prepared DiT，但常驻 worker 可在相同配置下复用结果。

## 2. 背景与当前约束

当前实现具备以下基础：

- `h3_weight_store` 负责发现 safetensors 与按名称加载权重；
- `h3_dit` 在加载阶段将主要 BF16 矩阵量化到 M5 INT8；
- prepared-DiT cache key 当前包含 prompt、shape 和质量参数；
- Web 服务通过一个常驻 `h3 --worker` 进程复用上下文；
- 离线 folding 工具已经支持三类 H3 LoRA：
  - 原生 `<target>.lora_A/B.weight`；
  - AI Toolkit / Diffusers `diffusion_model.*.lora_A/B.weight`；
  - Kohya / ComfyUI `lora_unet_*` down/up 与可选 alpha；
- adapter dtype 支持 F16、BF16、F32；
- 当前 Turbo 模型已有约 18 GiB 的 INT8 sidecar，单次冷启动由约 48.0 秒降至 37.75 秒，输出逐字节一致；
- 本机只剩约 66 GiB 可用空间，继续为每个 LoRA 创建完整模型副本不可持续。

## 3. 目标与非目标

### 3.1 目标

1. CLI 支持单个和多个 LoRA，以及独立 scale。
2. 支持现有三种键名格式和三种浮点 dtype。
3. 峰值 host scratch 保持在 256 MiB 以内，不构造完整 F32 权重矩阵。
4. 原始 checkpoint 与 LoRA 文件保持只读。
5. LoRA 折叠完成后沿用现有推理内核，稳定态 denoise 性能回归不超过 2%。
6. LoRA 文件内容哈希、scale、顺序和作用域进入 prepared-DiT cache key。
7. FL2VA 与 Ref2VA 明确隔离；不会将 FL2VA adapter 静默用于 Ref2VA。
8. 常驻 worker 支持切换 LoRA，并能复用完全相同的 LoRA 配置。
9. 运行时结果可与离线 folding 结果逐字节或在明确阈值内对齐。

### 3.2 非目标

- 第一版不支持在一次 denoise 过程中动态改变 scale。
- 第一版不为每个线性层增加 `B(Ax)` 推理分支。
- 第一版不尝试将任意 LoRA 叠加到已有 INT8 sidecar 上。
- 第一版不自动判断 LoRA 的语义用途或推荐强度。
- 不承诺未声明为 H3 的 LoRA 可以通过键名猜测运行。

## 4. 用户接口

### 4.1 CLI

`--lora` 可重复。`--lora-scale` 和 `--lora-scope` 修改最近一次出现的 `--lora`：

```sh
./h3 -d ./MiniMax-H3/turbo-v4-step600-ema \
  --lora HMNSFW_AIO_V2.safetensors --lora-scale 0.65 \
  --lora riding_pose_H3_i2v_v1.0.safetensors --lora-scale 0.8 \
  --lora-scope fl2va \
  -p "A person walks" --steps 5 --layers 50 --reuse 1 \
  -o outputs/custom.mp4
```

规则：

- `--lora-scale` 默认 `1.0`，必须为有限数；建议 UI 范围 `[-2, 2]`，C API 不强制该软限制。
- `--lora-scope` 为 `fl2va`、`ref2va` 或 `both`，默认 `fl2va`。
- 最多 8 个 adapter；按命令行顺序应用。
- `--lora-scale` 或 `--lora-scope` 出现在首个 `--lora` 前时直接报错。
- 当前生成模式不在 adapter scope 内时直接报错，不静默忽略。
- `--ssd-streaming` 第一版不支持 LoRA，组合使用时返回明确错误。

### 4.2 C API

在 `h3.h` 增加：

```c
typedef enum {
    H3_LORA_FL2VA = 1,
    H3_LORA_REF2VA = 2,
    H3_LORA_BOTH = 3
} h3_lora_scope;

typedef struct {
    const char *path;
    float scale;
    h3_lora_scope scope;
} h3_lora_spec;
```

在 `h3_params` 增加：

```c
const h3_lora_spec *loras;
size_t lora_count;
```

调用期间路径有效期与现有 reference path 相同。`h3_generate` 不取得调用者字符串所有权。
字段追加在结构体尾部，并同步更新 `H3_PARAMS_DEFAULT`、C++17 public-header smoke
和所有显式初始化 fixture。该变化需要提升公开 API 版本，不能假定旧二进制调用者与新
`libh3` ABI 兼容。

### 4.3 Worker 与 Web API

机器协议增加：

```text
!worker-lora fl2va 0.8 <hex-encoded-path>
!worker-lora both 0.5 <hex-encoded-path>
!worker-reset
```

Web 请求增加：

```json
{
  "loras": [
    {"token": "local-model-id", "scale": 0.8, "scope": "fl2va"}
  ]
}
```

Web 服务只允许来自配置白名单目录的 LoRA，不接受任意绝对路径。返回任务详情时保留文件名、scale、scope 和 SHA-256，不暴露完整本地路径。

## 5. LoRA 解析与规范化

新增 `h3_lora.c/.h`，不把格式判断散落到 CLI 或 DiT：

```text
h3_lora_open(path)
  -> 读取 safetensors header
  -> 识别 native / diffusion_model / kohya
  -> 将键名规范化为 checkpoint tensor name
  -> 配对 A/B 或 down/up
  -> 读取并计算 alpha/rank
  -> 校验 dtype、rank、shape 和 target coverage
  -> 计算文件 SHA-256
```

adapter 文件在解析开始时打开并持有文件描述符；哈希前后各执行一次 `fstat`，若
size、mtime 或 inode 发生变化则拒绝本次请求，避免校验后被替换的竞态。

规范化规则与 `tools/fold_turbo_lora.py` 保持相同：

- native：`blocks.0...lora_A.weight` → `blocks.0....weight`；
- Diffusers：先移除 `diffusion_model.`；
- Kohya：使用 checkpoint 的真实 tensor name 建立 `.` → `_` 反向表，不通过猜测拆分下划线；
- Kohya 有 alpha 时，实际系数为 `user_scale * alpha / rank`；
- 无 alpha 时，实际系数为 `user_scale`。

任何未配对、重复映射、未知 dtype、shape 不匹配或部分匹配都在开始加载模型前失败。

## 6. 加载时低内存折叠

### 6.1 数据流

扩展 `h3_weights`，增加带 adapter overlay 的矩阵加载接口：

```c
h3_weight_load_bf16_with_loras(
    base_store, gpu, tensor_name, shape,
    lora_set, scope, error, error_size);
```

没有 adapter 命中该 tensor 时，继续走现有零拷贝/直接加载路径。命中时：

1. mmap 或分块读取基础 BF16 行；
2. 将 LoRA A 保持为小型 F32 矩阵；
3. 按 16–64 行读取/转换 LoRA B；
4. 使用 Accelerate `cblas_sgemm` 计算该行块的 `B @ A`；
5. 加上基础权重并 round-to-nearest-even 转回 BF16；
6. 通过 `h3_gpu_tensor_write_bf16_range` 上传目标 tensor；
7. M5 路径立即执行现有 INT8 量化并释放临时 BF16 tensor。

不允许创建完整 F32 `W` 或完整 F32 `B@A`。单个 adapter 的 host scratch 目标小于 256 MiB。

### 6.2 多 LoRA 顺序

为与当前离线顺序 folding 完全一致，每个 tile 按 CLI 顺序执行：

```text
tile_bf16 = base_bf16
for adapter in adapters:
    tile_f32 = BF16_to_F32(tile_bf16)
    tile_f32 += effective_scale * B_tile @ A
    tile_bf16 = round_to_BF16(tile_f32)
```

每个 adapter 后都进行一次 BF16 舍入，因此交换两个 LoRA 的顺序可能产生细微差异；顺序必须进入 cache key。

### 6.3 覆盖范围

必须覆盖：

- 50 个主 DiT block 的 QKV、attention output、FC1、FC2；
- LoRA 中存在时的 token-refiner 两个 block；
- AdaLN projection 等 adapter 实际声明的其他二维 BF16 tensor。

不能复用仅面向主 block INT8 矩阵的加载逻辑处理 token-refiner。后者仍走 BF16 线性路径。

## 7. INT8 sidecar 交互

### 7.1 第一版规则

当 `lora_count > 0`：

- 自动发现的 `.h3-int8` sidecar 被禁用；
- 显式设置 `H3_INT8_SIDECAR` 时返回错误；
- 合并后的 BF16 权重继续走现有在线 INT8 量化；
- 日志明确输出 `runtime LoRA disables prequantized sidecar`。

原因是现有 sidecar 表示固定 checkpoint 的量化结果，不能再精确加入 LoRA delta。

### 7.2 后续缓存

第二阶段可按以下 key 生成/复用量化缓存：

```text
base checkpoint fingerprint
+ ordered(adapter SHA-256, effective scale, scope)
+ quantization format version
+ enabled matrix families
```

缓存写入临时目录，完成校验后原子 rename。该阶段不是 MVP 上线前置条件。

## 8. 缓存与生命周期

### 8.1 MVP

prepared-DiT key 增加：

```text
|loras=N
|lora[0]=scope:scale:sha256
|lora[1]=scope:scale:sha256
...
```

实际实现不拼接用户输入的 scale 文本，而是对以下二进制序列计算 SHA-256：格式
版本、adapter 数量，以及按顺序排列的 scope、F32 scale 位模式和 adapter 文件
SHA-256。这样 `0.8` 与 `0.800` 得到相同 key，同时保留 LoRA 顺序差异。

LoRA 不进入 Qwen conditioning key，因为原始文本/视觉 embedding 与 LoRA 无关。相同 prompt、shape、参数和 LoRA 集合可直接命中 prepared DiT；改变顺序、scale、文件内容或 scope 必须重建。

`h3_ctx` 可缓存已解析的 LoRA header、规范化映射和只读文件映射，避免每个任务重复解析及哈希。

### 8.2 不同 prompt 的权重复用

当前 prepared DiT 同时包含不可变权重和 prompt 相关的 refined text、maps 与调制状态。要让不同 prompt 也避免重新合并，应在第二阶段拆分：

```text
h3_dit_weights
  GPU device
  merged/quantized block weights
  fixed projection weights

h3_dit_run_state
  refined text
  layout/maps/RoPE
  AdaLN schedule
  activations and sampler state
```

`h3_ctx` 按 `(variant, LoRA signature, kernel options)` 缓存一个 `h3_dit_weights`。prompt、seed、时长变化只重建 run state。该拆分是获得跨 prompt 启动收益的关键，独立于 CLI LoRA 功能本身。

## 9. 失败处理与可观测性

在写入任何 GPU 权重前完成全量 preflight。以下情况直接失败：

- adapter 文件变化或哈希读取失败；
- A/B、down/up 或 alpha 不完整；
- tensor 名称不能唯一映射；
- adapter 与当前 FL2VA/Ref2VA scope 不匹配；
- rank 或输入/输出 shape 不匹配；
- scale、alpha 或中间结果出现 NaN/Inf；
- 与 `--ssd-streaming` 或显式 sidecar 冲突；
- adapter 数量或总 tensor 数超过限制。

进度阶段增加：

```text
LoRA inspect       files / total
LoRA merge         rows / total rows
LoRA quantize      blocks / active blocks
```

`--profile` 输出每个 adapter 的格式、SHA-256 前缀、scale、scope、匹配 tensor 数、merge wall time 和峰值 scratch；日志不得输出用户目录之外的信息或 adapter 内容。

失败时释放新建的 tensor 和 adapter store，不修改 checkpoint，不替换当前 worker 中仍可用的 prepared DiT。

## 10. 测试计划

### 10.1 单元测试

- 三种键名规范化；
- F16、BF16、F32 解码；
- alpha/rank 与全局 scale；
- 多 LoRA 顺序舍入；
- 缺失 pair、重复 target、未知 dtype、shape 错误；
- NaN/Inf 拒绝；
- FL2VA/Ref2VA scope；
- cache key 随路径内容、scale、顺序、scope 变化；
- sidecar 自动禁用和显式冲突报错。

### 10.2 Tensor 级一致性

用小型 safetensors fixture 比较：

1. Python 离线 folding；
2. C 加载时 folding；
3. 多 LoRA 顺序 folding。

目标是 BF16 输出逐字节一致。至少覆盖 rank 2、16、32、64，以及 F16/BF16/F32 adapter。

### 10.3 真实模型验证

对 Turbo、HMNSFW、Mini_Dick_Fix、Riding Pose 分别执行：

- `--dry-run` 目标覆盖率 100%；
- 运行时 folding 与离线 folding 的短视频 MP4 SHA-256 对比；
- 512×512×22、一个中等时长、一个长时长；
- 文生视频、首帧、首尾帧；
- Ref2VA 仅在显式 `scope=ref2va/both` 下测试。

### 10.4 性能与内存门槛

- 生成阶段 denoise 相对离线 folded 模型回归不超过 2%；
- 单 LoRA host scratch 小于 256 MiB；
- 无完整 F32 权重副本；
- 相同 LoRA 配置的 worker 第二次任务不得重新 merge；
- 不同 LoRA 配置必须可靠失效旧 prepared DiT；
- 失败后下一任务仍能在同一 worker 运行；
- 无 LoRA 路径的现有输出和性能不变。

## 11. 实施阶段

### 阶段 A：MVP CLI 与 C API

- 增加 `h3_lora.c/.h` 与公共参数；
- 移植三种格式解析、dtype 和 alpha/rank；
- 实现 BF16 行分块合并；
- 支持最多 8 个 LoRA；
- LoRA 请求禁用 sidecar 与 SSD streaming；
- 完成 tensor 级逐字节对照。

预计改动：约 700–1200 行；风险中等。

### 阶段 B：Worker 与 Web

- 扩展 worker 协议和请求 schema；
- 增加 LoRA 白名单、选择、排序、scale 与 scope；
- 将 LoRA signature 纳入 prepared key；
- 增加加载进度与失败恢复测试。

预计改动：约 300–600 行。

### 阶段 C：跨 prompt 权重常驻

- 拆分 `h3_dit_weights` 与 `h3_dit_run_state`；
- 同一 LoRA 集合跨 prompt 复用量化权重；
- 为常用 adapter 集合增加受限 LRU。

这是主要架构工作，必须单独评审和基准测试，不与阶段 A 合并提交。

### 阶段 D：可选的真正动态 LoRA

只有在产品明确要求“无需重建即可逐次切换 scale”时再考虑：

```text
y = W_int8(x) + scale * B(A(x))
```

它需要覆盖 fused MLP、QKV、attention-output、token-refiner 与多 LoRA 累加，并会增加每个 denoise step 的 GEMM 和 dispatch。默认不实施。

## 12. 发布门槛与回滚

MVP 进入默认功能前必须同时满足：

1. 所有 parser、shape、alpha、顺序和失败测试通过；
2. 至少一个真实 LoRA 与离线 folding 输出逐字节一致；
3. 三个下载目录 LoRA 均完成一段真实视频验证；
4. 峰值内存和启动耗时有 M5 Pro 64 GB 数据；
5. worker cache 命中、失效、取消、失败恢复通过；
6. 未配置 LoRA 时现有基准无回归；
7. 文档明确 sidecar/SSD streaming 冲突和 scope 行为。

回滚开关：

```text
H3_DISABLE_RUNTIME_LORA=1
```

出现异常时忽略所有 runtime LoRA 参数并给出明确错误，不修改已有 checkpoint 或 sidecar。

## 13. 推荐实施顺序

1. 先完成阶段 A 的 parser、C API 和单 LoRA 分块合并。
2. 用 Turbo adapter 对比当前已验证的离线 folded 模型，建立逐字节 oracle。
3. 增加三种真实 LoRA 与多 LoRA 顺序测试。
4. 接入 worker/API，并把 LoRA signature 加入 prepared key。
5. 完成跨 prompt 权重生命周期拆分后，再考虑将运行时 LoRA 暴露为网页默认功能。
6. 不启动真正的每层动态 LoRA，除非热切换需求证明值得承担持续推理开销。
