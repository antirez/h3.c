# H3 性能优化设计与验证计划

状态：设计草案

日期：2026-08-23

目标平台：Apple Silicon，当前重点为 M5 Pro/Max

## 0. 实施状态

实验代码位于分支 `codex/h3-performance-lab`，本地 `main` 保持在 `origin/main`。

| 工作项 | 状态 | 入口 |
| --- | --- | --- |
| 非均匀 reuse 搜索 | 已实现，等待模型验证 | `tools/search_reuse_schedule.py` |
| 独立 SDPA 基准 | 已实现并完成首轮 M5 Pro 测试 | `make h3_sdpa_bench` |
| INT8 sidecar | 离线转换器与C直载路径已实现，等待完整模型一致性验证 | `H3_INT8_SIDECAR` |
| 时间维 token reduction | host映射已实现，等待模型质量验证 | `H3_TOKEN_REDUCTION_AXIS=time` |
| 稀疏/量化 attention | 等长序列完整 DiT profile 后决策 | 第7节 |
| M3 INT8、在线FBCache、K/V缓存 | 暂缓 | 第8节 |

## 1. 目标

在保持可接受生成质量的前提下，降低 H3 视频生成的冷启动时间和 DiT 推理时间，并为长视频、高分辨率和低内存设备建立可扩展路径。

预期目标：

- 512×512、22帧：同等可用质量下端到端提速 1.2–1.5 倍。
- 768×1344、约5秒：端到端提速 1.7–2.5 倍。
- 冷启动及不同 prompt 的模型准备阶段：显著减少重复权重读取和量化。
- 不以“质量不变的5倍提速”作为项目承诺。

## 2. 已确认的代码事实

### 2.1 帧数与 token 数

目标视频帧数按 `5 + 17k` 对齐，时间 latent 为：

```text
latent_t = ((aligned_frames - 5) / 17) * 5 + 2
```

证据：[h3_host.c](./h3_host.c#L15-L38)。空间经过 VAE `/16`，再按 `2×2` patch 形成视频 token，证据：[h3_host.c](./h3_host.c#L231-L266)。

| 场景 | 对齐帧数 | latent_t | 视频 token | 音频 token | 总序列长度，假设100文本 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512²、22帧 | 22 | 7 | 1,792 | 74 | 约1,966 |
| 768×1344、5秒请求 | 124 | 37 | 37,296 | 414 | 约37,810 |
| 768×1344、约15秒上限 | 362 | 107 | 107,856 | 1,206 | 约109,162 |

5秒请求对应120帧，但会被对齐为124帧，实际约5.17秒。当前生成上限为362帧，约15.08秒。

### 2.2 每层理论 FLOPs

模型主要维度为：

```text
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = 7168
FFN = 14336
```

证据：[h3_dit.c](./h3_dit.c#L15-L29)。按一次乘加计2 FLOPs：

```text
线性层 = 2 × N × (4 × HIDDEN × INNER + 3 × HIDDEN × FFN)
       ≈ N × 770.7M FLOPs

Attention QK + PV = 4 × N² × INNER
```

理论上，约在 `N=26,880` 时 attention FLOPs 超过主要线性层。该比例不等于实际 GPU wall time，实施前必须实测。

### 2.3 当前后端与缓存

- DiT attention 使用按 shape 缓存的 MPSGraph SDPA，见 [h3_gpu.m](./h3_gpu.m#L1470-L1538)。
- 当前没有自定义 dense flash 或 block-sparse attention kernel。
- `H3_REUSE_STEPS` 已支持自定义评估步集合，但仅在 `reuse > 1` 时启用，并要求包含首末步，见 [h3_dit.c](./h3_dit.c#L2540-L2580)。
- 所有 timestep 的 AdaLN 调制会在加载 DiT 时预计算到 GPU tensor，见 [h3_dit_schedule.c](./h3_dit_schedule.c#L237-L330)。
- M5 INT8 权重在 `load_core()` 中运行时量化，见 [h3_dit.c](./h3_dit.c#L1241-L1267)。
- prepared-DiT cache key 包含 prompt；更换 prompt 会重建 DiT，见 [h3.c](./h3.c#L139-L190)。
- M5 TensorOps 当前依靠设备名称包含 `M5` 启用，见 [h3_gpu.m](./h3_gpu.m#L382-L399)。

## 3. 工作流一：非均匀步间复用

### 3.1 目标

在相同或更少 DiT 评估次数下，提高固定 `reuse=2/3` 的质量—速度比。

### 3.2 第一阶段：零代码 schedule 搜索

利用现有 `H3_REUSE_STEPS`，对不同评估步集合进行离线网格搜索。运行时仍需指定 `--reuse 2` 或 `--reuse 3` 以进入 velocity 缓存路径。

实验要求：

- 固定 prompt、seed、分辨率、帧数、steps 和 layers。
- 首轮保持与 `reuse=2` 相同的评估次数，只改变评估位置。
- 第二轮逐渐减少评估次数，寻找质量拐点。
- 每组至少覆盖低运动、中运动、高运动及镜头切换场景。

质量指标：

- 最终视频 SSIM/LPIPS。
- 光流一致性、闪烁和运动拖影。
- 主体身份、肢体和构图人工检查。
- 音频连续性及口型同步。

### 3.3 第二阶段：AdaLN 离线调度

AdaLN 调制只依赖模型、sigma schedule 和条件类型，不依赖 prompt。实现 GPU 归约，计算相邻 timestep 调制向量的相对距离，只回读每步或每层的少量标量。

建议输出：

```text
model_hash + steps + condition_mode -> selected_steps
```

这是一种模型级非均匀 schedule，不是真正根据当前视频内容动态决策的 TeaCache。

### 3.4 验收门槛

- 相同评估次数下，质量稳定优于或等于固定 reuse。
- 或在质量门槛内，将20步的11次评估减少到8–9次。
- 短视频 DiT 时间目标降低10%–30%。

## 4. 工作流二：SDPA 分段实测

### 4.1 目标

确认理论 attention FLOPs 是否转化为实际 wall-time 热点，避免过早投入自定义 kernel。

### 4.2 测试矩阵

至少覆盖以下总序列长度：

```text
N ≈ 2k / 8k / 20k / 38k
```

每个形状测试：

- 完整 attention。
- 当前水平 token reduction。
- BF16 QKV/输出投影。
- 默认 M5 INT8 QKV/输出投影。

采集：

- SDPA GPU 时间。
- QKV、attention-output 和 MLP 时间。
- command-buffer 等待时间。
- 峰值 tensor storage 和系统物理内存。
- 首次 graph 编译与 warm-run 分离数据。

### 4.3 决策门槛

- 若 `N≤8k` 时 SDPA 小于 DiT wall time 的15%，不开发自定义 attention。
- 若 `N≥20k` 时 SDPA 超过35%，进入 BF16 block-tiled 原型。
- 若现有 token reduction 已将 SDPA 降到25%以下，优先继续 token 路线。

### 4.4 首轮独立 SDPA 基准

2026-08-23 在 M5 Pro 64 GB 上使用新增的 `h3_sdpa_bench` 测得：

| N | Head-major wall time | Row-major wall time | Q/K/V/O buffers |
| ---: | ---: | ---: | ---: |
| 2,000 | 5.78 ms | 7.37 ms | 0.107 GiB |
| 8,000 | 65.98 ms | 70.30 ms | 0.427 GiB |
| 20,000 | 463.48 ms | 504.21 ms | 1.068 GiB |
| 38,000 | 1,828.83 ms | 1,980.51 ms | 2.029 GiB |

该测试隔离测量单次56头、128维 BF16 SDPA，不包含QKV、输出投影、MLP及其他DiT操作。`root_gpu_ms` 对较大MPSGraph任务可能为零，因为实际工作由子command buffer执行，因此以同步 wall/wait 时间为准。

复测命令：

```sh
make h3_sdpa_bench
./h3_sdpa_bench --warmup 1 --repeats 3 2000 8000 20000 38000
./h3_sdpa_bench --warmup 1 --repeats 3 --row-major 2000 8000 20000 38000
```

## 5. 工作流三：INT8 sidecar 与权重生命周期解耦

### 5.1 问题

当前每次构建 DiT 都从 BF16 checkpoint 加载主要矩阵并在 GPU 上量化。由于 prepared-DiT key 包含 prompt，不同 prompt 也会触发重建。

### 5.2 设计

增加离线转换工具，生成面向当前 M5 kernel 的量化 sidecar：

- FC1、FC2、QKV、attention-output 保存为 I8。
- 保存对应 F32 per-channel/per-group scales。
- manifest 记录原模型哈希、量化算法版本、group size、shape、内核布局和对齐信息。
- 加载时优先校验并 mmap sidecar；不匹配时回退到 BF16 在线量化。

同时将 `h3_dit` 拆分为：

```text
不可变 prepared weights
        +
prompt/shape 相关 execution state
```

不同 prompt 只重建 refined text、layout、AdaLN maps 和激活，不重新加载及量化19B级主要矩阵。

### 5.3 预期收益

- 稳态 denoise FLOPs不变。
- 冷启动和不同 prompt 的准备阶段预计减少数十秒，实际数值待profile。
- 避免在线量化期间的BF16与INT8临时双份峰值。

### 5.4 验收门槛

- sidecar 输出与当前在线 INT8 路径逐字节一致，或满足既有数值容差。
- 原 checkpoint 被修改、替换或量化参数变化时必须拒绝旧缓存。
- warm与cold加载分别测试，不把文件系统缓存误认为sidecar收益。

## 6. 工作流四：时间维 token reduction

### 6.1 设计原则

现有 GPU pooling/expand kernel 已由 pair、baseline 和 parent map 驱动，可以复用。需要修改的主要是 host 端：

- reduced token 数计算。
- 时间方向 pair 生成。
- full token 到 reduced parent 的映射。
- reduced RoPE 与调制 row map。

现有水平方向硬编码见 [h3_dit.c](./h3_dit.c#L447-L500)。

### 6.2 时间配对策略

不能从第0个时间 latent 开始简单两两平均。目标视频的时间跨度使用 `{1,4,4,4,4}` 周期，见 [h3_host.c](./h3_host.c#L12-L13)。

初始策略：

1. 第0个时间 latent 永久保留。
2. 后续 latent 按时间覆盖范围和相邻性配对。
3. 奇数尾部 token 原样保留。
4. reduced RoPE 使用两个来源 token 的实际时间坐标均值。
5. 默认仅在 `latent_t≥20` 时允许启用。

### 6.3 实验组合

- 不 reduction。
- 当前水平 reduction。
- 时间 reduction。
- 时空组合 reduction，仅作为激进实验。

时间和水平 reduction 不能简单按收益相乘；它们都通过降低N减少线性层与平方 attention成本。

### 6.4 预期收益

- 512²×22帧：不建议默认启用，运动质量风险高。
- 5秒以上长视频：中间block的 DiT 时间预计降低40%–60%。
- 必须重点检查运动速度、时间连贯性、首帧稳定性和拖影。

## 7. 工作流五：长序列稀疏或量化 attention

### 7.1 进入条件

只有 SDPA 实测满足热点门槛后才启动此工作流。

### 7.2 分阶段实现

1. 建立 BF16 block-tiled attention correctness oracle。
2. 与 MPSGraph SDPA 对比不同N下的性能和误差。
3. 单独评估 dense INT8 QK，不与稀疏策略捆绑。
4. 加入基于 `(t,h,w)` 的局部窗口或 block-sparse布局。
5. 再处理全局 token 与局部视频 token 的双向交互。

### 7.3 稀疏布局约束

`layout.positions` 提供三维坐标，但仍需构建窗口索引、block映射和边界策略。文本及少量音频可以保持全局；长参考视频不能全部视为全局 token，否则 Ref2VA 中仍会保留近似平方复杂度。

稀疏 attention 会改变模型原本的全局自注意力语义，应考虑：

- shifted window或周期性全局层。
- 全局摘要 token。
- 时间方向膨胀窗口。
- reference video 自身的局部化或摘要策略。

### 7.4 预期收益

- `N≈2k`：预计只有1%–5%的 DiT收益，不值得优先开发。
- `N≈38k`：dense attention加速2倍，理论整层约1.4倍；高效稀疏实现可能使DiT达到约1.5–2倍。
- 与 token reduction 同时启用时，attention的边际收益会下降。

## 8. 暂缓方向

### 8.1 M3 INT8

当前 INT8 kernel 依赖 M5 Metal 4 TensorOps。M3 在约2k序列下主要矩阵倾向算力受限；如果只能将 INT8 反量化后送入原有浮点管线，预计没有显著速度收益，主要价值是内存占用。

除非独立 microbenchmark 证明存在高效的公开整数 dot/matrix 路径，否则不进入当前提速排期。

### 8.2 在线 FBCache

在线判断需要：

- 在 block 0 后建立新的提交与同步边界。
- 保存上一轮中间 hidden。
- GPU差值归约及少量CPU读回。
- 根据判断继续当前 forward 或使用缓存结果。

它与 whole-denoiser reuse、core reuse 高度重叠，并可能引入GPU同步气泡，暂放在离线 schedule 之后。

### 8.3 跨步 K/V 缓存

文本 token 每步使用 video timestep 调制，见 [h3_dit_schedule.c](./h3_dit_schedule.c#L434-L457)。参考和条件 token 即使调制行固定，其 hidden 仍会在联合 self-attention 中吸收变化的视频与音频信息。

因此完整 K/V 只能近似周期刷新，不能精确缓存。可继续保留的精确优化包括 refined text、RoPE、AdaLN schedule 和静态输入 patch projection。

## 9. 综合收益预估

这些收益存在重叠，不能直接相加。

| 场景 | 保守、可用质量目标 | 激进实验范围 |
| --- | ---: | ---: |
| 512²×22帧 warm生成 | 1.2–1.5× | 1.8–2.5× |
| 512²×22帧含冷启动 | 1.4–2.0× | 2×以上 |
| 768×1344、约5秒 | 1.7–2.5× | 2.5–3.5× |
| 15秒级长视频 | 2–3.5×端到端 | 质量风险高，不承诺上限 |

其中：

- sidecar主要改善准备时间，不改变denoise吞吐。
- 非均匀reuse和在线FBCache互相重叠。
- 时间池化会降低后续稀疏attention的边际收益。
- VAE、文本编码、音频和FFmpeg不会随DiT优化同比例加速。

## 10. 实施顺序与发布门槛

1. 建立固定prompt/seed的短、中、长序列基准和质量基线。
2. 使用 `H3_REUSE_STEPS` 搜索非均匀schedule。
3. 测量2k、8k、20k、38k序列的SDPA实际wall time。
4. 实现并验证INT8 sidecar及不可变权重生命周期。
5. 增加时间token-reduction实验开关。
6. 只有长序列SDPA热点得到实证后，开发自定义attention。
7. 在线FBCache、M3 INT8和近似K/V缓存保持研究状态。

任何优化进入默认路径前必须同时满足：

- 至少两类内容、多个seed和三个时长档位。
- 记录冷启动、warm运行和热降频后的交叉测试。
- 没有崩溃、NaN、明显闪烁、肢体破坏或音频失真。
- 对近似优化明确标注质量变化，提供关闭开关。
- 端到端收益与单kernel收益分别报告。

## 11. 当前限制

当前工作区没有 `MiniMax-H3` 权重，本文的代码事实和FLOPs已经过源码核对，但实际wall-time、画质和内存收益仍需恢复模型后验证。理论估算不得替代真实设备上的ABBA测试。
