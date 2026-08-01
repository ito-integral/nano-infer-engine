# Nano Infer Engine 开发路线图

## 项目目标

构建一个小型、易读的 Llama 推理引擎：先使用 PyTorch 证明正确性，再逐步演进为支持在线连续批处理和优化分页注意力的服务。每项性能优化都应保留可读的参考实现，作为正确性基准。

## 当前进展

### 模型、缓存与注意力

- 支持加载转换后的 Hugging Face Llama 配置和权重。
- 支持 dense 与 paged greedy generation，并提供一致性测试。
- 支持带 padding 的 dense batch 和无 padding 的 ragged paged batch。
- 使用每序列独立 block table 管理预分配的 Paged KV Cache。
- 支持 KV block 的分配、写入、收集、回收、导出和导入。
- 提供易读的 PyTorch paged-attention 参考实现。
- 支持历史长度不等的 batched single-token decode。
- 保留 dense cache 和 contiguous gather 路径用于交叉验证。
- 支持 flattened ragged prefill：使用一维 token 布局和 `query_start_loc` 表示不同长度的 query。
- flattened 路径中的投影、RMSNorm 和 MLP 对所有有效 token 批量执行；参考 attention 暂时按序列边界循环计算。

### Prefill 与连续批处理

- 支持将等长 prompt 合并为一次 prefill model call。
- 支持 padding-free chunked prefill，长 prompt 可以跨多个 scheduler step 推进。
- 支持不同尾块长度在同一次 flattened ragged forward 中执行，无需 padding 或串行 model call。
- 使用 `context_length` 表示已有 KV 历史，使用 `query_start_loc` 表示本轮各请求的 query 边界。
- chunked prefill 的最终 logits 和逐层 KV 已与 one-shot prefill 对齐验证。
- 请求生命周期包含 pending、prefilling、active、completed、failed 和 cancelled。
- prefill chunk 与已有 decode step 可以交错执行，避免单个长 prompt 独占一次完整 prefill。

### 调度与请求生命周期

- 在 active batch 和 KV block 预算内接纳请求。
- 在已有请求 decode 时动态加入新请求。
- 请求命中 EOS 或 token 上限后立即退出并回收 block。
- 隔离 prefill 失败，并清理失败的 decode batch。
- 支持显式取消请求和幂等 scheduler shutdown。
- 每步发布 token event 和终态请求结果。

### 异步引擎与 HTTP 服务

- 使用长生命周期 `AsyncInferenceEngine` 驱动 scheduler。
- 支持并发提交，并在空闲时避免 busy loop。
- 提供逐请求异步 token iterator 和最终结果。
- 通过 FastAPI 管理进程级模型和 engine 生命周期。
- 提供 `/health`、`/generate`、`/v1/models` 和兼容 OpenAI 的 `/v1/chat/completions`。
- 支持非流式响应和 SSE token streaming。
- 返回 streaming usage，并以 `[DONE]` 结束流。
- streaming 客户端断开时取消未完成推理。
- 通过环境变量配置模型名、最大上下文、生成长度和 batch size。
- 单 GPU HTTP runtime 可通过 `NANO_PREFILL_CHUNK_SIZE` 和 `NANO_MAX_PREFILL_TOKENS_PER_STEP` 配置 chunked prefill 与全局预算。
- 根据可配置的 GPU memory utilization 推导 KV block 容量。

### Prefill/Decode 分离

- 支持从 prefill cache 导出 KV block，并导入 decode cache。
- 通过同设备和可选双 GPU 测试验证 KV 迁移。
- 支持跨两个 GPU 的同步单请求生成。
- 支持 prefill 与 decode kernel 可重叠的 batched P/D pipeline。
- `PDContinuousBatchingScheduler` 已连接 `AsyncPDInferenceEngine` 和 HTTP 动态提交。
- prefill 与 decode 支持独立的 GPU memory budget。

## 当前架构

```text
兼容 OpenAI 的 HTTP API / 离线调用方
                    |
            AsyncInferenceEngine
                    |
         连续批处理 Scheduler
          /                    \
flattened ragged prefill    batched decode
          \                    /
               PagedKVCache
                    |
              参考注意力实现
```

P/D runtime 使用独立的 prefill/decode 模型、设备和 cache，并负责 KV 迁移，替代单 scheduler、单 cache 的运行方式。

## 已知限制

- flattened ragged attention 仍是 correctness-first PyTorch 参考实现，内部按请求循环并 gather KV，尚未接入 FlashAttention varlen、Triton 或 CUDA kernel。
- chunked prefill 同时支持每请求 `prefill_chunk_size` 和全局每轮 `max_prefill_tokens_per_step`；prefill 与 decode token 尚未合并到同一次 model forward。
- Paged decode 使用 Python 循环和普通 PyTorch operator，每个 token 会启动较多小 kernel。
- engine 尚无 CUDA Graph capture，每个 decode iteration 都会返回 Python。
- SSE detokenization 会反复解码累计 token IDs，尚未使用增量 detokenizer。
- admission 为 active/prefilling 请求预留最坏情况 KV 容量，安全但偏保守。
- scheduler 共享一个 `GenerationConfig`，请求级生成参数仍较有限。
- P/D KV 迁移是同进程 tensor transfer，尚无 CUDA IPC、独立 worker process、P2P transport 或多节点传输。
- Python P/D coordinator 可在双 GPU 上重叠 kernel，但尚未使用显式 CUDA stream 和 event 管理传输流水线。
- 当前仅支持 `temperature=0` 的 greedy decoding。
- dense、padded、paged、ragged 和跨设备 BF16 路径可能因累加顺序不同，在 top logits 非常接近时选择不同 token。

## 阶段一：建立可信的性能基线

使用外部 `inference-benchmark` 项目，避免在本仓库重复实现负载生成器。

### 任务

1. 使用完全一致的模型、dtype、prompt/output 长度、并发数、请求数、上下文限制、显存比例和 warm-up 策略，记录可复现的 vLLM 与 nano-engine 基线。
2. 对比请求吞吐、输出 token 吞吐、TTFT、TPOT、端到端延迟、失败率和 GPU 峰值利用率。
3. 增加轻量指标：实际 decode batch size、prefill token 数、pending/prefilling/active 请求数、空闲 block 数和 scheduler step latency。
4. 修改 kernel 前先使用 PyTorch Profiler 和 Nsight Systems 分析代表性负载。

### 验收标准

- 可使用明确的服务端和客户端命令重复 benchmark。
- 能区分 HTTP、排队、prefill 和 decode 耗时。
- 每项性能修改都与 reference path 和已记录基线比较。

## 阶段二：增加更快的 PyTorch 对照路径

在编写自定义 kernel 前，先确定 reference paged attention 本身造成的性能损失。

### 任务

1. 增加可选 decode 路径：将 paged KV gather 为连续 tensor，并调用 PyTorch SDPA。
2. 保留当前 paged-attention 作为正确性基准。
3. 比较 logits、top-token 一致率、临时显存、TPOT 和吞吐：

   ```text
   paged reference
   -> contiguous gather + PyTorch SDPA
   -> 未来的 fused paged kernel
   ```

4. 不在同一提交中同时改变调度行为和 attention 对照路径。

### 验收标准

- 所有路径通过 block 边界、ragged length、多头、FP32 和 BF16 测试。
- benchmark 能判断主要瓶颈来自 attention kernel 还是外围 engine 开销。

## 阶段三：Batched 与 Chunked Ragged Prefill

这一阶段的 reference 功能闭环已经完成，后续重点转向调度策略和性能优化。

### 已完成

1. 等长 prompt 批量 prefill，并保持 logits 的原请求顺序。
2. 使用 `PREFILLING` 和 `prefill_offset` 跨 scheduler step 推进长 prompt。
3. 使用 flattened token layout 与 `query_start_loc` 支持不同长度 chunk 的单次 ragged forward。
4. 保持 padding-free KV storage，每个请求只写入自己的有效 token。
5. prefill chunk 与 decode step 交错，长 prompt 不再一次性阻塞全部 decode。
6. 验证 one-shot、chunked 和不同尾块 ragged prefill 的 logits/KV 一致性。
7. 使用全局 `max_prefill_tokens_per_step` 限制单轮 prefill 总工作量。
8. 使用轮转队列分配预算，预算不足时优先推进上一轮未获得计算的请求。

### 待完成

1. 评估统一的 prefill/decode token budget；当前 prefill 使用独立预算，并在同一 scheduler step 内先执行 prefill forward、再执行 decode forward。
2. 评估是否将 prefill 与 decode token 合并到同一个 flattened model forward。
3. 增加实际 prefill token 数、ragged batch 请求数和 padding-free 利用率指标。
4. 在混合 prompt 长度与持续请求到达场景下验证长期公平性。

### 验收标准

- 多个不同长度 prompt 可在少于串行实现的 model call 数量内完成。
- 单个长 prompt 不能无限阻塞已有 decode 请求。
- 每轮 prefill 工作量受全局 token budget 严格限制。
- batched/chunked prefill 在定义的 dtype tolerance 内匹配 one-shot prefill。

## 阶段四：优化 Ragged/Paged Attention Kernel（暂缓）

当前先保留 PyTorch reference path，不急于接入 FlashAttention 或 Triton。完成基线、指标和 token-budget scheduler 后，再根据 profiling 结果选择优化方向。

### 候选任务

1. 评估 FlashAttention varlen 作为 flattened ragged prefill backend。
2. 为 decode 实现直接读取 block table 的 fused paged-attention kernel。
3. 支持 ragged sequence length、GQA/MQA 和任意 final-block occupancy。
4. 使用数值稳定的 online softmax 跨 block 计算。
5. 减少 kernel launch 和中间 tensor，并针对常见 head dimension 调优。

### 验收标准

- 与 reference path 通过相同的正确性测试。
- 使用 tolerance 验证 logits，并报告 top-1 不一致时的 margin。
- 只有 profiling 证明收益后，才引入更高维护成本的自定义 kernel。

## 阶段五：降低逐 token 运行时开销

1. 将累计 SSE decoding 替换为正确的增量 detokenizer。
2. 减少每步 Python object 创建和 scheduler bookkeeping。
3. 按 profiling 结果融合或编译 RoPE、RMSNorm、MLP 和 token selection 等热点算子。
4. 为稳定 decode batch shape 增加 CUDA Graph capture，并保留动态 shape 的 eager fallback。
5. 将输出处理移出 GPU 调度关键路径。

### 验收标准

- SSE 输出与非流式输出文本等价。
- profiling 显示单 token CPU 时间和 GPU kernel launch 数下降。
- 优化不削弱取消、异常清理和 block 回收语义。

## 阶段六：调度与内存策略改进

1. 将 `max_new_tokens`、EOS 和未来 sampling 参数移入每请求状态。
2. 使用增量 block 分配和 decode safety reserve 替代最坏情况预留。
3. 增加 token-budget scheduling、公平性、anti-starvation 和过载 backpressure。
4. 增加请求超时、prompt 大小验证和显式队列限制。
5. 在混合负载下测量 KV fragmentation 和 admission efficiency。

## 阶段七：生产化 P/D 传输

1. 增加显式 CUDA stream 和 event，安全测量并重叠 prefill、KV transfer 和 decode。
2. 将 prefill/decode worker 移入独立进程。
3. 定义支持 staged host copy、CUDA IPC 和拓扑允许时 P2P 的传输接口。
4. 增加有界传输队列、backpressure、handoff 期间取消和兼容性元数据校验。
5. 在混合 prompt 长度下比较单 GPU、双 GPU P/D 和 vLLM，而不是假设 P/D 总会更快。

## 阶段八：Sampling 与更广泛的 API 兼容性

- 增加每请求 temperature、top-k、top-p 和确定性随机种子。
- 增加 stop string、多 EOS token ID 和 log probabilities。
- 扩展 OpenAI-compatible 参数验证和错误响应。
- 在确有需要时再增加认证、结构化指标和部署文档。

## 推荐的近期顺序

```text
记录 vLLM 与 nano 基线
-> 增加 scheduler 和阶段耗时指标
-> 压测并调优全局 prefill token budget
-> contiguous gather + PyTorch SDPA 对照路径
-> 评估 prefill/decode 统一 flattened batch
-> 根据 profiling 决定 FlashAttention varlen 或 Triton kernel
-> 增量 detokenization 与 CUDA Graph
-> 内存策略和生产化 P/D transport
```

## 开发原则

- 正确性、可观测性和性能修改尽量分开提交。
- 每个优化 kernel 都保留易读的 reference path。
- 优化前先 profiling，并记录完整 benchmark 参数。
- 每条异常路径都测试 allocator accounting 和请求状态。
- BF16 正确性不能只依赖生成文本完全相等，应比较 logits tolerance 和 top-token margin。
- HTTP、调度、cache、attention、transport 和 kernel 保持明确分层与接口边界。
- 每次实现代码或行为变更时，在同一任务中同步检查并更新本路线图；路线图使用中文，并准确区分已完成、待完成和暂缓事项。
