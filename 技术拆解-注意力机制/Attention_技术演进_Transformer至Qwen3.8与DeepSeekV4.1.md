# Attention 注意力机制演进：从 Transformer 到 Qwen3.8 与 DeepSeek-V4.1

基础视频解读：[Q、K、V：从三种投影到注意力读取](QKV基础/QKV_从三种投影到注意力读取.md)，包含完整手算、原视频截图与复算代码；随后可读 [多头注意力：从独立投影到拼接融合](多头注意力/多头注意力_从独立投影到拼接融合.md)。

> **技术教材｜2026-11-03**  
> **阅读环境：Typora** · 标准 Markdown + 内嵌 HTML/SVG · 所有图形离线可见  
> **范围：** MHA / MQA / GQA / FlashAttention / MLA / Linear Attention / Gated DeltaNet / Sparse Attention / QSA / CSA2 / CED

<div style="background:#eef6ff;border-left:5px solid #2563eb;border-radius:8px;padding:16px 20px;margin:16px 0;line-height:1.8">
<b>阅读主线</b>：现代 Attention 优化并不是一条单线。它至少沿着四条互相交叉的路线发展：<b>减少 KV Cache</b>、<b>提高 GPU 计算效率</b>、<b>降低长序列计算复杂度</b>、<b>重组跨层的注意力架构</b>。
</div>

## 一、技术地图与原始文献

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;background:#f8fafc;margin:16px 0">
<b>图 1 · 四条 Attention 优化路线（HTML）</b>
<div style="text-align:center;margin:14px auto;padding:12px;background:#dbeafe;border:1px solid #60a5fa;border-radius:8px;max-width:310px"><b>2017 Transformer · Multi-Head Attention</b></div>
<div style="text-align:center;color:#64748b;font-size:22px">↓</div>
<div style="display:flex;gap:9px;flex-wrap:wrap">
<div style="flex:1;min-width:180px;border:1px solid #93c5fd;border-radius:8px;background:#eff6ff;padding:12px"><b>① KV Cache</b><p>MHA → MQA → GQA → MLA</p><small>减少缓存与带宽</small></div>
<div style="flex:1;min-width:180px;border:1px solid #fbbf24;border-radius:8px;background:#fffbeb;padding:12px"><b>② GPU Kernel</b><p>FlashAttention → FA2/FA3</p><small>降低 HBM 读写和调度开销</small></div>
<div style="flex:1;min-width:180px;border:1px solid #5eead4;border-radius:8px;background:#f0fdfa;padding:12px"><b>③ 长序列计算</b><p>Linear / GDN / Sparse</p><small>不必总是访问完整历史</small></div>
<div style="flex:1;min-width:180px;border:1px solid #c4b5fd;border-radius:8px;background:#faf5ff;padding:12px"><b>④ 跨层架构</b><p>Hybrid / CED / CSA2</p><small>跨层共享 KV 与索引</small></div>
</div>
</div>

| 首次公开 | 技术 / 研究 | 原始论文 / 官方资料 | 核心问题 |
|---|---|---|---|
| 2017 | MHA | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 多头建模 Token 关系 |
| 2019 | MQA | [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150) | 一组共享 KV，减少解码缓存 |
| 2022 | FlashAttention | [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 减少 GPU HBM/SRAM 数据搬运 |
| 2023 | GQA | [Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | MHA 与 MQA 之间的折中 |
| 2023 | FlashAttention-2 | [Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | 提升硬件利用率与并行效率 |
| 2024 | MLA | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 潜在维度压缩 KV |
| 2024–2025 | Gated DeltaNet | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | 以有限状态高效记忆历史 |
| 2025 | Hybrid Attention | [Qwen3-Next 官方博客](https://qwen.ai/blog?id=qwen3-next) | 混合 GDN 与传统 Attention |
| 2025 | DeepSeek Sparse Attention | [DeepSeek-V3.2 文档](https://huggingface.co/docs/transformers/model_doc/deepseek_v32) | 索引式稀疏选择 |
| 2026 | CSA / HCA | [DeepSeek-V4 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | 压缩注意力和长上下文检索 |
| 2026 | GDN + QSA | [Qwen3.8-Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) · [arXiv 2608.30320](https://arxiv.org/abs/2608.30320) | 高效记忆与 Micro-block 检索 |
| 2026 | CED + CSA2 | [DeepSeek-V4.1-Flash 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | 跨层共享与极限 KV 压缩 |

> **阅读提醒：** 上表按研究首次公开的时间组织，不代表后一篇论文必然在前一篇基础上直接升级。

## 二、2017：原始 Multi-Head Attention

原始 Attention：

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$$

$$\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

MHA 将输入投影到多个 Head，分别计算注意力并拼接输出。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:14px 0;background:#fff">
<b>图 2 · MHA 多头并行计算（HTML）</b>
<div style="text-align:center;margin:12px auto;padding:10px;background:#dbeafe;border:1px solid #93c5fd;border-radius:7px;max-width:250px">Input Hidden States</div>
<div style="text-align:center;color:#94a3b8;font-size:20px">↙　↓　↘</div>
<div style="display:flex;gap:8px;text-align:center;flex-wrap:wrap">
<div style="flex:1;min-width:120px;background:#e0f2fe;border:1px solid #38bdf8;padding:13px;border-radius:7px"><b>Head 1</b><br>Q₁ K₁ V₁</div>
<div style="flex:1;min-width:120px;background:#ccfbf1;border:1px solid #2dd4bf;padding:13px;border-radius:7px"><b>Head 2</b><br>Q₂ K₂ V₂</div>
<div style="flex:1;min-width:120px;background:#f3e8ff;border:1px solid #c084fc;padding:13px;border-radius:7px"><b>Head 3</b><br>Q₃ K₃ V₃</div>
</div>
<div style="text-align:center;color:#94a3b8;font-size:20px">↘　↓　↙</div>
<div style="text-align:center;margin:auto;padding:10px;background:#fef3c7;border:1px solid #fbbf24;border-radius:7px;max-width:250px"><b>Concat + Output Projection</b></div>
</div>

**两个瓶颈：**

1. 完整注意力计算 $QK^T$ 的复杂度约为 $O(N^2d)$，随序列长度 $N$ 二次增长。
2. 自回归推理的历史 Key/Value 通常需要缓存；上下文与并发增长时，显存与内存带宽可能成为瓶颈。

标准 Decoder-only KV Cache 的近似大小（单位 Byte）：

$$M_{KV}=2\times L\times N\times H_{KV}\times d_h\times b$$

其中 $L$ 为缓存层数、$N$ 为历史 Token 数、$H_{KV}$ 为 KV Heads、$d_h$ 为 Head 维度、$b$ 为每元素字节数。不同模型可能还有额外状态、压缩表示或量化元数据，不能无条件套用这个式子。

## 三、2019–2023：MHA → MQA → GQA

MQA 的思路：多个 Query Heads 共用一组 Key/Value；GQA：一组 Query Heads 共用一组 Key/Value。

假设 32 个 Query Heads，并保持单个 KV Head 的维度和精度不变：

| 架构 | Query Heads | KV Heads | 相对 MHA KV Cache |
|---|---:|---:|---:|
| MHA | 32 | 32 | 100% |
| GQA | 32 | 8 | 25% |
| MQA | 32 | 1 | 3.125% |

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:14px 0;background:#f8fafc">
<b>图 3 · MHA / GQA / MQA 的 KV Cache 相对大小（HTML）</b>
<div style="margin-top:13px"><strong>MHA：32 KV Heads</strong><div style="height:25px;width:100%;background:#64748b;border-radius:5px;color:white;text-align:center">100%</div></div>
<div style="margin-top:13px"><strong>GQA：8 KV Heads</strong><div style="height:25px;width:25%;background:#3b82f6;border-radius:5px;color:white;text-align:center">25%</div></div>
<div style="margin-top:13px"><strong>MQA：1 KV Head</strong><div style="height:25px;width:3.125%;min-width:12px;background:#0d9488;border-radius:5px"></div><span style="font-size:13px;color:#64748b">3.125%</span></div>
</div>

**意义：** 减少 KV Heads 降低缓存和 Decode 阶段读取带宽，但实际质量与性能取决于训练方式。GQA 论文提供从 MHA Checkpoint 继续训练为 GQA/MQA 的方法。

## 四、2022–2024：FlashAttention 是 GPU 实现优化

FlashAttention 的核心不是删除注意力关系，而是利用分块计算、Online Softmax 和硬件存储层次降低 HBM 访问。它计算的是精确 Attention（允许浮点误差）。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;background:#fff;margin:14px 0">
<b>图 4 · 标准 Attention 与 FlashAttention（HTML）</b>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">
<div style="flex:1;min-width:220px;border-radius:8px;padding:14px;background:#fef2f2;border:1px solid #fca5a5"><b>朴素实现</b><p>Q × Kᵀ → 大型中间 Score 矩阵 → Softmax → × V</p><small>较多中间数据写入/读取 HBM</small></div>
<div style="flex:1;min-width:220px;border-radius:8px;padding:14px;background:#ecfdf5;border:1px solid #6ee7b7"><b>FlashAttention</b><p>Q/K/V 分块 → 在线归一化 → 分块聚合</p><small>减少中间结果落到 HBM，提升 IO 效率</small></div>
</div>
</div>

**FlashAttention ≠ Sparse Attention：** 前者主要优化如何高效算，后者改变需要算哪些 Token/Block。FlashAttention-2 则进一步优化并行与 Warp 分工。原文：[FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)。

## 五、2024：DeepSeek MLA——压缩 KV 表示

MQA/GQA 是减少 KV Head 数量；MLA 进一步尝试将每个 Token 的 KV 历史压缩到低秩潜在表示：

$$c_t^{KV}=W^{DKV}h_t$$

在需要计算时，可经上投影得到内容分量的 Key/Value：

$$k_t\approx W^{UK}c_t^{KV},\quad v_t\approx W^{UV}c_t^{KV}$$

这只是概念化的内容分支公式。实际 MLA 还对 RoPE 位置分量单独处理，缓存内容也不是单一的 $c_t^{KV}$。MLA 通过矩阵吸收等技术避免无谓恢复所有多头 KV。

<div style="background:#ecfdf5;border-left:5px solid #059669;border-radius:7px;padding:14px 18px;margin:14px 0"><b>记忆点：</b>GQA 减少 KV 的<strong>组数</strong>；MLA 压缩 KV 的<strong>表示维度</strong>。两者不是同一种技术。</div>

原始论文：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)。

## 六、Linear Attention 与 Gated DeltaNet

传统完整 Attention 会读取历史 KV；部分 Linear Attention 使用固定尺寸状态 $S_t$ 累积历史信息。

简化示意：

$$S_t=S_{t-1}+\phi(k_t)v_t^T$$

$$o_t\propto\phi(q_t)^TS_t$$

其中实际实现可能还有归一化项。线性递归能避免每步对全部历史显式做完整 $QK^T$，但如何删除过时信息、修改错误记忆是难题。

Gated DeltaNet 将门控与 Delta Rule 引入状态更新。一个教学化简式：

$$S_t=\alpha_t S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^T$$

$\alpha_t$ 控制历史保留，$\beta_t$ 控制更新强度，$(v_t-S_{t-1}k_t)$ 是新信息和当前状态预测的差值。真实模型有归一化、特征变换和并行 Kernel 细节。

**核心理解：** 不只是不断增加历史状态，还能根据当前内容对状态进行修正。原文：[Gated Delta Networks](https://arxiv.org/abs/2412.06464)。

## 七、Qwen3.8-Flash-Next：GDN + QSA

**官方资料：** [Qwen GitHub/技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next) · [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) · [阿里云技术博客](https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501) · [arXiv 2608.30320](https://arxiv.org/abs/2608.30320)

Qwen3.8-Flash-Next 的注意力层分布：

$$12\times(3\times\text{Gated DeltaNet}+1\times\text{Qwen Sparse Attention})$$

即 48 层中有 36 层 GDN 和 12 层 QSA（每个注意力子层后还有相应的 MoE 等结构）。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:14px 0;background:#f8fafc">
<b>图 5 · Qwen3.8-Flash-Next 的 48 层 Attention 组织（HTML）</b>
<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px">
<div style="background:#fff;border:1px solid #cbd5e1;padding:8px;border-radius:6px;text-align:center">组 1–3<div style="display:flex;margin-top:5px;height:19px"><span style="flex:3;background:#3b82f6"></span><span style="flex:1;background:#0d9488"></span></div></div>
<div style="background:#fff;border:1px solid #cbd5e1;padding:8px;border-radius:6px;text-align:center">组 4–6<div style="display:flex;margin-top:5px;height:19px"><span style="flex:3;background:#3b82f6"></span><span style="flex:1;background:#0d9488"></span></div></div>
<div style="background:#fff;border:1px solid #cbd5e1;padding:8px;border-radius:6px;text-align:center">组 7–9<div style="display:flex;margin-top:5px;height:19px"><span style="flex:3;background:#3b82f6"></span><span style="flex:1;background:#0d9488"></span></div></div>
<div style="background:#fff;border:1px solid #cbd5e1;padding:8px;border-radius:6px;text-align:center">组 10–12<div style="display:flex;margin-top:5px;height:19px"><span style="flex:3;background:#3b82f6"></span><span style="flex:1;background:#0d9488"></span></div></div>
</div>
<p style="font-size:13px;color:#64748b"><span style="color:#3b82f6">■</span> GDN：每组 3 层；<span style="color:#0d9488">■</span> QSA：每组 1 层。每个小块代表连续的三个四层组，共 12 组。</p>
</div>

### 7.1 QSA 是怎样减少长上下文计算的？

QSA 采用压缩轻量 Indexer，先定位重要的 Micro-block，再读取选中块对应的 KV 并执行稀疏注意力。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:14px 0;background:#fff">
<b>图 6 · QSA 两阶段处理（HTML）</b>
<div style="display:flex;align-items:center;justify-content:center;gap:7px;flex-wrap:wrap;text-align:center;margin-top:12px">
<div style="background:#dbeafe;border:1px solid #93c5fd;padding:11px;border-radius:7px">当前 Query</div><strong>→</strong>
<div style="background:#fef3c7;border:1px solid #fbbf24;padding:11px;border-radius:7px">Compressed Indexer<br><small>选重要 Micro-block</small></div><strong>→</strong>
<div style="background:#ccfbf1;border:1px solid #5eead4;padding:11px;border-radius:7px">读取选中 KV</div><strong>→</strong>
<div style="background:#f3e8ff;border:1px solid #c4b5fd;padding:11px;border-radius:7px">Sparse Attention</div>
</div>
</div>

官方模型结构中 QSA 预算为 512 Blocks / 2048 Tokens。若完整上下文为 262,144 Tokens，核心选取的 Token 比例为：

$$\frac{2048}{262144}\times100\%=0.78125\%$$

**警告：** 0.78% 是选中 Token 比例，不是整个 QSA 模块的总 FLOPs、延迟或带宽成本。Indexer 本身也要计算。

### 7.2 官方关键参数

| 参数 | 值 |
|---|---:|
| 总层数 | 48 |
| GDN 层 | 36 |
| QSA 层 | 12 |
| GDN V Heads / QK Heads | 48 / 16 |
| GDN Head Dim | 128 |
| QSA Q Heads / KV Heads | 24 / 2 |
| QSA Head Dim | 256 |
| QSA RoPE Dim | 64 |
| Indexer Q Heads / Shared Key Heads | 4 / 1 |
| Indexer Head Dim | 128 |
| Sparse Budget | 512 Blocks / 2048 Tokens |
| 原生上下文 | 262,144 Tokens |

**路线总结：** GDN 用低成本状态保留大量历史模式；QSA 选择性读取重要的原始或块级上下文，以弥补有限状态对精确检索的不利之处。

## 八、DeepSeek：MLA → DSA → CSA / HCA → CSA2

DeepSeek-V3.2 通过 DeepSeek Sparse Attention（DSA）引入可学习的索引选择，在 MLA 基础上减少长上下文的 Attention 计算。[官方文档](https://huggingface.co/docs/transformers/model_doc/deepseek_v32)

DeepSeek-V4 则利用：

- **CSA（Compressed Sparse Attention）**：压缩历史并选择重要的条目。
- **HCA（Heavily Compressed Attention）**：更高压缩比，覆盖更长历史。

这些技术还必须处理压缩块的时间位置与因果 Mask，不能让一个当前 Token 读取到未来历史块中的信息。[DeepSeek-V4 原始技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。

## 九、DeepSeek-V4.1-Flash：CED + CSA2

**官方资料：** [技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) · [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [发布博客](https://deepseek.com/en/news/deepseek-v4-1-flash/)

### 9.1 Causal Encoder–Decoder（CED）

V4.1-Flash 使用 40 层 CED：20 层 Causal Encoder + 20 层 Decoder。这里的 Encoder **并非原始 Transformer 那种双向 Encoder**。它保持因果约束，用于自回归架构。

重要变化：Decoder 的全局 KV Cache 从最终 Encoder Hidden States 投影，而不需要每个 Decoder 层从自己的 Hidden States 重新构造完全独立的全局 KV。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;background:#f8fafc;margin:14px 0">
<b>图 7 · DeepSeek-V4.1 CED 与共享全局 KV（HTML）</b>
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;justify-content:center;text-align:center;margin-top:13px">
<div style="background:#dbeafe;border:1px solid #93c5fd;border-radius:7px;padding:12px">Input Tokens</div><b>→</b>
<div style="background:#dbeafe;border:1px solid #93c5fd;border-radius:7px;padding:12px">20 层<br><b>Causal Encoder</b></div><b>→</b>
<div style="background:#ccfbf1;border:1px solid #5eead4;border-radius:7px;padding:12px">Encoder Final Hidden<br>→ Shared Global KV</div><b>→</b>
<div style="background:#f3e8ff;border:1px solid #c4b5fd;border-radius:7px;padding:12px">20 层<br><b>Decoder</b></div><b>→</b>
<div style="background:#fef3c7;border:1px solid #fbbf24;border-radius:7px;padding:12px">Output</div>
</div>
<p style="font-size:13px;color:#64748b;margin-bottom:0">突出全局 KV 的共享来源；省略 Sliding Window、复用模式、残差连接与完整生成循环。</p>
</div>

### 9.2 CSA2 的三种模式

| 模式 | 作用 |
|---|---|
| Full | 建立本层需要的主要历史访问与索引基础 |
| Reindex | 重新计算重要上下文的索引 |
| Reuse | 复用已得到的索引，减少跨层重复计算 |

Hierarchical Sparse Indexer 将后续层的搜索范围限制在前层构造的候选集合内，减小深层索引成本。FP4 主 KV Cache 进一步降低存储占用。SWA Bounded Replay 则按需重放最近窗口 Token，避免持久保存全部局部窗口 KV。

### 9.3 关键规模

| 项目 | 官方披露值 |
|---|---:|
| 模型主干参数 | 552B |
| Causal Encoder / Decoder | 20 / 20 层 |
| Prefill 激活参数 | 8B |
| Decode 激活参数 | 16B |
| 全局 KV Cache | 890 Bytes / Token |
| 相对 V4-Flash 全局 KV | 约 1/4 |
| 上下文支持 | 约 1M Tokens |

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:14px 0;background:#fff">
<b>图 8 · DeepSeek-V4.1 全局 KV Cache 与 V4-Flash 对比（HTML）</b>
<div style="margin-top:13px"><strong>V4-Flash（作为 100% 参照）</strong><div style="height:25px;width:100%;background:#64748b;border-radius:5px"></div></div>
<div style="margin-top:13px"><strong>V4.1-Flash（约 25%）</strong><div style="height:25px;width:25%;background:#0d9488;border-radius:5px"></div></div>
<p style="font-size:13px;color:#64748b;margin-bottom:0">比例为官方报告的全局 KV 存储量相对值；不代表模型权重、局部缓存或推理总显存按相同比例下降。</p>
</div>

## 十、Qwen 与 DeepSeek 的架构对照

| 对比 | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
|---|---|---|
| 主要路线 | GDN + QSA 混合注意力 | CED + CSA2 压缩稀疏注意力 |
| 历史压缩方式 | GDN 固定状态 | 压缩 KV / 跨层共享 |
| 历史精确检索 | Micro-block 索引器 | Hierarchical Sparse Indexer |
| 层结构 | 36 GDN + 12 QSA | 20 Causal Encoder + 20 Decoder |
| KV 优化重点 | 状态压缩与稀疏读取 | 跨层 KV / 索引复用及量化 |
| Prefill / Decode | 混合层高效执行 | 非对称激活参数与 CED |
| 长上下文目标 | ~1M | ~1M |

<div style="display:flex;flex-wrap:wrap;gap:12px;margin:16px 0">
<div style="flex:1;min-width:220px;padding:15px;border-radius:8px;background:#eff6ff;border-left:4px solid #3b82f6"><b>Qwen 思路</b><p>用状态记忆大量历史，再用稀疏块检索补充精确细节。</p></div>
<div style="flex:1;min-width:220px;padding:15px;border-radius:8px;background:#ecfdf5;border-left:4px solid #059669"><b>DeepSeek 思路</b><p>压缩历史 KV，同时让全局 KV 与索引在多层之间复用。</p></div>
</div>

不能只凭架构表判断哪家推理一定更快。还应控制模型质量、Token 长度、并发、输出长度、GPU、KV 精度、索引器开销、Kernel 及缓存命中率等条件。

## 十一、精读与部署实践建议

建议按以下顺序精读八篇文献：

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Q/K/V、MHA、计算复杂度。
2. [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)：MQA 与 Decode 带宽。
3. [GQA](https://arxiv.org/abs/2305.13245)：分组共享 KV。
4. [FlashAttention](https://arxiv.org/abs/2205.14135)：Online Softmax 与 HBM/SRAM。
5. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)：MLA 与低秩 KV。
6. [Gated Delta Networks](https://arxiv.org/abs/2412.06464)：状态更新与 Delta Rule。
7. [Qwen3.8-Next 技术报告](https://arxiv.org/abs/2608.30320)：GDN 与 QSA。
8. [DeepSeek-V4.1-Flash 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：CED、CSA2、KV/索引复用。

### 对 vLLM / L20 实践最重要的测量指标

| 观察对象 | 代表指标 | 目的 |
|---|---|---|
| Prefill | Prompt Tokens/s、TTFT | 观察长 Prompt 处理效率 |
| Decode | Generation Tokens/s、TPOT | 观察带宽与缓存压力 |
| KV / 状态 | GPU Cache 使用率、KV dtype | 理解并发与上下文容量 |
| Sparse Indexer | 检索预算、Block 数、Indexer 开销 | 避免只统计主 Attention 的算量 |
| 并行与分离 | TP/EP、Prefill/Decode 分离、通信成本 | 分析多卡、多机效率 |

## 十二、结论

**现代 Attention 的核心问题，已经从“如何计算 $QK^T$”扩展为：**

1. 如何以更低的成本记住历史？
2. 如何快速找到真正相关的历史？
3. 如何让不同层复用历史表示与索引？
4. 如何让 GPU 高效执行这些数据读写与计算？

Qwen3.8 和 DeepSeek-V4.1 分别给出了不同但相关的答案。想真正读懂两者的技术报告，应把注意力数学、KV 存储、索引与 GPU Kernel 分开理解，再研究其组合。

---

<p style="text-align:center;color:#64748b;font-size:12px">Attention 技术演进笔记 · 内嵌 HTML 图示 · 无需 Mermaid 或外部图片资源</p>
