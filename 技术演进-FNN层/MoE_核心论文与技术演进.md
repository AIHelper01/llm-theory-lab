# MoE 核心论文与技术演进

> **整理日期：** 2026-11-03  
> **主题：** 从 Sparsely-Gated MoE 到 Switch Transformer、Expert Choice、DeepSeekMoE 与 OLMoE  
> **阅读方式：** 本文使用标准 Markdown + 内嵌 HTML，适合 Typora 阅读与导出 PDF。

<div style="background:#eef6ff;border-left:5px solid #377dce;padding:14px 18px;border-radius:6px;margin:16px 0;line-height:1.8">
  <strong>核心问题</strong><br>
  MoE 如何在增加<strong>总参数容量</strong>的同时，控制每个 Token 的<strong>激活计算量</strong>？答案不仅是稀疏激活，还包括专家粒度、路由选择、负载均衡与训练稳定性。
</div>

## 一、论文总览

| 首次公开 | 论文 | 团队 | 核心贡献 | 原文 |
|---|---|---|---|---|
| 2017-01 | **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** | Google | 稀疏门控、Top-k 专家路由 | [arXiv](https://arxiv.org/abs/1701.06538) |
| 2021-01 | **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** | Google | Top-1 Routing、低成本路由、稳定训练 | [JMLR](https://jmlr.org/papers/v23/21-0998.html) · [arXiv](https://arxiv.org/abs/2101.03961) |
| 2022-02 | **Mixture-of-Experts with Expert Choice Routing** | Google | 专家选择 Token、固定专家容量 | [arXiv](https://arxiv.org/abs/2202.09368) |
| 2024-01 | **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models** | DeepSeek-AI | Fine-grained Experts、Shared Experts | [ACL 2024](https://aclanthology.org/2024.acl-long.70/) · [arXiv](https://arxiv.org/abs/2401.06066) |
| 2024-09 | **OLMoE: Open Mixture-of-Experts Language Models** | Ai2 等 | 开放训练研究、专家粒度与路由分析 | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b224ace8963c9385ad5e2b5c9039b97-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2409.02060) |
| 2025-07 | **Mixture of Experts in Large Language Models** | Danyang Zhang 等 | MoE 系统综述 | [arXiv](https://arxiv.org/abs/2507.11181) |

<div style="display:flex;gap:10px;flex-wrap:wrap;margin:16px 0">
  <div style="flex:1;min-width:180px;padding:12px;background:#f3f4f6;border-radius:8px"><strong>2017–2021</strong><br><span style="color:#4b5563">稀疏激活与稳定训练</span></div>
  <div style="flex:1;min-width:180px;padding:12px;background:#eef6ff;border-radius:8px"><strong>2022</strong><br><span style="color:#4b5563">路由策略优化</span></div>
  <div style="flex:1;min-width:180px;padding:12px;background:#ecfdf5;border-radius:8px"><strong>2024–2025</strong><br><span style="color:#4b5563">专家结构细化与实证研究</span></div>
</div>

## 二、逐篇论文梳理

### 1. Sparsely-Gated MoE（2017）

**论文：** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*  
**作者：** Noam Shazeer 等  
**链接：** https://arxiv.org/abs/1701.06538

传统 Dense FFN 对每个 Token 都使用同一套参数。稀疏门控 MoE 将 FFN 扩展为多个 Expert，由 Router 为输入计算分数，只激活少量专家。

$$
y(x)=\sum_{i=1}^{N}G_i(x)E_i(x)
$$

其中 $E_i$ 是第 $i$ 个专家，$G_i(x)$ 是路由权重；稀疏门控使大部分权重为零，因此无须执行所有专家。

<div style="border:1px solid #d1d5db;border-radius:8px;padding:12px 16px;margin:12px 0">
  <strong>核心贡献：</strong> 模型可以拥有大量总参数，但每个输入只消耗少数专家的计算资源。<br>
  <span style="color:#6b7280">这是现代稀疏 MoE 研究的重要基础，而非 MoE 思想的最早起源。</span>
</div>

### 2. Switch Transformers（2021；JMLR 2022）

**论文：** *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*  
**作者：** William Fedus、Barret Zoph、Noam Shazeer  
**链接：** https://jmlr.org/papers/v23/21-0998.html

**关键改进：** 使用 **Top-1 Routing**，即每个 Token 只分配给一个专家。相比常见的 Top-2 稀疏路由，这可以降低专家计算和通信复杂度，但需要处理专家负载不均、容量限制及数值不稳定性。

**对应截图：**

1. 论文首页：标题 *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*。
2. **Section 3 / Figure 1 — Training Instabilities for Sparse Models**：左图训练 Loss 后期突然发散，右图 Loss 稳定下降。

<table style="width:100%;border-collapse:collapse;margin:14px 0">
<tr style="background:#f3f4f6"><th style="padding:10px;border:1px solid #ddd">不稳定训练</th><th style="padding:10px;border:1px solid #ddd">稳定训练</th></tr>
<tr><td style="padding:10px;border:1px solid #ddd">Loss 后期迅速上升甚至发散</td><td style="padding:10px;border:1px solid #ddd">Loss 随训练步数逐步下降</td></tr>
<tr><td style="padding:10px;border:1px solid #ddd">难以获得可用模型</td><td style="padding:10px;border:1px solid #ddd">可持续收敛</td></tr>
</table>

**阅读重点：** Top-1 Routing、专家负载均衡、Router 的选择性高精度计算、初始化与正则化。

### 3. Expert Choice Routing（2022）

**论文：** *Mixture-of-Experts with Expert Choice Routing*  
**链接：** https://arxiv.org/abs/2202.09368

它改变了路由方向：

| 机制 | 谁做选择 | 典型约束 |
|---|---|---|
| Token Choice | 每个 Token 选择专家 | 每 Token 选 Top-k 个 Expert |
| Expert Choice | 每个 Expert 选择 Token | 每 Expert 选择固定预算的 Token |

Expert Choice 能直接控制各专家接收的 Token 数量，但一个 Token 被多少专家选中可以变化，也可能没有被选中。

**与截图关系：** 后面 2025 年 MoE 综述的 **Figure 4** 用两幅示意图比较 Token Choice 与 Expert Choice；该图本身来自综述，而这篇 2022 年论文是 Expert Choice 的原始研究来源。

### 4. DeepSeekMoE（2024）

**论文：** *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*  
**作者：** Damai Dai 等  
**链接：** https://aclanthology.org/2024.acl-long.70/  
**官方代码：** https://github.com/deepseek-ai/DeepSeek-MoE

DeepSeekMoE 在常规 MoE 基础上提出两项核心策略：

1. **Fine-grained Expert Segmentation（细粒度专家划分）**：将较大的专家切分为更多较小的专家，使 Router 能构造更灵活的专家组合。
2. **Shared Expert Isolation（共享专家隔离）**：部分专家始终激活，专门承载跨 Token 的共性知识；其余 Routed Experts 通过路由激活，鼓励差异化专业分工。

**对应截图：** 三联图从 Conventional Top-2 Routing → Fine-grained Expert Segmentation → Shared Expert Isolation，说明在控制参数量和理论计算量的条件下改变专家结构。截图中标为 Figure 2；ACL 正式版本对应架构示意图的编号可能不同。

<div style="background:#ecfdf5;border-left:5px solid #16a34a;padding:14px 18px;border-radius:6px;margin:14px 0">
  <strong>记忆点：</strong> 一个大专家变成多个小专家，是为了获得更灵活的组合；共享专家负责常见模式，路由专家更专注于不同的模式。
</div>

### 5. OLMoE（2024；ICLR 2025）

**论文：** *OLMoE: Open Mixture-of-Experts Language Models*  
**链接：** https://arxiv.org/abs/2409.02060  
**正式版本：** https://proceedings.iclr.cc/paper_files/paper/2025/file/9b224ace8963c9385ad5e2b5c9039b97-Paper-Conference.pdf

**对应两张截图：**

- **预训练效果比较**：OLMoE-1B-7B 与 OLMo-1B、OLMo-7B 在 HellaSwag、MMLU、ARC-Challenge、PIQA、COPA、WinoGrande 上的曲线。注意：模型之间还存在训练数据及超参数差异，不能把全部差距归因于 MoE。
- **Expert Granularity（专家粒度）**：在总参数、激活参数及计算预算基本固定的比较条件下，改变 Expert 数量和 FFN 维度。截图对应早期版 Figure 5；ICLR 2025 正式版本对应 Figure 3。

| 专家数 N | 每 Token 激活 K | 每专家 FFN 中间维度 | 理论组合数 $\binom{N}{K}$ |
|---:|---:|---:|---:|
| 8 | 1 | 8,192 | 8 |
| 32 | 4 | 2,048 | 35,960 |
| 64 | 8 | 1,024 | 4,426,165,368 |

$$
\binom{N}{K}=\frac{N!}{K!(N-K)!}
$$

实验表明，更细粒度的专家通常改善训练 Loss 和下游任务表现；但从 32 增至 64 个专家时，性能提升出现**边际收益递减**。组合数表示数学上的可选子集数量，并非模型实际使用过全部组合。

### 6. Mixture of Experts in Large Language Models（2025 综述）

**论文：** *Mixture of Experts in Large Language Models*  
**作者：** Danyang Zhang、Junhao Song、Ziqian Bi 等  
**链接：** https://arxiv.org/abs/2507.11181

**对应两张截图：**

- **Figure 3**：Decoder-only Transformer 上的稀疏 MoE，Router 以 Top-k（示意为 $k=2$）选取 FFN Experts，再将专家结果按权重聚合。
- **Figure 4**：Token Choice 与 Expert Choice 的路由机制对照。

这是一篇**综述论文**，适合搭建知识框架；其中具体的原创算法贡献仍应回溯到对应原始论文。

## 三、按技术维度建立联系

| 优化方向 | 核心问题 | 代表论文 | 典型技术 |
|---|---|---|---|
| 稀疏计算 | 如何增加参数容量而控制单 Token 计算量？ | Sparsely-Gated MoE | Top-k Sparse Gating |
| 简化路由 | 如何降低路由与通信复杂度？ | Switch Transformers | Top-1 Routing |
| 负载均衡 | 如何避免少数 Expert 过载？ | Switch、Expert Choice | Load Balancing、Expert Choice |
| 专家专业化 | 如何避免 Expert 功能重复？ | DeepSeekMoE | Fine-grained Experts、Shared Experts |
| 粒度实证 | 小专家是否更有效？ | OLMoE | 固定计算预算的专家粒度对照 |
| 系统综述 | 如何连接各类 MoE 方法？ | Mixture of Experts in Large Language Models | 架构与路由方法分类 |

## 四、推荐阅读顺序

<div style="padding:14px 18px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;line-height:2">
  <strong>① 基础机制：</strong> Sparsely-Gated MoE<br>
  <strong>② 工程可训练性：</strong> Switch Transformers<br>
  <strong>③ 路由策略：</strong> Expert Choice Routing<br>
  <strong>④ 专家结构：</strong> DeepSeekMoE<br>
  <strong>⑤ 实验验证：</strong> OLMoE<br>
  <strong>⑥ 全局复盘：</strong> MoE Survey
</div>

## 五、结论

**MoE 的演进不只是“专家越来越多”。** 它同时沿着四条路线发展：

1. **激活更稀疏**：减少每个 Token 执行的参数量。
2. **路由更合理**：优化 Token 与 Expert 的匹配及负载分布。
3. **专家更细、更专业**：提高专家组合灵活性，并减少重复学习。
4. **训练更稳定、部署更高效**：控制通信、容量与数值稳定性问题。

---

<p style="text-align:center;color:#6b7280;font-size:12px">MoE 核心论文阅读笔记 · Markdown + 内嵌 HTML · 适配 Typora</p>
