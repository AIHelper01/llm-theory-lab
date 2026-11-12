# 旋转位置编码（RoPE）深度解读：从原始论文到 Qwen3.8 与 DeepSeek-V4

基础视频解读：[正弦位置编码：从序列编号到多频率坐标](正弦位置编码/正弦位置编码_从序列编号到多频率坐标.md) → [RoPE：从二维旋转到相对位置点积](RoPE基础/RoPE_从二维旋转到相对位置点积.md)。两篇分别附原视频截图、公式手算与复算代码。

> 技术专题笔记 · 2026-11-03  
> 格式：Markdown + 内嵌 HTML/SVG（适配 Typora，可离线显示，无外部图片依赖）

<div style="background:#edf6ff;border-left:5px solid #2563eb;padding:16px 20px;border-radius:8px;line-height:1.9;margin:16px 0">
<b>最重要的结论：</b>RoPE 使用 Token 的<strong>绝对位置</strong>旋转 Query 和 Key，却使 Attention 中的位置因素仅依赖<strong>相对位移</strong>。Qwen 将其扩展为多模态 MRoPE；DeepSeek 则围绕 KV 压缩、共享 KV 等注意力结构重新组织 RoPE。
</div>

## 一、论文出处与阅读路线

| 研究 | 时间 | 主要贡献 | 原始资料 |
|---|---|---|---|
| *RoFormer: Enhanced Transformer with Rotary Position Embedding* | 2021.04（预印本） | 提出 RoPE，使用旋转矩阵将相对位置引入 Attention | [arXiv](https://arxiv.org/abs/2104.09864) · [论文 PDF](https://arxiv.org/pdf/2104.09864) · [实现](https://github.com/ZhuiyiTechnology/roformer) |
| *DeepSeek-V3 Technical Report* | 2024.12 | MLA + Decoupled RoPE | [论文](https://arxiv.org/abs/2412.19437) · [官方代码](https://github.com/deepseek-ai/DeepSeek-V3) |
| *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* | 2026 | Partial RoPE 与 Inverse RoPE，配合共享 KV 和压缩注意力 | [技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) · [vLLM 官方解析](https://vllm.ai/blog/2026-04-24-deepseek-v4) |
| Qwen3.8-27B / Flash-Next 官方模型资料 | 2026 | MRoPE + Partial RoPE；可用 YaRN 扩展长上下文 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [Flash-Next 技术资料](https://github.com/QwenLM/Qwen3.8-Flash-Next) |
| *YaRN: Efficient Context Window Extension of Large Language Models* | 2023 | 面向 RoPE 的长上下文扩展方法 | [论文](https://arxiv.org/abs/2309.00071) |

**RoPE 原始论文作者：** Jianlin Su（苏剑林）、Yu Lu、Shengfeng Pan、Ahmed Murtadha、Bo Wen、Yunfeng Liu。预印本最早公开于 2021 年 4 月，后发表在 *Neurocomputing*。

## 二、为什么需要位置编码

Self-Attention 的基础计算是：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

不考虑额外的位置机制时，Self-Attention 本身并不为 Token 顺序设置固定偏好。因此“狗咬人”与“人咬狗”这类语序差异必须通过位置相关信息体现。

传统 Sinusoidal 方案把位置向量加入 Token Embedding；RoPE 则对 Query、Key 做旋转：

$$q_m'=R(m\theta)q_m,\qquad k_n'=R(n\theta)k_n$$

其中 $m,n$ 是绝对位置，$\theta$ 是相应二维旋转分量的频率。

## 三、从二维向量理解旋转

二维旋转矩阵：

$$
R(\alpha)=\begin{bmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{bmatrix}
$$

假设原始向量为 $x=(1,0)^T$，旋转 $30^\circ$ 后：

$$
R(30^\circ)x=\begin{bmatrix}0.866\\0.500\end{bmatrix}
$$

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:14px;background:#f8fafc;margin:16px 0">
<div style="font-weight:700;margin-bottom:7px">图 1 · 二维向量旋转 30°（HTML/SVG）</div>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 300" style="max-width:100%;display:block;margin:auto" role="img" aria-label="原向量水平向右，旋转30度后斜向右上">
<defs><marker id="rp-arrow-gray" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#64748b"/></marker><marker id="rp-arrow-teal" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#0d9488"/></marker></defs>
<line x1="80" y1="238" x2="555" y2="238" stroke="#94a3b8" stroke-width="2"/><line x1="125" y1="270" x2="125" y2="15" stroke="#94a3b8" stroke-width="2"/>
<circle cx="125" cy="238" r="5" fill="#334155"/>
<path d="M125 238 L365 238" stroke="#64748b" stroke-width="5" marker-end="url(#rp-arrow-gray)"/>
<path d="M125 238 L333 118" stroke="#0d9488" stroke-width="5" marker-end="url(#rp-arrow-teal)"/>
<path d="M198 238 A73 73 0 0 0 188 202" fill="none" stroke="#f59e0b" stroke-width="4"/>
<text x="198" y="218" fill="#b45309" font-size="17">30°</text><text x="371" y="232" fill="#475569" font-size="17">原向量 (1, 0)</text><text x="340" y="105" fill="#0f766e" font-size="17">旋转后 (0.866, 0.500)</text><text x="104" y="263" fill="#334155" font-size="15">O</text>
</svg>
<div style="font-size:13px;color:#64748b;text-align:center">旋转不改变向量长度，只改变方向；图形按几何意义绘制。</div>
</div>

### 数字例子：相同距离，相同位置旋转差

假设每个 Token 的位置递增时，旋转角递增 $30^\circ$（仅为教学简化，并非模型固定角度）。

| 位置 | 旋转角 | 旋转单位向量（近似） |
|---|---:|---|
| 0 | 0° | $(1,0)$ |
| 1 | 30° | $(0.866,0.500)$ |
| 2 | 60° | $(0.500,0.866)$ |
| 3 | 90° | $(0,1)$ |

比较位置 0 与 1：

$$ (1,0)\cdot(0.866,0.500)=0.866 $$

比较位置 1 与 2：

$$ (0.866,0.500)\cdot(0.500,0.866)\approx0.866 $$

<div style="display:flex;flex-wrap:wrap;gap:12px;margin:14px 0">
<div style="flex:1;min-width:210px;padding:14px;background:#eff6ff;border-radius:8px"><strong>位置 0 → 1</strong><br>相对距离：1<br>点积：<strong>0.866</strong></div>
<div style="flex:1;min-width:210px;padding:14px;background:#ecfdf5;border-radius:8px"><strong>位置 1 → 2</strong><br>相对距离：1<br>点积：<strong>0.866</strong></div>
</div>

**注意：** 上述两次点积使用了相同的原始单位向量。真实 Attention 的分数仍取决于不同 Token 的 $Q,K$ 内容，绝非只由距离决定。

### RoPE 最关键的数学等式

$$
\begin{aligned}
(q'_m)^Tk'_n
&=(R(m\theta)q_m)^TR(n\theta)k_n\\
&=q_m^TR(-m\theta)R(n\theta)k_n\\
&=\boxed{q_m^TR((n-m)\theta)k_n}
\end{aligned}
$$

**结论：** 两个向量分别按绝对位置旋转，但点积中的位置变换只取决于 $n-m$，也就是相对位置。

## 四、真实模型中如何处理高维向量

假设一个 Attention Head 的维度为 128，可以分为 64 对二维分量，不同分量使用不同频率：

$$\theta_i=\mathrm{base}^{-2i/d_{rotary}}$$

第 $m$ 个 Token 的第 $i$ 对分量旋转 $m\theta_i$。快频率和慢频率共同表示不同尺度的位置变化。常规 RoPE 应用于 Query 和 Key，而 Value 通常不旋转。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0;background:#fff">
<strong>图 2 · RoPE 插入 Self-Attention 的位置（HTML）</strong>
<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:center;margin-top:14px;text-align:center">
<div style="padding:12px;border-radius:7px;background:#eff6ff;border:1px solid #93c5fd">Hidden States</div><span style="font-size:22px;color:#64748b">→</span>
<div style="padding:12px;border-radius:7px;background:#f1f5f9;border:1px solid #cbd5e1">Q / K 投影</div><span style="font-size:22px;color:#64748b">→</span>
<div style="padding:12px;border-radius:7px;background:#ccfbf1;border:1px solid #2dd4bf"><b>RoPE 旋转 Q、K</b></div><span style="font-size:22px;color:#64748b">→</span>
<div style="padding:12px;border-radius:7px;background:#fef3c7;border:1px solid #fbbf24">Attention Score</div><span style="font-size:22px;color:#64748b">→</span>
<div style="padding:12px;border-radius:7px;background:#f1f5f9;border:1px solid #cbd5e1">Softmax × V</div>
</div>
</div>

## 五、Qwen3.8：MRoPE、Partial RoPE 与 YaRN

### 5.1 从文本 1D RoPE 到多模态 MRoPE

文本通常只有位置 $p$；图像和视频同时具有时间、行、列坐标，可用 $(t,h,w)$ 表示。MRoPE 将旋转位置维度分配到不同的坐标轴。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0;background:#f8fafc">
<strong>图 3 · MRoPE 位置维度分配（HTML）</strong>
<div style="display:flex;gap:6px;margin-top:12px;flex-wrap:wrap;text-align:center">
<div style="flex:11;min-width:100px;background:#dbeafe;border:1px solid #60a5fa;border-radius:7px;padding:14px 7px"><b>Temporal</b><br>11 对 = 22 维</div>
<div style="flex:11;min-width:100px;background:#ccfbf1;border:1px solid #2dd4bf;border-radius:7px;padding:14px 7px"><b>Height</b><br>11 对 = 22 维</div>
<div style="flex:10;min-width:100px;background:#f3e8ff;border:1px solid #c084fc;border-radius:7px;padding:14px 7px"><b>Width</b><br>10 对 = 20 维</div>
</div>
<p style="font-size:13px;color:#64748b;margin-bottom:0">示意分组为 32 对、共 64 个旋转维度。实际 <code>mrope_interleaved=true</code> 采用交错编排，不能把上图当作内存中连续排列。</p>
</div>

### 5.2 Qwen3.8 的关键配置

官方给出的长上下文配置中包含：

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25
}
```

假设 Q/K Head 为 256 维，旋转部分：

$$256\times0.25=64\text{ 维}$$

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0;background:#fff">
<strong>图 4 · Qwen Partial RoPE：64 / 256 维（HTML）</strong>
<div style="display:flex;height:42px;margin-top:12px;font-size:13px;color:white;text-align:center;line-height:42px"><div style="flex:1;background:#0d9488;border-radius:6px 0 0 6px">64 维 RoPE</div><div style="flex:3;background:#64748b;border-radius:0 6px 6px 0">192 维不旋转</div></div>
<div style="font-size:13px;color:#64748b;margin-top:8px">25% 的维度使用 RoPE，75% 不执行旋转。非旋转部分仍参与 Attention 的内容计算。</div>
</div>

`rope_theta` 是频率基数，不是旋转角度。`mrope_section` 指定多模态旋转对的划分，而 `partial_rotary_factor` 指定参与旋转的维度比例。

### 5.3 YaRN 扩展到约 1M 上下文

Qwen3.8 官方模型卡说明原生上下文为 262,144 Tokens，并给出 YaRN 扩展配置：

```json
{
  "rope_type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

完整覆盖配置还应保留上述 `mrope_*`、`rope_theta` 和 `partial_rotary_factor` 字段。由于多数框架实现静态 YaRN，固定缩放也可能影响短文本性能，官方建议只在长上下文有需要时启用，并按目标长度选择缩放因子。

**Flash-Next 特别说明：** Qwen3.8-Flash-Next 具有 Gated DeltaNet + Qwen Sparse Attention（QSA）混合架构。MRoPE 是注意力分支所使用的位置处理机制之一，不能把整个网络概括为每层都是标准 Full Attention + RoPE。

## 六、DeepSeek-V3：MLA + Decoupled RoPE

### 6.1 为什么 MLA 需要解耦位置？

MLA（Multi-head Latent Attention）将 Key 和 Value 压缩为更小的潜在向量，以减少 KV Cache。旋转依赖每个 Token 的位置，通常不能与任意线性投影矩阵无条件交换，这会影响矩阵吸收和缓存压缩。

DeepSeek 将 Query 和 Key 的内容部分与位置部分拆开：

- **NoPE**：不旋转的内容分量，适合与潜在表示、投影吸收结合。
- **RoPE**：单独的位置分量，专门负责旋转位置表示。

官方代码明确区分 `q_nope`、`q_pe`、`k_pe`，并独立缓存潜在 KV 和位置分量。概念上注意力分数可看作：

$$\operatorname{Score}=\operatorname{Score}_{NoPE}+\operatorname{Score}_{RoPE}$$

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0;background:#f8fafc">
<strong>图 5 · DeepSeek-V3 MLA 与 Decoupled RoPE（HTML）</strong>
<div style="text-align:center;margin:12px auto;padding:10px;background:#e0f2fe;border:1px solid #7dd3fc;border-radius:7px;max-width:300px"><b>Token Hidden States</b></div>
<div style="display:flex;justify-content:center;gap:12px;margin:6px 0;color:#64748b">↙ <span style="padding:0 35px">↓</span> ↘</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;text-align:center">
<div style="flex:1;min-width:145px;background:#f1f5f9;border:1px solid #94a3b8;border-radius:8px;padding:12px"><b>Q NoPE</b><br><small>内容匹配</small></div>
<div style="flex:1;min-width:145px;background:#ccfbf1;border:1px solid #5eead4;border-radius:8px;padding:12px"><b>Q/K RoPE</b><br><small>独立位置分量</small></div>
<div style="flex:1;min-width:145px;background:#dbeafe;border:1px solid #93c5fd;border-radius:8px;padding:12px"><b>KV Latent</b><br><small>压缩存储</small></div>
</div>
<div style="text-align:center;margin:9px;color:#64748b">↓ 组合内容与位置贡献 ↓</div>
<div style="text-align:center;padding:12px;background:#fef3c7;border:1px solid #fcd34d;border-radius:7px;max-width:350px;margin:auto"><b>Attention Score</b></div>
</div>

**关键理解：** MLA 优化 KV Cache；Decoupled RoPE 让位置分量可以独立保留，而不破坏内容分量的压缩结构。

## 七、DeepSeek-V4：Partial RoPE + Inverse RoPE

DeepSeek-V4 采用共享 Key/Value 表示与压缩注意力。对共享 KV 的旋转部分，若 Attention 输出包含 $R(j)k_j$，其输出会直接携带历史绝对位置 $j$ 的旋转因素。

DeepSeek-V4 对输出的相应 RoPE 维度应用当前位置 $i$ 的逆向旋转 $R(-i)$：

$$R(-i)R(j)k_j=\boxed{R(j-i)k_j}$$

这样位置因素重新只依赖相对距离。这不是将全部 Attention 输出整体逆转，而是针对 RoPE 子空间进行的操作。

### 数值示例

假设每个 Token 相差 $30^\circ$：

- Query 位置 $i=3$：逆旋转 $-90^\circ$；
- 历史 KV 位置 $j=5$：原旋转 $150^\circ$；
- 合并后：$-90^\circ+150^\circ=60^\circ$，只对应相对距离 $j-i=2$。

<div style="border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin:16px 0;background:#f8fafc">
<strong>图 6 · DeepSeek-V4 Inverse RoPE（HTML）</strong>
<div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;text-align:center;margin:12px 0">
<div style="background:#dbeafe;border:1px solid #93c5fd;border-radius:7px;padding:13px"><b>历史 KV</b><br>R(150°)</div><b style="color:#64748b">×</b>
<div style="background:#fef3c7;border:1px solid #fbbf24;border-radius:7px;padding:13px"><b>输出逆旋转</b><br>R(-90°)</div><b style="color:#64748b">=</b>
<div style="background:#ccfbf1;border:1px solid #2dd4bf;border-radius:7px;padding:13px"><b>相对旋转</b><br>R(60°)</div>
</div>
<div style="font-size:13px;color:#64748b;text-align:center">示例中 60° 对应两个 Token 的距离；真实模型有多组不同旋转频率。</div>
</div>

### V4 的位置参数

| 参数 | DeepSeek-V4-Flash / Pro 的对应配置 |
|---|---:|
| Head Dimension | 512 |
| RoPE Dimension | 64 |
| 比例 | 12.5% |
| `rope_theta` | 10,000 |
| YaRN Factor | 16 |
| 原始序列长度 | 65,536 |
| 最大配置位置 | 1,048,576 |

这套设计同时需要区分其压缩注意力（如 CSA/HCA）与 Sliding Window 分支；RoPE 是位置机制，并不负责全部的长上下文效率收益。

## 八、Qwen 与 DeepSeek 的对照

| 维度 | Qwen3.8 | DeepSeek-V3 | DeepSeek-V4 |
|---|---|---|---|
| 基础方法 | RoPE | RoPE | RoPE |
| 主要变体 | MRoPE + Partial RoPE | Decoupled RoPE | Partial + Inverse RoPE |
| 主要目标 | 多模态时间/空间位置 | 配合 MLA、压缩 KV Cache | 配合共享 KV、压缩注意力 |
| 位置子空间 | 多维交错旋转 | 内容/位置解耦 | 部分旋转、输出逆旋转 |
| 长上下文 | 可选 YaRN | YaRN | YaRN |

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0">
<div style="flex:1;min-width:230px;padding:15px;background:#eff6ff;border-left:4px solid #3b82f6;border-radius:7px"><strong>Qwen 路线</strong><p>如何把 RoPE 从文本一维位置拓展到视觉与视频的多维坐标，同时合理限制旋转维度？</p></div>
<div style="flex:1;min-width:230px;padding:15px;background:#ecfdf5;border-left:4px solid #10b981;border-radius:7px"><strong>DeepSeek 路线</strong><p>如何让 RoPE 与 KV 压缩、共享 KV 和高效注意力运算保持兼容？</p></div>
</div>

## 九、进一步阅读：RoPE 长上下文扩展

| 技术 | 主要思想 | 资料 |
|---|---|---|
| Position Interpolation | 把较长位置映射到已训练的位置范围 | [论文](https://arxiv.org/abs/2306.15595) |
| NTK-aware Scaling | 调整 RoPE 频率分布 | 需结合具体实现核对 |
| YaRN | 对 RoPE 进行频率与注意力缩放等长上下文优化 | [论文](https://arxiv.org/abs/2309.00071) |

**RoPE 不是 YaRN：** 前者是基础位置编码机制，后者是围绕 RoPE 的长度扩展方法。长上下文效果仍需在目标任务上验证。

## 十、最终记忆框架

1. **RoPE**：按绝对位置旋转 Q/K；点积位置因子只依赖相对距离。
2. **Partial RoPE**：只旋转 Head 的一部分维度。
3. **MRoPE（Qwen）**：将旋转维度分配给时间、高度、宽度等坐标。
4. **Decoupled RoPE（DeepSeek-V3）**：内容与位置分量解耦，方便 MLA KV 压缩。
5. **Inverse RoPE（DeepSeek-V4）**：共享 KV 的输出相应分量做逆旋转，恢复相对位置性质。
6. **YaRN**：帮助 RoPE 模型延长可用上下文。

---

<p style="font-size:12px;color:#64748b;text-align:center">RoPE 原理与 Qwen / DeepSeek 架构解读 · HTML/SVG 图均直接嵌入 Markdown · 无需联网加载图片</p>
