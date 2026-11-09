# Transformer 位置编码：原始论文与技术演进

> 整理日期：2026-11-03  
> 阅读环境：Typora（标准 Markdown + 内嵌 HTML）

<div style="background:#eef6ff;border-left:5px solid #3178c6;padding:14px 18px;border-radius:8px;line-height:1.8;margin:16px 0">
<strong>核心问题：</strong> Self-Attention 本身不显式感知序列顺序。位置编码将 Token 的位置或相对距离引入模型。Sinusoidal、T5、RoPE、ALiBi 采用了不同的数学机制，不能简单理解为严格的逐代替换关系。
</div>

## 一、论文出处总览

| 技术 | 首次公开 | 原始论文 | 作者 / 团队 | 原文链接 |
|---|---|---|---|---|
| **Sinusoidal** | 2017-06 | *Attention Is All You Need* | Ashish Vaswani 等 / Google | [arXiv](https://arxiv.org/abs/1706.03762) · [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) |
| **Relative Position Representations** | 2018-03 | *Self-Attention with Relative Position Representations* | Peter Shaw、Jakob Uszkoreit、Ashish Vaswani | [arXiv](https://arxiv.org/abs/1803.02155) |
| **T5 Relative Position Bias** | 2019-10 | *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | Colin Raffel 等 / Google | [arXiv](https://arxiv.org/abs/1910.10683) · [JMLR PDF](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) |
| **RoPE** | 2021-04 | *RoFormer: Enhanced Transformer with Rotary Position Embedding* | Jianlin Su（苏剑林）等 | [arXiv](https://arxiv.org/abs/2104.09864) · [PDF](https://arxiv.org/pdf/2104.09864) |
| **ALiBi** | 2021-08 | *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* | Ofir Press、Noah A. Smith、Mike Lewis | [arXiv](https://arxiv.org/abs/2108.12409) · [ICLR PDF](https://openreview.net/pdf?id=R8sQPpGCv0) |

> **辨析：** 截图列出四种代表方案，但 T5 并不是相对位置表示的最早来源。因此本笔记补入 Shaw et al.（2018）作为重要历史节点。

## 二、逐项解读

### 1. Sinusoidal Positional Encoding（2017）

**论文：** *Attention Is All You Need*  
**发表：** NeurIPS 2017  
**位置：** 原论文第 3.5 节 Positional Encoding。

原始 Transformer 使用不同频率的正弦与余弦函数生成固定的绝对位置向量，并与 Token Embedding 相加。

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

<div style="border:1px solid #cbd5e1;background:#f8fafc;border-radius:8px;padding:12px 16px;margin:14px 0">
<strong>直观理解：</strong> 每个位置对应一组不同频率的波形数值。模型根据这些数值区分位置，不需要专门学习位置向量参数。
</div>

**关键特点：** 固定函数、绝对位置、无可训练位置参数。原论文还讨论了借助三角函数结构学习相对位移的潜力。

### 2. Relative Position Representations（2018）

**论文：** *Self-Attention with Relative Position Representations*  
**发表：** NAACL 2018  
**链接：** https://arxiv.org/abs/1803.02155

这一研究将 Token 之间的相对距离信息引入 Self-Attention，强调模型不只需要知道“第几个 Token”，也需要知道“两个 Token 相隔多远”。它与后来的 T5 Position Bias 设计相关，但具体公式和实现并不相同。

### 3. T5 Relative Position Bias（2019）

**论文：** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*  
**正式发表：** JMLR 2020  
**链接：** https://arxiv.org/abs/1910.10683

T5 将相对位置距离映射为有限数量的 Bucket，并为相应 Attention Head 学习位置偏置。简化记法：

$$
\operatorname{Score}_{h}(i,j)=\frac{Q_{h,i}K_{h,j}^{\mathsf T}}{\sqrt{d_k}}+b_h\big(\operatorname{bucket}(i-j)\big)
$$

这里 $b_h$ 是学习得到的位置 Bias，$\operatorname{bucket}$ 将相对距离归入分桶。较近的距离保留更细的区分，较远距离可以合并表示。

**关键特点：** 相对位置、可学习 Bias、分桶、直接修改 Attention Logits。

### 4. Rotary Position Embedding（RoPE，2021）

**论文：** *RoFormer: Enhanced Transformer with Rotary Position Embedding*  
**作者：** Jianlin Su 等  
**链接：** https://arxiv.org/abs/2104.09864

RoPE 对 Query 和 Key 的成对维度进行旋转，旋转角与位置有关。二维简化形式：

$$
R(m\theta)=\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}
$$

$$
q_m'=R(m\theta)q_m,\qquad k_n'=R(n\theta)k_n
$$

由于旋转矩阵的性质：

$$
(q_m')^{\mathsf T}k_n'=q_m^{\mathsf T}R((n-m)\theta)k_n
$$

**关键解释：** Query 和 Key 各自按绝对位置旋转，而二者点积可表现出相对位置差 $n-m$。实际模型按多个二维子空间及频率应用旋转。

<div style="background:#ecfdf5;border-left:5px solid #16a34a;border-radius:6px;padding:14px 18px;margin:14px 0">
<strong>最适合教学的记忆方式：</strong> Token 的位置不直接加在 Embedding 上，而是改变 Query / Key 的方向；相对旋转角影响 Attention 匹配结果。
</div>

### 5. ALiBi（2021）

**论文：** *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*  
**正式发表：** ICLR 2022  
**链接：** https://arxiv.org/abs/2108.12409

ALiBi 直接在 Attention Logits 中加入随距离线性变化的 Bias。以因果 Attention、$j\le i$ 为例：

$$
\operatorname{Score}_h(i,j)=\frac{Q_{h,i}K_{h,j}^{\mathsf T}}{\sqrt{d_k}}-m_h(i-j)
$$

其中 $m_h>0$ 是 Attention Head 的预设斜率。距离越远，惩罚通常越大；不同 Head 使用不同斜率。

**关键特点：** 不必添加位置 Embedding；使用线性距离偏置；强调超过训练序列长度时的外推能力。

## 三、四种方法关键比较

| 对比维度 | Sinusoidal | T5 PE | RoPE | ALiBi |
|---|---|---|---|---|
| 基本位置类型 | 绝对位置向量 | 相对距离分桶 Bias | 绝对旋转、相对点积 | 相对距离线性 Bias |
| 作用位置 | Token Embedding | Attention Logits | Query / Key | Attention Logits |
| 位置参数是否需要学习 | 否 | 是 | 标准版否 | 标准版否 |
| 是否显式体现相对距离 | 间接 | 是 | 在 Q/K 点积中体现 | 是 |
| 重点优势 | 简单、固定波形 | 距离抽象与学习偏置 | 兼容 Q/K、表达相对位置 | 简单、注重长度外推 |
| 长上下文注意事项 | 并不保证外推性能 | 远距离分桶分辨率有限 | 通常需验证外推或搭配扩展策略 | 仍需验证超长范围的实际效果 |

## 四、技术演进时间线

<div style="border-left:3px solid #60a5fa;padding-left:18px;margin:16px 4px;line-height:2">
<p><strong>2017｜Sinusoidal</strong><br><span style="color:#64748b">固定正弦余弦函数表示绝对位置</span></p>
<p><strong>2018｜Relative Position Representations</strong><br><span style="color:#64748b">在 Self-Attention 中直接考虑相对位置</span></p>
<p><strong>2019｜T5 Position Bias</strong><br><span style="color:#64748b">距离分桶与可学习的相对 Bias</span></p>
<p><strong>2021.04｜RoPE</strong><br><span style="color:#64748b">通过旋转 Q/K 将位置差融入点积</span></p>
<p><strong>2021.08｜ALiBi</strong><br><span style="color:#64748b">线性距离偏置，强调长度外推</span></p>
</div>

> **不要误读箭头：** 上述为时间排序，而非 Sinusoidal → T5 → RoPE → ALiBi 的严格继承链。RoPE 和 ALiBi 是不同的技术路线。

## 五、结合现代 Decoder-only LLM 的阅读建议

1. 从原始 Transformer 的 Sinusoidal 理解“为什么必须给 Self-Attention 提供位置信息”。
2. 对照 T5 Bias 理解：位置可以直接影响 Attention Logits，而不是必须加到 Embedding 上。
3. 重点推导 RoPE 中 $R(m)^\mathsf{T}R(n)=R(n-m)$ 的性质。
4. 用 ALiBi 理解“训练较短、推理较长”的长度外推目标。
5. 后续再学习 RoPE 长上下文扩展方法，例如 NTK-aware、Position Interpolation 与 YaRN；这些是额外技术，不应与原始 RoPE 混为一谈。

---

<p style="text-align:center;color:#64748b;font-size:12px">Transformer 位置编码论文阅读笔记 · Typora Markdown + HTML</p>
