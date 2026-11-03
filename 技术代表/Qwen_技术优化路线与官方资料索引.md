# Qwen 技术优化路线与官方资料索引

基于原始 Transformer 架构的演进解读 · 资料检索截至 2026 年 11 月 3 日

> 阅读环境：Typora。正文使用 Markdown，公式使用 LaTeX。本文根据公开资料整理技术路线，不代表官方内部研发规划。原始资料链接保留在相应章节；官方发布版本、产品分支与具体型号应分别对待。

**基于我们刚才学过的 Transformer，Qwen 的优化路线可以理解为：先改进基本零件，再降低每个 Token 的计算成本，随后重构长上下文的信息存储和检索方式。**

已检索的资料包括官方博客、技术报告、官方模型卡和代码仓库。下面先建立一份“技术路线地图＋原始资料索引”。

为了方便以后对比其他厂家，我们始终围绕五个问题：

1. **改了原始 Transformer 的哪个位置？**
2. **解决什么问题？**
3. **数学计算发生什么变化？**
4. **收益和代价是什么？**
5. **哪些版本实际采用，官方证据在哪里？**

## 一、先把 Qwen 的改动放回基本架构

你已经理解的基本结构是：

**输入表示 → Attention → 残差与归一化 → FFN → 残差与归一化 → 输出预测。**

Qwen 的主要优化，可以对应到以下位置：

| 基本模块 | Qwen 的主要优化方向 | 要解决的问题 |
|---|---|---|
| 输入 Embedding | 常规 Token Embedding；新架构引入 N-gram Embedding | 如何以较低计算成本扩大模型容量 |
| 位置编码 | RoPE；长上下文扩展；多模态位置编码 | 如何表示顺序、空间和时间 |
| Attention | GQA、QK-Norm、Gated Attention、Gated DeltaNet、QSA | 如何兼顾信息交互、稳定性和长上下文效率 |
| FFN | SwiGLU；Dense FFN → MoE | 如何扩大参数容量，同时控制每个 Token 的计算量 |
| 残差与归一化 | Pre-RMSNorm、稳定性改进、Gated Residual | 如何让深层网络稳定训练、更有效地传递信息 |
| 输出预测 | Next-token prediction；MTP 辅助模块 | 如何改善训练效果，并加速自回归生成 |
| 训练流程 | 数据扩展、SFT、RL、蒸馏、Agent 环境训练 | 如何把网络容量变成实际能力 |

这些变化分布在不同版本和分支中，**并不是每个 Qwen 模型都同时具备全部技术**。Qwen3、Qwen3-Next 和 Flash-Next 的官方资料尤其适合用于对照。

资料依据：[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)、[Qwen3-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[Flash-Next 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)。

## 二、按时间看：几个关键转折

| 阶段 | 代表版本 | 最值得研究的变化 |
|---|---|---|
| 建立基础模型 | Qwen，2023 | 理解早期 Qwen 的模型设计、预训练和对齐流程 |
| 探索稀疏计算 | Qwen1.5-MoE，2024 | 用多个专家 FFN 扩大容量，每次只调用一部分 |
| 完善现代 Decoder | Qwen2、Qwen2.5，2024 | GQA、SwiGLU、RoPE、Pre-RMSNorm；持续改进训练 |
| 扩展上下文 | Qwen2.5-1M，2025 | 长上下文训练与稀疏注意力推理相结合 |
| 强化推理 | QwQ、Qwen3，2025 | 推理强化学习、思考模式、推理预算与蒸馏 |
| 重构效率骨架 | Qwen3-Next，2025 | Gated DeltaNet＋Gated Attention、高稀疏 MoE、MTP |
| 扩展原生多模态与 Agent | Qwen3.5 及后续系列，2026 | 在混合架构基础上继续扩展多模态和任务执行能力 |
| 再次探索新骨架 | Qwen3.8-Flash-Next，2026 | QSA、Gated Residual、N-gram Embedding、Muon |

这张表是阅读顺序。实际研发有多条并行分支，不能简单理解成“所有模型依次替换上一代”。例如，Qwen2.5 的开源通用模型主要是 Dense，而官方 API 产品中包含 MoE；Qwen3 同时发布 Dense 和 MoE。

资料依据：[Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)、[Qwen3 发布博客](https://qwenlm.github.io/blog/qwen3/)、[Flash-Next 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)。

## 三、第一条路线：改进 Attention，但先保留熟悉的计算框架

### 1. GQA：多个 Query 头共享 K、V

原始多头注意力中，各头有各自的 Q、K、V。

GQA 保留较多 Query 头，让一组 Query 头共享 K、V，主要减少 KV Cache 和相关访存成本。

例如 Qwen3-8B 的官方配置是：

| 项目 | 数量 |
|---|---:|
| Query Heads | 32 |
| KV Heads | 8 |
| 每组共享关系 | 4 个 Query 头对应 1 个 KV 头 |

在其他条件相同时，这种配置的 KV 存储量约为具有 32 个 KV 头的 MHA 的四分之一。**减少的是 K、V 的头数，不是把全部 Attention 计算都减少到四分之一。**

资料依据：[Qwen3 技术报告，第 2 节 Architecture](https://arxiv.org/html/2505.09388v1)。

### 2. QK-Norm：控制 Q、K 的数值尺度

Qwen3 在延续 Qwen2.5 基本骨架的同时，去掉 QKV bias，并增加 QK-Norm，以改善训练稳定性。

对应我们学过的公式：

$$
\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

可以先理解为：**在 Q、K 进入匹配计算前，增加数值尺度的控制。**

它和 GQA 的目标不同：GQA 主要关注效率，QK-Norm 主要关注稳定性。

资料依据：[Qwen3 技术报告，第 2 节 Architecture](https://arxiv.org/html/2505.09388v1)。

### 3. Gated Attention：融合之后，再控制哪些特征通过

普通 Attention 先得到：

$$
z=\operatorname{Attention}(Q,K,V)
$$

门控机制可以用下面的教学表达式理解：

$$
z'=z\odot \sigma(xW_g)
$$

也就是：Attention 已经收集了信息，再根据当前输入生成一组门值，控制输出特征。

相关论文研究了不同门控位置，并报告了稳定性、注意力汇聚现象及长上下文外推方面的改进。**这里的特征门控，与 MoE 中选择专家的路由门控，是两种不同机制。**

原始资料：

- [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
- [作者官方实现](https://github.com/qiuzh20/gated_attention)

## 四、第二条路线：改进 FFN，从“加工特征”到“选择专家加工”

### 1. ReLU FFN → SwiGLU

我们前面学过：

$$
FFN(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2
$$

SwiGLU 的核心可以简化写为：

$$
FFN(x)=\left[\operatorname{SiLU}(xW_{\text{gate}})\odot(xW_{\text{up}})\right]W_{\text{down}}
$$

这里有两条分支：一条产生特征，另一条调节特征。

**它仍然是在每个 Token 内部进行非线性特征变换，符合我们此前对 FFN 的理解。**Qwen2.5、Qwen3 的技术报告明确列出了 SwiGLU。

资料依据：[Qwen2.5 技术报告](https://arxiv.org/html/2412.15115v2)、[Qwen3 技术报告](https://arxiv.org/html/2505.09388v1)。

### 2. Dense FFN → MoE

Dense FFN：每个 Token 使用同一套完整 FFN。

MoE：有多套专家 FFN，路由器为当前 Token 选择部分专家：

$$
y=\sum_{i\in TopK}p_i(x)\,E_i(x)
$$

有共享专家时，还会加入共享专家输出。这个公式是帮助理解的简化形式。

Qwen 的专家设计并不是一直不变：

| 代表模型 | 路由专家配置 | 共享专家 |
|---|---|---|
| Qwen3 MoE | 128 个，每 Token 激活 8 个 | 无 |
| Qwen3-Next-80B-A3B | 512 个，每 Token 激活 10 个 | 1 个 |

这意味着路线并非“专家越多越好”，而是同时调整专家规模、粒度、激活数量和负载均衡。

要特别记住：

**激活参数较少，主要降低每 Token 的计算需求；全部专家权重仍然需要存储，分布式部署还涉及通信和负载均衡。**

原始资料：

- [Qwen1.5-MoE 官方博客](https://qwenlm.github.io/blog/qwen-moe/)
- [Qwen2 技术报告](https://arxiv.org/abs/2407.10671)
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
- [Qwen3-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

## 五、第三条路线：长上下文，从“保存历史”转向“压缩＋检索”

这一条是 Qwen 后续架构变化最大的部分。

### 1. Qwen2.5-1M：模型训练和推理系统共同优化

它不只是修改一个最大长度参数。

官方资料同时讨论了长上下文训练与推理框架，并引入稀疏注意力方法来降低超长输入处理成本。官方报告的 3～7 倍加速针对其百万 Token 输入实验设置，不能直接套用到其他模型、硬件和请求长度。

- [官方技术博客](https://qwenlm.github.io/blog/qwen2.5-1m/)
- [Qwen2.5-1M 技术报告](https://arxiv.org/abs/2501.15383)

### 2. Qwen3-Next：Gated DeltaNet＋Gated Attention

普通全注意力会维护随上下文增长的历史 K、V。

Gated DeltaNet 则持续更新一个状态，用有限大小的状态表示历史信息；它不再逐项保存同样形式的全部历史。

但压缩历史与精确读取历史之间存在取舍，因此采用混合设计：

| 一个重复单元 | 信息处理方式 |
|---|---|
| 第 1 层 | Gated DeltaNet → MoE |
| 第 2 层 | Gated DeltaNet → MoE |
| 第 3 层 | Gated DeltaNet → MoE |
| 第 4 层 | Gated Attention → MoE |

Qwen3-Next-80B-A3B 将该单元重复 12 次，共 48 层。可以直观理解为：

**多数层高效维护历史状态，部分层保留对历史的直接注意力访问。**这是对官方结构的教学解释。

原始资料：

- [Qwen3-Next 官方技术博客](https://qwen.ai/blog?id=qwen3-next)
- [官方模型卡与结构参数](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

最后一篇是相关基础方法论文，**不是 Qwen3-Next 的整模型技术报告**。

### 3. Flash-Next：进一步把全局检索变成稀疏检索

Qwen3.8-Flash-Next 在混合架构中引入 Qwen Sparse Attention，简称 QSA。

它以轻量索引器选择重要的历史微块，再进行稀疏注意力计算。目标不仅是减少正式 Attention 的成本，还包括控制“先找出该关注哪里”这一步的索引成本。

所以可以把这条路线概括为：

**完整历史访问 → 历史状态压缩与全局访问交替 → 历史状态压缩与稀疏检索交替。**

资料依据：[阿里云官方技术博客](https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towardsultimate-cost-efficiency_603501)。

## 六、第四条路线：连残差、Embedding 和训练优化器也开始改

Flash-Next 值得单独研究，因为它超出了“Attention＋MoE”的常见讲解范围：

| 改动 | 对应基本架构 | 核心思路 |
|---|---|---|
| Gated Residual | 残差连接 | 将残差流扩展为 4 个分支，通过动态门控控制读取与写入 |
| N-gram Embedding | 输入表示及额外查表容量 | 利用局部 Token 组合查表，以较少额外计算增加容量 |
| Muon 与 AdamW 分工 | 训练优化 | 根据参数类别选择优化方法，并调整训练配方 |
| MTP | 辅助预测模块 | 多步预测训练，并用于推测解码加速 |

这些是官方披露的不同方向，不能全部算成 Attention 优化。N-gram 表也不是外部知识库检索：它属于模型学习到的参数容量。

优先收藏：

- [Qwen3.8-Flash-Next 官方博客](https://qwen.ai/blog?id=qwen3.8-flash-next)
- [官方技术报告 PDF：On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- [官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [阿里云官方技术博客版本](https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towardsultimate-cost-efficiency_603501)

目前已核实报告入口存在；这里对它的概括依据官方博客和模型卡，尚未逐页解读 PDF。

## 七、第五条路线：结构相近，能力也可以通过训练大幅改变

这是比较厂家路线时必须单列的一条。

Qwen2.5 的重点包含数据扩展和多阶段后训练；QwQ 着重探索推理强化学习；Qwen3 则明确介绍了长思维链冷启动、推理 RL、思考模式融合、通用 RL，以及强到弱蒸馏。

| 优化方向 | 改变什么 | 是否必然需要改变网络骨架 |
|---|---|---|
| 更多、更高质量的数据 | 模型学到的知识和模式 | 否 |
| SFT | 指令遵循与回答形式 | 否 |
| 推理 RL | 多步推理策略 | 否 |
| Agent 环境训练 | 工具使用、执行与纠错 | 否 |
| 蒸馏 | 把强模型能力迁移给小模型 | 否 |
| Thinking Budget | 推理时投入多少计算 | 否 |

因此，看到某代模型推理或编程能力提升，不能直接推出“它换了新的 Attention”。

相关官方资料：

- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)
- [QwQ-32B 官方博客](https://qwenlm.github.io/blog/qwq-32b/)
- [Qwen3 发布博客](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)
- [Qwen3-Coder-Next 技术报告](https://arxiv.org/abs/2603.00729)
- [Qwen3.7 官方博客](https://qwen.ai/blog?id=qwen3.7&locale=en)

## 八、多模态作为并行分支研究

这条路线关注的是：**图像、视频、音频如何进入这个基本骨架，以及模型如何生成语音。**

| 分支 | 官方资料中值得研究的重点 | 论文或官方博客 |
|---|---|---|
| Qwen2.5-VL | 动态分辨率、视觉编码器的 Window Attention、视频时间信息 | [技术报告](https://arxiv.org/abs/2502.13923) |
| Qwen3-VL | 文本、图片、视频交错的长上下文理解 | [技术报告](https://arxiv.org/abs/2511.21631) |
| Qwen2.5-Omni | Thinker–Talker 与多模态输入输出 | [技术报告](https://arxiv.org/abs/2503.20215) |
| Qwen3-Omni | Thinker–Talker MoE、流式语音生成及低延迟设计 | [技术报告](https://arxiv.org/abs/2509.17765) |
| Qwen3.5 | 原生多模态与 Agent 方向 | [官方博客](https://qwen.ai/blog?id=qwen3.5) |

其中 VL 和 Omni 需要分开：**看懂图像、视频，与实时理解并生成语音，涉及不同的模块和训练目标。**

## 九、建议接下来的讲解顺序

为了延续“向量＋矩阵＋具体数字”的方式，建议依次展开：

1. **原始 Transformer → Qwen2.5 基础层**：先把 RoPE、GQA、SwiGLU、RMSNorm 放回结构图。
2. **GQA 数值示例**：看 4 个 Query 头如何共享 2 组 K、V。
3. **Dense FFN → MoE 数值示例**：用 4 个专家、Top-2 路由实际算一次。
4. **Qwen3-Next**：对比“保存历史 K、V”和“更新历史状态”。
5. **Flash-Next**：再讲稀疏检索、门控残差和 N-gram 查表。

**第一步最值得先讲的是“原始 Transformer 到 Qwen2.5：一层网络究竟换了哪些零件”。**这能把基础架构与后续所有优化连接起来。

## 附：早期基础报告与补充官方入口

- [Qwen Technical Report（2023）](https://arxiv.org/abs/2309.16609)
- [Qwen2 Technical Report（2024）](https://arxiv.org/abs/2407.10671)
- [Qwen2.5 官方发布博客](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [Qwen2.5-Omni 官方博客](https://qwenlm.github.io/blog/qwen2.5-omni/)
- [Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)
- [Qwen 官方文档](https://qwen.readthedocs.io/en/stable/)

> 资料使用说明：本文件保存的是技术路线梳理和原始资料索引，不包含论文全文。文中的示意公式与概念解释用于教学；具体实现以相应版本的报告、模型配置和代码为准。
