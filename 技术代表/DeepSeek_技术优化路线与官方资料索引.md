# DeepSeek 技术优化路线与官方资料索引

基于原始 Transformer 架构的演进解读 · 截至 2026 年 11 月 3 日 · Typora 阅读版

> 本文延续 Qwen 篇的方式：先找出“改了 Transformer 的哪个模块”，再解释解决的问题、计算变化、收益与代价，最后列出官方论文、公告及代码。技术路线是对公开资料的教学整理，并非官方内部规划；示意公式不等于完整实现。

## 一、先确认最新版本：已经不能只讲 V3 和 R1

截至核查日，官方最新公开更新为 **DeepSeek-V4.1-Flash-1017，更新日期为 2026 年 10 月 17 日**。

| 要区分的对象 | 已核实信息 | 官方依据 |
|---|---|---|
| 最新模型更新 | DeepSeek-V4.1-Flash-1017 | [更新日志](https://api-docs.deepseek.com/updates/) |
| 对应架构家族 | DeepSeek-V4.1-Flash，2026-09-10 发布 | [发布公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) |
| 对应整模型技术报告 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression，2026-09-17 | [论文](https://arxiv.org/abs/2609.19969) |
| 最新权重入口 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| 当前 API 模型名 | deepseek-flash | [更新日志](https://api-docs.deepseek.com/updates/) |

1017 官方模型卡将它称为 V4.1-Flash 的小版本更新，重点包括推理、编程、Agent、视觉能力与任务执行效率。**不能因为出现新日期后缀，就推断模型再次更换了架构。**

另外，V4.1 首发公告中曾出现 V4-Pro 服务迁移安排，而当前 API 更新日志明确保留 V4-Pro 服务。本文涉及当前服务状态时采用更新后的 API 文档，不沿用早期公告的旧安排，也不把“提到未来型号”当作“已经发布”。

## 二、把技术路线放回基本 Transformer

我们此前理解的基本结构是：

**输入表示 → Attention → 残差与归一化 → FFN → 残差与归一化 → 输出预测。**

DeepSeek 的研究可以分为几条不同的轴：

| 基本模块或系统位置 | 代表技术 | 主要问题 |
|---|---|---|
| Attention 的 KV 表示 | MLA | 每个历史 Token 的缓存能否更小？ |
| Attention 的访问范围 | DSA、CSA、HCA、CSA2 | 能否只读取必要历史，或压缩历史条目？ |
| Attention 的跨层状态 | CSA2 的 KV 与索引复用 | 不同层能否减少重复缓存与重复检索？ |
| FFN | DeepSeekMoE | 能否扩大容量而不成比例增加每 Token 计算？ |
| 专家路由 | Auxiliary-Loss-Free Balancing | 专家如何既专业又不过度拥挤？ |
| 残差连接 | mHC、Single-Pass mHC | 如何改善跨层信息流，同时控制稳定性和访存？ |
| 参数化记忆 | Engram | 哪些局部模式适合查表，而不必反复计算？ |
| 预测与生成 | MTP、DSpark | 如何提供训练信号，或减少生成时的串行开销？ |
| 整体计算组织 | Causal Encoder–Decoder（CED） | 输入处理和输出生成能否使用不同计算预算？ |
| 后训练 | GRPO、R1、Agent RL | 如何训练推理和工具执行策略？ |
| 底层系统 | FP8、DualPipe、DeepEP、FlashMLA、DeepGEMM、DualPath | 算法收益如何真正变成吞吐与成本收益？ |

这些技术并非全部存在于每个型号中。尤其不能将 V2、V3 的 MLA 结构直接套在 V4、V4.1 上，也不能把独立研究论文自动视作某一旗舰模型的配置。

## 三、时间线：几次关键转折

| 阶段 | 代表模型或研究 | 主要变化 | 优先资料 |
|---|---|---|---|
| 2023—2024 初 | DeepSeek LLM | 建立通用语言模型与扩展规律研究 | [基础报告](https://arxiv.org/abs/2401.02954) |
| 2024-01 | DeepSeekMoE | 细粒度专家与共享专家 | [方法论文](https://arxiv.org/abs/2401.06066) |
| 2024-02 | DeepSeekMath | 数学数据、GRPO | [报告](https://arxiv.org/abs/2402.03300) |
| 2024-05 | DeepSeek-V2 | MLA＋DeepSeekMoE | [报告](https://arxiv.org/abs/2405.04434) |
| 2024-12 | DeepSeek-V3 | 路由均衡、MTP、FP8 与系统协同 | [报告](https://arxiv.org/abs/2412.19437) |
| 2025-01 | DeepSeek-R1 | 推理强化学习、冷启动与蒸馏 | [报告](https://arxiv.org/abs/2501.12948) |
| 2025 下半年 | V3.1、V3.1-Terminus | 混合思考及 Agent 使用改进 | [官方更新日志](https://api-docs.deepseek.com/updates/) |
| 2025-09 / 12 | V3.2-Exp / V3.2 | DSA；进一步扩展推理与工具使用训练 | [V3.2 报告](https://arxiv.org/abs/2512.02556) |
| 2025-12—2026-01 | mHC、Engram | 残差拓扑与条件记忆研究 | [mHC](https://arxiv.org/abs/2512.24880)、[Engram](https://arxiv.org/abs/2601.07372) |
| 2026-04 起 | DeepSeek-V4 系列 | CSA/HCA、mHC、Muon、百万上下文 | [V4 报告](https://arxiv.org/abs/2606.19348) |
| 2026-09 | DeepSeek-V4.1-Flash | CED、CSA2、FP4 KV、Engram、DSpark、原生视觉 | [V4.1 报告](https://arxiv.org/abs/2609.19969) |
| 2026-10-17 | V4.1-Flash-1017 | 同一家族的能力与任务效率更新 | [模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |

说明：模型发布、论文提交和论文修订可能不是同一天。表中日期表达发布或阶段，不将技术报告日期统一替代模型发布日期。V4 的发布依据 2026-04-24 官方公告；论文链接使用当前官方研究索引指向的 arXiv 条目。

## 四、第一条路线：先降低“每个历史 Token 存多少”——MLA

### 1. 原始注意力为何需要 KV Cache？

自回归生成时，新的 Query 会读取过去位置的 Key、Value。缓存它们可以避免重复计算，但历史越长、层数越多，缓存越大。

在普通 MHA 中，可概念性写为：

$$
k_t=h_tW_K,\qquad v_t=h_tW_V
$$

### 2. MLA 的核心：先缓存一个更小的潜在表示

用行向量约定，教学示意为：

$$
c_t=h_tW_{DKV},\qquad k_t^{C}=c_tW_{UK},\qquad v_t=c_tW_{UV}
$$

可以理解为：**用一个较小的压缩表示，承载内容 K、V 所需的信息。**真实 MLA 还涉及独立的位置相关分量与权重吸收，不能只凭这三个式子实现完整模型。

与 GQA 的区别：GQA 让多个 Query 头共享 K、V 头；MLA 学习跨头的低维潜在表示。两者都旨在节省缓存，但不是同一种结构。

**MLA 减小缓存，不意味着只关注部分历史。**在 V2/V3 阶段，缓存压缩与注意力稀疏需要分开理解。

原始资料：[DeepSeek-V2 技术报告](https://arxiv.org/abs/2405.04434)，重点阅读 MLA 与 Decoupled RoPE 部分。

## 五、第二条路线：改进 FFN 与路由——DeepSeekMoE

### 1. 细粒度专家与共享专家

原始 Dense FFN 是每个位置都使用同一个完整网络。DeepSeekMoE 的方向是将专家切分得更细，让路由有更多组合空间，并设置共享专家来承担共通知识。

简化示意：

$$
y=E_{shared}(x)+\sum_{i\in\mathcal S(x)}p_i(x)E_i(x)
$$

这里专家是数值计算网络，不是人工分配的“数学老师”“英语老师”。它们的分工通过训练形成，也不保证符合人类学科边界。

资料：[DeepSeekMoE 方法论文](https://arxiv.org/abs/2401.06066)。

### 2. 专家不能全挤到少数几个节点

路由失衡会让部分专家拥挤，其他专家闲置。Auxiliary-Loss-Free Balancing 在选择专家时加入可动态调整的偏置，以改变被选中的机会；专家输出的加权仍与原始路由分数有关。

资料：[Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)。

### 3. V3 的实例与必要限制

V3 总参数约 671B、每 Token 激活约 37B；除最初三层外采用 MoE，每个 MoE 层有 256 个路由专家、激活 8 个，另有 1 个共享专家。

V3 的均衡方案仍保留一个很小的序列级辅助损失，因此“完全没有任何辅助损失”并不准确。激活参数少也不等于权重只占相应大小的显存。

资料：[V3 技术报告，第 2、4 节](https://arxiv.org/html/2412.19437v2)。

## 六、第三条路线：从全历史访问到稀疏检索——DSA

### 1. V3.2：压缩表示之外，再减少实际读取的位置

DSA 使用轻量 Lightning Indexer 选择值得访问的位置，主注意力再读取被选中的 KV。

教学示意：

$$
\mathcal S_t=TopK\bigl(Indexer(q_t,history)\bigr)
$$

$$
z_t=\sum_{j\in\mathcal S_t}a_{t,j}v_j
$$

在报告对应设置中，每个 Query 选择 2048 个 KV 位置。这个 Top-K 控制访问集合，不是把模型最大上下文改成 2048。

还要计入索引器本身的开销，不能只按“选中位置数／上下文长度”估计整个模型加速比。

资料：[V3.2 报告，第 2 节](https://arxiv.org/html/2512.02556v1)、[V3.2 官方发布公告](https://api-docs.deepseek.com/news/news251201/)。

### 2. NSA 是相关研究，但不等于 DSA

NSA 研究了结合压缩、选择与局部注意力的硬件友好稀疏设计。DSA 是 V3.2 公开采用的具体方案。不能因为二者都由 DeepSeek 研究，就把名称互换，或称 V3 已经使用 NSA。

资料：[Native Sparse Attention](https://arxiv.org/abs/2502.11089)。

## 七、第四条路线：压缩序列条目与改变残差拓扑——V4

V4 保留 DeepSeekMoE 和 MTP，注意力升级为 **CSA＋HCA**，并引入 mHC 与 Muon。

| 模块 | 核心思路 | 对应基本架构 |
|---|---|---|
| CSA：Compressed Sparse Attention | 对历史条目压缩，再选择性访问 | Attention |
| HCA：Heavily Compressed Attention | 以更高压缩率保留历史信息 | Attention |
| 局部滑动窗口分支 | 保留邻近位置的信息通路 | Attention |
| mHC | 扩展残差流，并约束跨流映射 | 残差连接 |
| Muon | 调整训练参数更新方式 | 优化器 |

V4 报告中，Flash 为约 284B 总参数、13B 激活参数；Pro 为约 1.6T 总参数、49B 激活参数。它们是具体型号配置，不能与下一代 V4.1-Flash 混用。

**V4 不是把 V3 的 MLA 原样保留，再简单叠加稀疏遮罩。**报告明确介绍了新的混合注意力组件。

原始资料：[V4 技术报告](https://arxiv.org/abs/2606.19348)、[HTML 全文](https://arxiv.org/html/2606.19348v1)、[V4 官方发布公告](https://deepseek.com/en/news/v4-preview/)。

### mHC 为什么值得单独学？

原始残差写作：

$$
x_{l+1}=x_l+F_l(x_l)
$$

mHC 将残差信息通路扩展为多个流，并对连接映射施加约束，以改善稳定性。它改变的是跨层信息传递拓扑，不是增加注意力头，也不是 MoE 的专家路由。

原始资料：[mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)。

## 八、最新架构：V4.1-Flash 为何采用 CED？

### 1. 从 Decoder-only 讨论到输入／输出计算不对称

V4.1-Flash 的官方模型卡描述：40 层由 20 层因果 Encoder 和 20 层 Decoder 组成。输入预填充每 Token 激活约 8B 参数，输出解码约 16B。

Decoder 的全局 KV 从最终 Encoder 隐藏状态投影得到，而不是每个 Decoder 层都从自身隐藏状态独立生成。

**它不是直接恢复 2017 年翻译模型的“双向源文本 Encoder＋目标文本 Decoder”。这里的 Encoder 是因果的，设计目标是降低输入密集型 Agent 的 Prefill 成本。**

资料：[V4.1 官方发布公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)、[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。

### 2. CSA2：跨层复用缓存与检索结果

| 模式 | 核心行为 |
|---|---|
| Full | 生成全局 KV 并进行索引 |
| Reindex | 复用 KV，但重新选择位置 |
| Reuse | 复用 KV 和已有选择结果 |

**Full 是完整执行 CSA2 计算路径，不等于标准的稠密 Full Attention。**各模式仍计算当前层的 Query 和局部窗口 KV。

CSA2 与 FP4 主 KV 缓存共同压缩全局缓存；分层索引减少后续层重复检索。官方报告给出的全局 KV 口径为约 890 bytes/token。它不是整个模型、全部窗口状态与服务开销的总内存。

### 3. 还有哪些部件发生了变化？

| 位置 | 官方披露组件 |
|---|---|
| 残差 | Single-Pass mHC |
| 条件记忆 | Engram |
| 生成加速 | DSpark |
| 持久缓存 | SWA Bounded Replay |
| 图像输入 | DeepSeek-ViT＋MLP 投影 |

V4.1 在主干预训练时不再使用 V3 风格的 MTP 联合训练，而采用单独阶段训练的 DSpark。SWA Bounded Replay 以有限重放恢复缺失窗口状态，包含近似处理，不能描述为完全免费、无损的恢复。

资料：[V4.1 技术报告，第 2—3 节](https://arxiv.org/html/2609.19969v1)、[PDF 入口](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)。

## 九、条件记忆：Engram 与 MoE 是两种不同的稀疏

Engram 使用局部 N-gram 相关的确定性寻址与可学习查表，结合上下文门控，将部分模式处理从重复计算转向条件记忆访问。

| 维度 | MoE | Engram |
|---|---|---|
| 主要动作 | 选择专家并执行网络计算 | 寻址并读取学习到的记忆 |
| 主要扩展方向 | 条件计算容量 | 条件记忆容量 |
| 与外部知识库关系 | 模型参数 | 同样属于模型参数，并非 RAG 文档检索 |

确定性寻址有利于提前从主机内存预取，但实际性能仍依赖带宽、缓存与实现。

研究论文与旗舰采用要分开：Engram 有独立论文；V4.1 官方资料明确列出采用，不能据此反推所有更早 DeepSeek 版本均有 Engram。

资料：[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)。

## 十、训练路线：DeepSeekMath → R1 → 工具使用与 Agent RL

### 1. GRPO：从一组回答中估计相对好坏

DeepSeekMath 提出 GRPO。教学上可用组内标准化奖励理解其优势估计的一部分：

$$
A_i\approx\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}{\operatorname{std}(r_1,\ldots,r_G)+\epsilon}
$$

同一题产生多个候选回答，以相对表现指导更新。这里仅展示直觉，不是完整目标函数；真实算法还有策略比率、裁剪等细节。

资料：[DeepSeekMath](https://arxiv.org/abs/2402.03300)。

### 2. R1：推理能力主要来自后训练路线变化

R1-Zero 探索直接通过 RL 激发推理；R1 在此基础上结合冷启动数据与多阶段训练，改善可用性。不能将“R1-Zero 没有先做 SFT”推广为“所有 R1 模型都没有 SFT”。

R1 的意义不是重新发明一套 Attention。它说明在既有基础模型上，训练目标与训练流程也能带来显著能力变化。

蒸馏版需要按底座区分：R1-Distill-Qwen 与 R1-Distill-Llama 不会因为名字中有 R1，就自动拥有原版 DeepSeek MoE 主干。

资料：[R1 技术报告](https://arxiv.org/abs/2501.12948)、[官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)。

### 3. 工具调用：把推理与执行放入同一条轨迹

V3.2 官方资料将 Thinking in Tool-Use 单独列为改进方向：模型需要结合工具反馈继续推理，而不仅是生成一个调用 JSON。

应区分网络结构、后训练、对话模板和 Agent 运行框架；工具调用成功并不能单独证明 Attention 改了。

资料：[V3.2 官方公告](https://api-docs.deepseek.com/news/news251201/)。

### 4. 数学验证作为并行研究

DeepSeekMath-V2 进一步探索自验证数学推理，可用于研究“生成解答”与“检查推导”的关系，不能把数学专用训练效果直接视作所有通用任务提升。

资料：[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)。

## 十一、生成路线：MTP 与 DSpark 要分别理解

| 技术 | 研究重点 | 要避免的误解 |
|---|---|---|
| V3/V4 的 MTP | 额外未来 Token 的训练信号，并支持推测生成 | 不等于取消自回归依赖 |
| V4.1 的 DSpark | 半自回归草稿与按置信度调度验证 | 草稿不是未经检查直接输出 |

“预测多个 Token”既可能指训练目标，也可能指生成加速机制。讨论时应明确是哪个阶段，并计入草稿计算、验证、接受率与负载。

资料：[V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)、[V4.1 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。

## 十二、系统路线：算法改变之外，还必须解决访存和通信

| 项目 | 类型 | 解决的主要问题 | 官方资料 |
|---|---|---|---|
| FP8 训练与系统配方 | 训练数值与实现 | 降低部分矩阵运算成本，兼顾数值稳定 | [V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) |
| FlashMLA | Attention 内核 | MLA 相关运算的高效实现 | [仓库](https://github.com/deepseek-ai/FlashMLA) |
| DeepEP | 专家并行通信 | 专家分发与合并的通信 | [仓库](https://github.com/deepseek-ai/DeepEP) |
| DeepGEMM | 矩阵计算库 | 高效执行矩阵运算 | [仓库](https://github.com/deepseek-ai/DeepGEMM) |
| DualPipe | 训练流水线并行 | 计算与通信重叠 | [仓库](https://github.com/deepseek-ai/DualPipe) |
| 3FS | 分布式文件系统 | 训练、推理的数据存储和访问 | [仓库](https://github.com/deepseek-ai/3FS) |
| DualPath | 推理系统研究 | Agent 多轮请求的 KV 缓存加载与带宽瓶颈 | [论文](https://arxiv.org/abs/2602.21548) |

这些项目不是七种新的 Transformer 层。它们让模型在真实集群中更有效率；支持的硬件、精度和内核能力需要以具体版本为准。

进一步阅读：[Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343)。

### 与长上下文部署直接相关的判断框架

评估一个模型时，应分别统计：

1. 权重占用，而非只看激活参数。
2. 全局 KV、局部窗口状态与其他运行状态。
3. Prefill 的输入处理成本。
4. Decode 的延迟、吞吐与带宽需求。
5. 专家通信、缓存传输及其重叠程度。

一个模型可能降低单 Token FLOPs，却仍受到通信或访存限制。更换模型架构后的收益也不能直接归因于某一个推理参数。

## 十三、多模态分支：视觉理解、图像生成与 OCR 分开看

| 分支 | 研究方向 | 原始论文 |
|---|---|---|
| DeepSeek-VL2 | 动态图像分块＋MoE/MLA 语言主干，视觉理解 | [2412.10302](https://arxiv.org/abs/2412.10302) |
| Janus-Pro | 统一多模态理解与图像生成的训练和规模扩展 | [2501.17811](https://arxiv.org/abs/2501.17811) |
| DeepSeek-OCR | 视觉方式承载文档上下文与 OCR | [2510.18234](https://arxiv.org/abs/2510.18234) |
| DeepSeek-OCR 2 | Visual Causal Flow | [2601.20552](https://arxiv.org/abs/2601.20552) |
| V4.1-Flash | 通用模型中的原生图像、文本输入及文本生成 | [2609.19969](https://arxiv.org/abs/2609.19969) |

不要因为某个分支可以生成图片，就推断全部 DeepSeek 通用模型都能生成图片；也不要把独立 OCR 项目等同于所有旗舰模型的视觉编码器。

## 十四、后续数值讲解建议

延续“给定向量与矩阵，逐步计算”的教学方式，建议按以下顺序：

1. **MHA → GQA → MLA**：先比较保存哪些 K、V，再用小矩阵演示压缩。
2. **Dense FFN → DeepSeekMoE**：用 4 个路由专家和 1 个共享专家，算一次 Top-2 路由。
3. **MLA → DSA**：把“缓存如何存”与“检索哪些位置”分开。
4. **DSA → CSA/HCA → CSA2**：分别讲压缩序列条目与跨层复用。
5. **普通残差 → mHC**：看多条残差流如何读取、混合和写回。
6. **Decoder-only → CED**：比较 Prefill 与 Decode 为什么可以计算不对称。
7. **GRPO → R1 → Agent RL**：在网络结构之外理解能力如何形成。

## 十五、优先阅读清单

### A. 最新事实与版本入口

- [DeepSeek 官方 Research & News](https://www.deepseek.com/en/news/)
- [API 更新日志：最新版本与服务状态](https://api-docs.deepseek.com/updates/)
- [V4.1-Flash 发布公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)
- [V4.1-Flash-1017 官方模型卡与权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- [V4.1-Flash 原始模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

### B. 通用模型主线

| 资料 | 用途 |
|---|---|
| [DeepSeek LLM](https://arxiv.org/abs/2401.02954) | 最初的通用语言模型路线 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA＋MoE |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | MoE 均衡、MTP、FP8、系统协同 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 推理 RL 与蒸馏 |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DSA＋工具使用与 RL |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | CSA/HCA、mHC、Muon |
| [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) | CED、CSA2 与缓存压缩体系 |

### C. 可单独拆解的关键方法

- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [mHC](https://arxiv.org/abs/2512.24880)
- [Engram](https://arxiv.org/abs/2601.07372)
- [DualPath](https://arxiv.org/abs/2602.21548)
- [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)

### D. 资料口径与核查边界

- 内容以 DeepSeek 官网、API 文档、deepseek-ai 官方模型卡／代码和作者论文为依据。
- 最新状态优先采用更新日志与 1017 模型卡；架构采用 V4.1 整模型报告。
- 未将媒体预测的型号或未正式发布的后继型号写入已发布清单。
- 未逐项重现性能榜单；官方性能、缓存和成本数据有实验口径，不能直接当作不同硬件部署的保证。
- 本文是路线梳理与阅读索引，不是各篇论文的完整逐页解读。
