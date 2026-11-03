# Qwen3.8-Flash-Next 架构深度解读：从 Transformer 到 GDN、QSA、GR 与 N-gram

> 版本：2026-11-03 · 面向希望真正读懂模型结构图、参数配置与推理部署的读者。  
> 主资料：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.8-flash-next) · [官方技术报告](https://arxiv.org/abs/2608.30320) · [官方 GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## 阅读说明

全文用三种标记区分证据强度：**【官方配置】**表示模型卡明确给出的数值；**【论文机制】**表示论文或官方博客解释的机制；**【教学简化】**表示为帮助理解而设计的例子，不能直接当作该模型实际权重或完整代码。公式首先解释基本机制；具体门控、归一化、索引和并行 Kernel 的精确实现，应以官方技术报告和所部署版本源码为准。

## 0. 一分钟读懂整个模型

<div style="padding:18px;border:1px solid #bcc9d7;border-radius:12px;background:#f7faff;color:#17263b;line-height:1.8">
<div style="text-align:center;background:#e6d7ee;padding:10px;border-radius:8px"><b>输入文本 / 图片</b> → Token IDs / 视觉特征</div>
<div style="text-align:center;margin:8px">↓</div>
<div style="text-align:center;background:#e8d8f1;padding:10px;border-radius:8px"><b>Vocabulary Embedding</b>：把 Token ID 变成 2560 维向量<br><b>N-gram Embedding</b>：在第 2 层注入局部词组记忆</div>
<div style="text-align:center;margin:8px">↓</div>
<div style="text-align:center;background:#e5f0ea;padding:12px;border-radius:8px"><b>重复 12 组 Hybrid Block，每组 4 层</b><br>① GDN → MoE<br>② GDN → MoE<br>③ GDN → MoE<br>④ QSA → MoE<br><small>每个子层围绕 4 路 Expanded Residual 做 GR Read 和 GR Write</small></div>
<div style="text-align:center;margin:8px">↓</div>
<div style="text-align:center;background:#f8e8c8;padding:10px;border-radius:8px"><b>GR Read → LM Prediction Head</b>：输出下一 Token 概率<br><b>MTP Modules</b>：额外训练/推理辅助预测后续 Token</div>
</div>

四个最重要的分工：**GDN 是压缩式记忆；QSA 是稀疏式回查；MoE 是逐 Token 特征加工；GR 是跨层信息高速公路。** N-gram 则为常见局部组合提供低计算成本的参数化查表记忆。

## 1. 从标准 Decoder-only Transformer 出发

2017 年 Transformer 中，一个典型子层可写成 `x → Self-Attention → Add & Norm → FFN → Add & Norm`；现代 Decoder-only 模型多使用 Pre-Norm、因果掩码以及不同残差布局。这里先抽象为：

$$
a_t=\operatorname{Attention}(x_1,\ldots,x_t),\quad u_t=x_t+a_t,\quad y_t=u_t+\operatorname{FFN}(u_t)
$$

- **Attention / Token Mixer**：把不同位置的信息引入当前 Token。例如“他”根据上文“学生”得到人物线索。它负责**跨 Token 混合**。
- **FFN / MoE**：在每个 Token 已融合上下文的特征向量内部，进行非线性变换，负责**同一 Token 内跨特征加工**。不会在 FFN 内重新直接计算不同 Token 的注意力。
- **Residual**：保留旧表示并允许新计算结果叠加，让深层训练更稳定。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

### 一个可算的例子：注意力与 FFN 干的事不同

假设当前 Token“他”的特征向量原来是 `x=[0.2,0.1]`；注意力从“学生”等前文提取信息 `a=[0.6,0.3]`，则 `u=x+a=[0.8,0.4]`。随后一个**教学简化**的 FFN：

$$
W_1=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad W_1u=[1.2,0.4]^T,
\quad \operatorname{ReLU}(W_1u)=[1.2,0.4]^T
$$
再乘 `W₂`、接残差得到最终表示。注意：**FFN 的输入是已经融合上下文的 `u`，不是独立地访问“学生”Token。** 这也是理解 GDN/QSA 与 MoE 分工的起点。

## 2. 图左边：48 层究竟怎么排列？

**【官方配置】**隐藏布局：`12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`。

| 层号（按 1 起算） | Token Mixer | 后续特征变换 |
|---|---|---|
| 1、2、3 | GDN | MoE |
| 4 | QSA | MoE |
| 5、6、7 | GDN | MoE |
| 8 | QSA | MoE |
| … | … | … |
| 45、46、47 | GDN | MoE |
| 48 | QSA | MoE |

所以总共有 **36 个 GDN 层、12 个 QSA 层、48 个 MoE 层**；图上的 `3×` 与 `1×` 指的是每一组中的频次，不是同一个 GDN 模块内部执行 3 次。图中省略号代表中间重复层。

### 为什么不全部用 GDN 或全部用 QSA？

GDN 用固定大小状态递推，长期历史被**压缩**，每步不必回看全部旧 token；但压缩有信息损失。QSA 为少量重要历史位置提供更精确的**显式回查**，但需要索引与选中 KV 的访存。3:1 混合，试图折中长期记忆成本和精确查找能力。官方博客把它概括为 GDN 负责记住、QSA 负责找回。注意：**GDN 不等于没有注意力，QSA 也不等于无损完整注意力。**

## 3. GDN：Gated DeltaNet 的递推记忆

来源：[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 与 [NVIDIA 官方研究页](https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule)。

### 3.1 从标准 Attention 过渡到线性状态

标准因果注意力在位置 `t` 使用 `q_t` 对历史各 $k_i$ 计算权重，再加权 `v_i`：

$$
o_t=\sum_{i\le t}\operatorname{softmax}(q_tk_i^T/\sqrt d)_i v_i
$$
线性注意力类方法尝试把历史压缩为状态矩阵 `S`，直观形式是 `S_t=S_{t-1}+k_tv_t^T`、`o_t=q_tS_t`（**教学简化**；实际还有归一化、门控等）。不再需要每次重读全部历史 KV。

### 3.2 Delta Rule：覆盖旧记忆，而非无限叠加

一个有启发性的简化写法：

$$
\hat v_t=k_t^TS_{t-1},\qquad S_t=\alpha_tS_{t-1}+\beta_tk_t(v_t-\hat v_t)^T
$$

这里 `α` 是旧记忆衰减门、`β` 是写入强度；$v_t-\hat v_t$是**预测误差 / 修正量**。维度和左右乘约定会随实现改变，公式表达的是 Delta 更新思想。

**【教学简化】数值例子**：假设某一记忆槽旧值为 `0.4`，当前需要写入 `0.9`，写入比例 `β=0.5`、先不衰减，则 `new=0.4+0.5×(0.9−0.4)=0.65`。如果新观察与旧记忆一致，差值为零，就没必要重复写入。与 `old+new` 的无差别累加相比，它允许**定向改写**。

### 3.3 GDN 的 Q、K、V 参数

| 配置 | 数值 | 直观含义 |
|---|---:|---|
| Q/K head 数 | 16 | 查询与键的通道组 |
| V head 数 | 48 | 值/输出通道组 |
| head dimension | 128 | 单个头内部特征宽度 |

$16×128=2048$ 是 Q/K 按头拼接的名义宽度，`48×128=6144` 是 V 的名义宽度；它们**不需要都等于 2560 隐藏维度**。模型会通过投影完成升降维。$48/16=3$ 说明 V 头数是 QK 头数的三倍，不能直接理解为“48 个完整的标准 Softmax 注意力头”。

### 3.4 对服务部署意味着什么？

GDN 的核心历史状态大小不随序列长度线性增加（在固定配置下）；但整个模型依然有 12 层 QSA KV、索引及其他缓存，**绝非 1M 上下文零显存成本**。Prefill 仍需遍历输入并计算状态，Decode 每一步更新状态。

## 4. QSA：Qwen Sparse Attention 如何定位重要历史？

**【官方博客和技术报告】**QSA 引入按 micro-block（微块）评估重要性的轻量索引器，在稀疏选择后进行注意力计算；它不是逐 Token 全量扫描后的简单 Top-K。主参考：[官方博客 Attention 章节](https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towardsultimate-cost-efficiency_603501) 和 [官方论文](https://arxiv.org/abs/2608.30320)。

<div style="padding:14px;border:1px solid #b3c7da;border-radius:10px;background:#f7fbff;color:#183047">
<b>QSA 的概念性检索流程（非精确 Kernel 流程图）</b><br>
<div style="padding:8px;margin:8px 0;background:#e6eef8;border-radius:6px">当前查询 Query</div>
<div style="text-align:center">↓ 对压缩后的历史微块索引打分</div>
<div style="display:flex;gap:5px;margin:8px 0;flex-wrap:wrap"><span style="padding:8px;background:#d3e9d6">块 A ✓</span><span style="padding:8px;background:#e7e9ee">块 B</span><span style="padding:8px;background:#d3e9d6">块 C ✓</span><span style="padding:8px;background:#e7e9ee">块 D</span><span style="padding:8px;background:#d3e9d6">块 E ✓</span></div>
<div style="text-align:center">↓ 在选中块所对应的 Token 上做 Attention</div>
<div style="padding:8px;background:#e4f3ec;border-radius:6px">输出检索到的上下文表示</div>
</div>

### 4.1 Q 与 KV 头：GQA

**【官方配置】**Q heads=24、KV heads=2、head dimension=256。这是一种分组查询注意力的配置，`24/2=12`：平均每 **12 个 Q 头共用 1 组 KV 头**。

$$
\mathrm{Q\ width}=24\times256=6144,\quad \mathrm{K\ width}=\mathrm{V\ width}=2\times256=512
$$

不能把 `6144` 误认为输入隐藏维；输入是 2560 维，经线性层映射到多头 Q 空间。

### 4.2 Indexer 与实际 Attention 不是同一个东西

| 部件 | 配置 | 工作 |
|---|---|---|
| 轻量 Indexer | MQA，4 Q heads、1 shared K head | 判断哪些历史块值得查看 |
| Indexer head dim | 128 | 索引向量的特征宽度 |
| 正式 QSA Attention | 24 Q / 2 KV，head dim 256 | 对候选历史内容做加权信息融合 |
| Budget | 512 blocks 或 2048 tokens | 限制被选入稀疏计算的历史预算 |

**特别注意**：模型卡把 `512 blocks or 2048 tokens` 放在一起，并不意味着所有实现、所有请求都固定恰好处理 2048 个历史 token；这是配置的预算表述，实际还涉及块粒度、局部/特殊位置、算子实现和序列边界。也不能把它理解成模型最大输出 2048 token。一个简单的算术直觉是 `2048÷512=4 token/block`，但**不要仅凭这条除法断言每个内部索引单元总是四个 token**，以实现为准。

### 4.3 RoPE 64 与 Head 256 不矛盾

**【官方配置】**每个 QSA 头是 256 维，但旋转位置编码维度只有 64。可以将其理解为：只有头向量中指定的 64 维承载旋转位置信息，其余维度用于其他内容表达；不意味着位置只能表示 64 个 token。

二维旋转示意：

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}=
\begin{bmatrix}\cos\theta_p&-\sin\theta_p\\\sin\theta_p&\cos\theta_p\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
$$

例如向量 `[1,0]` 在 `θ=90°` 时旋转为 `[0,1]`。真实 RoPE 为不同维度对采用不同频率，利用相对位置差影响 QK 内积。参考：[RoFormer](https://arxiv.org/abs/2104.09864)。

### 4.4 复杂度的正确理解

完整 Attention 训练计算常写作序列长度 `N` 的二次规模 `O(N²)`；稀疏 Attention 在选中预算 `K≪N` 的前提下，主注意力部分近似与 `N×K` 相关，但**索引构建、索引打分、缓存读写和实现常数不可忽略**。QSA 的意义之一正是降低索引自身的开销。官方给出其特定实验场景下 1M token 的 Attention Kernel Prefill、Decode 加速数值；该数据不应直接外推成端到端 vLLM 每个请求同倍提速。

## 5. MoE：512 个专家为何每个 Token 只激活 10+1？

标准 FFN 可写为 `FFN(x)=W_down σ(W_up x)`；SwiGLU 型专家通常还有门控投影，例如 `W_down(SiLU(W_gate x)⊙W_up x)`。它在**每个 Token 的特征维**进行非线性组合。

MoE 让 Router 为各 Token 打分，选择少数专家：

$$
y_t=\sum_{i\in\mathrm{Top10}(r(x_t))}w_{t,i}E_i(x_t)+E_{\rm shared}(x_t)
$$

上式是**教学概念式**；真实实现中共享专家可能有单独的缩放、门控与归一化。

**【官方配置】**每层 512 个 routed experts，选 10 个，外加 1 shared expert；expert intermediate dimension=640。

**【教学简化】**想象某个 Token 表示为 `[语法=0.7, 数学=0.1, 代码=0.8]`，Router 可能将主要权重放在擅长代码、语法的专家，而共享专家始终提供通用加工。路由是**按 Token / 按层动态发生**的，不是“整段对话永久选择同 10 位专家”。

### 为什么“125B 总参数，6B 激活”不矛盾？

125B 是主模型存有的参数量，6B 是处理一个 Token 时实际参与计算的大致模型参数规模（官方公布的架构级口径）；并非激活 6B 后其余 119B 永远不用。**不同 Token 可能选择不同专家**。51B N-gram 和 4B MTP 另计，不能简单将所有参数都当作每 token Dense 前向计算量。

### 显存的一个陷阱

“6B active”不等于只需存 6B 权重。通常 512 个 routed experts 的权重仍须可访问；若做 Expert Parallel，可分摊不同 GPU 并增加跨卡通信。经验计算权重体积是 `参数数 × 每参数字节`，例如主模型 125B 的纯 BF16 权重理论约 `250 GB`（十进制），加上额外嵌入、MTP、视觉模块、量化元数据、KV/state、CUDA 工作区与碎片等；不能据此直接得出可运行机器配置。

## 6. GR：为什么图上有四条 Expanded Residual？

这是图中**最需要抛开传统单一残差思维**的部分。标准残差通常只有 `x ← x + F(x)`；GR 扩展为 4 条并行残差分支，使不同信息通路保留不同内容。图中的白色四圆长方框表示四路状态的**概念示意**，不表示每路只有四个数字。

### 6.1 GR Read：如何从四条通路读？

**【论文机制】**读操作使用随当前内容变化的、逐元素的动态门来调制分支信息，再得到供 GDN/QSA/MoE 使用的 hidden state。可以用**教学简化**表达：

$$
h=\sum_{b=1}^{4}g_b(R)\odot R_b
$$

其中 `R_b∈ℝ^{2560}` 是第 `b` 条残差，`g_b(R)` 是依赖输入的逐元素门控向量。真实实现还包含专门的低秩参数化、归一化等，不应将该式当作源码等式。

**【教学简化】**若四条通路在某一特征上的值是 `[1.0,0.2,0.8,0.1]`，对应读门是 `[0.1,0.2,0.9,0]`，那么读结果约 `0.1+0.04+0.72=0.86`。同一层处理另一个 Token 时门控可能完全不同。与四路简单平均相比，GR 能让当前内容更依赖第 3 路保存的特征。

### 6.2 GR Write：把结果写回哪一路？

**【论文机制】**写门采用分支级标量门控。简化写为：

$$
R'_b=R_b+\lambda_b\,F(h),\quad b=1,2,3,4
$$

例如一个子层更新值 `F(h)=0.5`，四个写门 `[1,0,0.2,0]`，则对应分支增量为 `[0.5,0,0.1,0]`；第二、四路可继续保留其先前内容。**不是把 4 条残差拼接起来就变成 10240 维，直接喂给注意力**；它们通常先经 GR Read 汇聚回用于子层计算的 hidden state。

### 6.3 Bottleneck Rank=320 是什么意思？

**【官方配置】**4 branches，bottleneck rank=320。低秩瓶颈的直觉：若直接学习一个 `2560×2560` 的大变换，会有约 `6.55M` 个矩阵元素；若某种映射因子化为 `2560×320` 与 `320×2560`，就约 `1.64M` 个元素，参数量减少约 75%。**这是低秩一般原理的算术示例，不代表 GR 必然恰好使用这两个矩阵或这个精确参数量。** GR 实际门控张量形状应查看技术报告及代码。

### 6.4 图里一层的箭头怎样走？

1. **Expanded Residual → GR Read**：四路历史信息被选择性读出。
2. **GR Read → GDN/QSA**：做跨 Token 信息处理，产生新的 hidden state。
3. **Hidden State → GR Write**：把注意力结果按门写入四路残差。
4. **更新的 Expanded Residual → GR Read**：为 MoE 读取信息。
5. **GR Read → MoE**：当前 Token 内特征重组。
6. **MoE → GR Write**：再把 MoE 输出写入多路残差，传给下一层。

每个 GDN 和 QSA 层都采用这种 GR+Mixer+MoE 思路。**图上横向箭头有部分从右向左，是为了排版，不代表时间倒流或逆向因果注意力。** 官方指出 GR 结合 Hyper-Connections 的宽残差思想与动态门控，并去掉不必要的分支间额外混合。参考：[Hyper-Connections](https://arxiv.org/abs/2409.19606)、[Qwen 技术报告](https://arxiv.org/abs/2608.30320)。

## 7. Vocabulary Embedding 和 N-gram Embedding

### 7.1 Vocabulary Embedding：一个 Token ID 查一行

**【官方配置】**词表大小（补齐后）`248,320`；hidden dimension `2560`。Embedding 可视为矩阵 `E∈ℝ^{248320×2560}`。Token ID 为 `i` 时取 `E[i]`。

其元素数是 `248320×2560=635,699,200≈0.636B`。该乘积只是输入嵌入矩阵大小；输出 Prediction Head 是否权重绑定、其他投影及其精确占用，需查实际权重配置，**不要直接把输入与输出都算两遍**。

### 7.2 N-gram：不是看单词，而是看局部组合

假设分词结果 `['New','York','City']`，则 bigram 可以是 `('New','York')` 和 `('York','City')`，trigram 是 `('New','York','City')`。**N-gram 在这里作用于 Token IDs 而不保证对应人类单词边界。**

普通 embedding：`E[token_t]`；N-gram 思路：对当前位置前两个/三个 token 组成的键进行哈希或映射，取 `T[key]`，再投影/融合到模型状态。其关键区别是：**单个 token 的向量在不同邻近组合中可以得到不同的额外查表信息**。

### 7.3 20,000,000 与 51B 怎么对应？

**【官方配置】**N-gram embedding 表有 `20,000,000` 的大小指标，额外参数约 `51B`。数值上 `20,000,000×2560=51,200,000,000=51.2B`，与官方约 51B 一致。这强烈提示主要容量可理解为约 2000 万行 × 每行 2560 维，但具体多表组织、哈希碰撞和预取实现要以源码/报告为准；不能仅从模型卡断言每个 bigram 和 trigram 都有唯一独立无冲突的行。

`51.2B×2 byte≈102.4 GB` 是若全部 BF16 驻留时的纯数据量级；若采用量化，容量会不同。官方明确指出这部分可放在主机内存，通过提前确定的访问地址异步预取，而不是必须永久驻留 GPU 显存。**Host offload 不等于数据读取零延迟**，实际性能取决于 RAM、PCIe、预取命中和服务端实现。

### 7.4 为什么只在第 2 层？

**【官方配置】**N-gram embedding 位于 layer 2 only。它在网络早期提供局部组合信息，让后续 46 层左右继续加工，而不是每一层都重复存一份 51B 表。图里的旁路从 Input Tokens 指向 N-gram，再接入早期 GDN 层，**不能误解成第二套完整 Transformer**。

来源：[Qwen 官方博客 Embedding 章节](https://qwen.ai/blog?id=qwen3.8-flash-next)；对照：[DeepSeek Engram: Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)。两者思想相关，但**不能认为 Qwen 的 N-gram 模块与 Engram 在哈希、融合与参数细节上完全相同**。

## 8. Prediction Head、LM Output 和 MTP

### 8.1 Prediction Head 如何给下一个 Token 打分？

最终隐藏向量 `h_t∈ℝ^{2560}` 通过输出投影得到 `248320` 个 logits：

$$
z_t=W_{out}h_t,\qquad p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

输出是词表每个 token 的概率分布，采样策略（temperature、top-p 等）再选择输出 token。图最上方的 GR Read 表示**在 LM Head 之前从多路残差中读出最终表示**。

### 8.2 MTP：Multi-Token Prediction 并非默认一次输出所有 Token

**【官方配置】**额外约 4B 参数，1 层 MTP，采用 multi-step training。普通 next-token 目标预测 `x_{t+1}`；MTP 额外学习后续位置的预测信号，给模型更丰富的训练监督，也可能在支持的推理框架中用于自推测解码（speculative decoding）。

**【教学简化】**上下文“今天上海的天气”，主头预测下一 token“很”，MTP 辅助分支可以尝试预估后面“好”等 token；推理时需要验证候选是否被主模型接受，**并不是无条件跳过每个 token 的计算和校验**。图上 MTP 和 Prediction Head 是不同的输出路径。可对照：[DeepSeek-V3 Technical Report 的 MTP 设计](https://arxiv.org/abs/2412.19437)，但不能把 DeepSeek 的细节全盘当成 Qwen 实现。

## 9. 视觉编码器：图没有画完的部分

**【官方配置】**类型为 *Causal Language Model with Vision Encoder*。因此完整多模态系统不只是图中语言模型部分：图片经视觉编码/投影变成可供语言模型处理的视觉表示，与文本序列共同进入因果语言模型。图中只呈现 LM 主干；不能由这张图推断视觉编码器的层数、参数量、分辨率策略和图像 token 数量。需要查具体视觉配置与源码。

## 10. 262,144 与 1,000,000 上下文代表什么？

**【官方配置】**原生支持 `262,144` tokens（256 Ki tokens），可扩展到最多 `1,000,000` tokens。官方博客指出扩展可采用 YaRN。**原生训练/支持长度与经扩展可运行长度不是同一个保证**；部署时还受推理引擎、RoPE scaling、内存、并发、索引预算及视觉输入影响。参考：[Qwen 博客](https://qwen.ai/blog?id=qwen3.8-flash-next)。

对于你目前的 **8×L20 48GB、vLLM、FP8 KV、256K 目标**，务必分开看：

1. **权重**：125B 主模型及 N-gram、MTP、视觉模块的存储/分片/Offload。
2. **QSA KV Cache**：只有部分层是 QSA；KV 头数较少，但长度和并发会增加缓存量，具体分配取决于稀疏实现。
3. **GDN state**：固定大小的递推状态按 batch/sequence/层维护。
4. **激活与工作区**：Prefill 的中间张量、CUDA graph、索引/Kernels。
5. **并发总占用**：48 个同时活跃的 256K 请求，与“模型能支持单请求 256K”完全不是同一硬件需求。

不要直接使用标准 Dense 全层 KV 公式给出 QSA+GDN 的最终显存结论，也不要将 `max-num-batched-tokens=8192` 错看为单用户最多生成 8192 个 token。

## 11. 官方参数速查表：逐项对应架构图

| 参数 | 官方值 | 图中定位 / 应知概念 |
|---|---|---|
| Model Type | Causal LM + Vision Encoder | 图展示 LM 部分，视觉编码器未完整绘制 |
| Training | Pre-training & Post-training | 训练阶段，不是两种推理层 |
| Main parameters | 125B total / 6B active | 主要由稀疏 MoE 支撑容量 |
| Extra N-gram | 51B | 底部紫色外部表 |
| Extra MTP | 4B | 左上 MTP Modules |
| Hidden dimension | 2560 | 内部 Token 特征维度 |
| Token Embedding | 248320 padded | 底部 Vocabulary Embedding |
| N-gram Entries | 20,000,000 | 双/三元局部组合查表 |
| Number of Layers | 48 | 12 组 × 4 层 |
| GDN heads | V 48 / QK 16; dim 128 | 绿色 Gated DeltaNet |
| QSA heads | Q 24 / KV 2; dim 256 | 浅绿 Qwen Sparse Attn，GQA |
| QSA RoPE | 64 dims | Q/K 头向量部分维度旋转编码 |
| QSA Indexer | MQA; Q 4 / K 1; dim 128 | 独立的轻量块索引器 |
| QSA Budget | 512 blocks / 2048 tokens | 稀疏历史候选预算 |
| Routed Experts | 512 | 每层 MoE 候选专家池 |
| Activated Experts | 10 routed + 1 shared | 单 Token 的专家路径 |
| Expert Intermediate | 640 | 专家内部 FFN 宽度 |
| GR Branches | 4 | 四圆 Expanded Residual |
| GR Bottleneck | 320 | 低秩门控参数化配置 |
| LM Output | 248320 padded | Prediction Head |
| MTP | 1 layer, multi-step training | 辅助多 Token 预测 |
| Context | 262144 native / up to 1M extended | 模型上下文长度，不是并发保证 |

## 12. 图中每个模块的“自测问题”

1. **Input Tokens**：Token ID 是词本身还是整数索引？为什么图片需要视觉编码？
2. **Vocabulary Embedding**：为何 `248320×2560` 就能估计矩阵大小？
3. **N-gram Embedding**：为什么相同 Token 遇到不同前文可以得到不同查表补充？为什么只在第二层？
4. **Expanded Residual**：四路状态和四个 attention heads 有何区别？
5. **GR Read**：动态逐元素门与简单求和有什么区别？
6. **Gated DeltaNet**：为什么 Delta Rule 要写入 `目标值−旧估计值`？
7. **Qwen Sparse Attention**：索引器与真正做 Attention 的 Q/K/V 是同一套头吗？
8. **Hidden States**：这是单个 token 的当前表征，还是整个历史 KV Cache？
9. **MoE**：为什么 512 个专家的参数要存放可访问，但每个 Token 仅激活 10+1？
10. **GR Write**：为什么写门是分支级控制？
11. **Hybrid Block × L/4**：为什么 48 层恰好有 12 个 QSA 层？
12. **Prediction Head**：logits、softmax 概率和最终采样 token 的关系？
13. **MTP Modules**：为什么辅助预测不能等同于无限制一次生成多个最终 token？

## 13. 技术演进与必读原始资料

| 年份 | 技术 / 论文（原始来源） | 连接到本模型的作用 |
|---|---|---|
| 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Self-Attention、FFN、残差的基准架构 |
| 2021 | [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | QSA 中 64 维 RoPE 的原理 |
| 2024 | [Hyper-Connections](https://arxiv.org/abs/2409.19606) | 多分支残差的思想来源 |
| 2024/2025 | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | 用带门控的 Delta Rule 压缩序列历史 |
| 2024/2025 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 理解 MoE 与 MTP 的对照架构 |
| 2026 | [Conditional Memory / Engram](https://arxiv.org/abs/2601.07372) | N-gram 查表与条件记忆的相关路线 |
| 2026 | [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320) | **最直接的一手架构论文**：QSA、GR、N-gram、优化器及实验 |
| 2026 | [Qwen3.8-Flash-Next 官方博客](https://qwen.ai/blog?id=qwen3.8-flash-next) | 设计动机和性能实验的官方解读 |
| 2026 | [Qwen3.8-Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 参数表、上下文、使用说明 |
| 2026 | [Qwen GitHub 及技术报告 PDF](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | 官方 PDF 原文与可追溯仓库 |

### 建议的学习顺序

**第一轮（模块作用）**：1 → 2 → 5 → 6 → 7 → 8。  
**第二轮（核心数学）**：3 的状态更新 → 4 的稀疏检索 → 6 的读写门控。  
**第三轮（工程性能）**：9 → 10 → 官方技术报告的实验与 vLLM 实测。

---

**最后的总括**：Qwen3.8-Flash-Next 的突破不是单独“换了一种 Attention”，而是将 **时间维信息处理（GDN+QSA）、特征维非线性加工（MoE）、层间传输（GR）、局部静态记忆（N-gram）、辅助预测（MTP）** 分别优化后组合成 48 层主干。读图时始终问清：**信息从哪里来、沿哪条通路流、如何被加工、为此付出什么计算和内存代价**。
