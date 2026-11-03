# DeepSeek V4／V4.1 架构深度解读：从 CSA、HCA 到 CED、CSA2、Engram 与 DSpark

> 资料核对日期：2026-11-03。延续《Qwen3.8-Flash-Next 架构深度解读》的阅读方式：先看整体，再拆模块，穿插公式、可手算例子和部署含义。图示使用内嵌 HTML，适合在开启 HTML 与数学公式渲染的 Typora 中阅读。
>
> 主线：**V4 改造“如何读取长历史”；V4.1 进一步改造“哪些层需要重复计算、重复存储”。**

## 阅读说明

- **【官方配置】**：来自对应仓库的 `config.json`，不要跨模型套用。
- **【官方机制】**：来自技术报告、模型卡或官方实现。
- **【教学简化】**：自拟向量、权重、示意路径，只解释机制，不代表真实模型权重。
- **【推导】**：由公开配置进行的算术估计，必须同时看计算口径。

本文覆盖 V4-Flash、V4-Pro，以及 V4.1-Flash；补充已经核实的 V4-Flash-0731 和 V4.1-Flash-1017。没有把尚未在本文核实的 V4.1-Pro 规格填进表里。用户提供的 [ModelScope 官方合集](https://modelscope.cn/collections/deepseek-ai/DeepSeek-V4) 可作为下载入口；架构以逐个模型的官方配置和论文为准。

## 0. 一分钟看懂两代模型

<div style="padding:18px;background:#f5f8fc;border:1px solid #becddd;border-radius:12px;line-height:1.8;color:#17263b">
<div style="padding:10px;background:#ddeaf8;border-radius:8px"><b>V4：文本 → Embedding → 多层因果 Transformer → 预测头</b><br>每层：mHC 包围 Attention 与 MoE；Attention 兼顾局部窗口和长程压缩记忆。</div>
<div style="margin:10px 0;padding:10px;background:#e4f1e8;border-radius:8px"><b>长程分工</b><br>CSA：较轻压缩，再挑重要条目。<br>HCA：重度压缩，再读取压缩后的全局概览。<br>SWA：保留最近上下文的细节。</div>
<div style="padding:10px;background:#f6ead3;border-radius:8px"><b>V4.1：文本／图片 → 20 层因果 Encoder → 20 层 Decoder → 预测头</b><br>Decoder 全局 KV 来自 Encoder 末层；CSA2 跨层共享 KV 和索引；另有 Engram、Single-Pass mHC 与 DSpark。</div>
</div>

【官方机制】V4 使用 CSA／HCA 混合注意力；V4.1-Flash 使用 CED 与 CSA2。V4.1 的 Encoder 仍然是**因果的**，输出仍然是自回归文本。它不是把原始 2017 Transformer 的双向翻译 Encoder 原样搬回来。[S1][S2][S3]

## 1. 从熟悉的 Decoder-only Transformer 出发

先只看一个 Token 的表示 $x_t$。一层通常做两类工作：

$$
u_t=\operatorname{Attention}(x_t,\{x_j:j\le t\}),\qquad y_t=\operatorname{FFN}(u_t).
$$

【教学简化】用“上海旅行”说明。当前在生成“外滩”相关内容，注意力从历史里的“上海”“夜景”“预算”提取信息。假设三个 Value 是：

$$v_1=[1,0],\quad v_2=[0,2],\quad v_3=[1,1],\qquad a=[0.6,0.3,0.1].$$

则注意力聚合结果：

$$u=0.6v_1+0.3v_2+0.1v_3=[0.7,0.7].$$

为了直观看 FFN，临时用一个极小网络：

$$W_1=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\qquad W_2=\begin{bmatrix}1&0\\0&2\end{bmatrix}.$$

$$y=W_2\operatorname{ReLU}(W_1u)=W_2[1.4,0]^T=[1.4,0]^T.$$

**注意力组合历史位置的信息；FFN 加工当前向量里的特征。** 实际 DeepSeek 的专家使用 SwiGLU 等设计，上面的 ReLU 只是为了手算。MoE 改变的是 FFN 选择；稀疏注意力改变的是历史读取，两者不是同一个“稀疏”。

## 2. 版本与参数口径：先避免张冠李戴

| 模型 | 参数口径 | 每 Token 激活 | 本文关注点 |
|---|---:|---:|---|
| V4-Flash | 284B 总参数 | 13B | CSA／HCA、mHC、Hash MoE |
| V4-Pro | 1.6T 总参数 | 49B | 同一技术路线的更大配置 |
| V4-Flash-0731 | 官方称正式更新 | 不在此另造数字 | 官方模型卡称附带投机模块，与 DSpark 版本同结构 |
| V4.1-Flash | 552B 主干，另有 196B Engram | Prefill 8B；Decode 16B | CED、CSA2、跨层共享、原生视觉 |
| V4.1-Flash-1017 | 官方称 V4.1-Flash 小版本更新 | 不在此另造数字 | 推理、编码、Agent、视觉等能力及效率改进 |

来源：[S3][S4][S5][S6]。**552B 不能直接理解为包含 Engram 在内的全部存储参数。** 主干与 Engram 相加约 748B；这仍不是一个包含视觉、草稿模块、量化元数据等所有项目的精确文件体积清单。

“激活 16B”描述计算访问范围，不表示只需要保存 16B 权重。某个 Token 没选中的专家，另一个 Token 可能马上要选。

## 3. V4 的层如何排列？

【官方配置】V4-Flash 主干 43 层、隐藏维度 4096。前两层只使用 SWA，其后交替采用 CSA 与 HCA。每层都有 MoE，前 3 层采用 Hash 路由。配置还列出 1 个 next-token 辅助预测层，应与 43 层主干区分。[S1][S7]

<div style="padding:16px;background:#f7faff;border:1px solid #becddd;border-radius:10px;line-height:1.8;color:#17263b">
<div style="padding:8px;background:#e2eaf8;text-align:center">Token Embedding：每个位置获得 4096 维表示</div>
<div style="text-align:center">↓</div>
<div style="padding:8px;background:#e6f1e7">主干第 1～2 层：SWA + MoE</div>
<div style="text-align:center">↓</div>
<div style="padding:8px;background:#fff0d7">主干第 3～43 层：CSA + MoE、HCA + MoE 交替<br>Attention 与 MoE 周围都有 mHC；前 3 个 MoE 使用 Hash 路由。</div>
<div style="text-align:center">↓</div>
<div style="padding:8px;background:#eadff2;text-align:center">汇聚残差流 → 归一化 → 词表预测头</div>
</div>

V4-Pro 的规模不同：主干 61 层、隐藏维度 7168，前两层使用 HCA；不要把 Flash 的开头直接复制到 Pro。[S1，§4.2.1]

## 4. CSA：先压缩，再找重要历史

### 4.1 为什么需要两步？

假设有 100 万个历史 Token。全部保留细节、每步都完整读取，缓存和带宽都会很大。

**压缩减少存多少；选择减少读多少。** 二者可以组合：先建立更小的历史表示，再对当前 Query 挑出值得读取的条目。

【官方机制】V4 的 CSA 使用轻度压缩与 Lightning Indexer，再把选中的长程条目和局部 SWA 分支共同用于注意力。Flash 的压缩率为 4，Top-K 为 512。[S1][S7]

### 4.2 用四个向量看压缩

【教学简化】设：

$$v_1=[1,0],\ v_2=[1,2],\ v_3=[3,0],\ v_4=[3,2].$$

最简单的摘要是平均：

$$\bar v=\frac{v_1+v_2+v_3+v_4}{4}=[2,1].$$

它只保留一个条目，但已经无法从 $[2,1]$ 唯一恢复四个输入。**压缩是学习有用表示，不是无损压缩文件。** 实际 CSA 有学习式、逐通道的压缩权重；压缩率 4 的条目来自跨度为 8 的重叠来源窗口，不能把上述四向量均值当成源码。[S1，§2.3.1]

### 4.3 Indexer 找位置，主注意力读取内容

【教学简化】压缩后的条目为 $c_1,\ldots,c_6$，轻量索引分数为：

$$s=[0.2,0.9,0.1,0.8,0.3,0.4].$$

若 Top-K=2，选择位置 $\{2,4\}$。随后主注意力使用自己的 Query 与这些条目计算权重。假设最后权重为 $[0.7,0.3]$：

$$o=0.7c_2+0.3c_4.$$

**索引器回答“去哪里找”；主注意力回答“读到以后怎样组合”。** 实际还要加入可见的局部窗口与因果掩码。

### 4.4 百万上下文里大约发生什么？

【推导】以 1,000,000 个 Token、压缩率 4 粗估：全局压缩池约 250,000 条；Flash 每个 Query 从中选最多 512 个长程条目，再结合 128 范围的局部窗口。

不能把 $1,000,000/512$ 当作真实加速倍数：仍有压缩、索引、投影、MoE、通信及缓存访问成本。也不能由“只读 512 个”推断“只存 512 个”，候选历史通常还要保留，以供下一次查询。

## 5. HCA：把历史压得更小，再读全局概览

【官方机制】V4 的 HCA 使用 128 倍的重压缩，不使用 CSA 那样的索引筛选，而对压缩后的可见条目进行注意力；同样保留局部 SWA 分支。[S1，§2.3.2]

【教学简化】可以把一本书分成许多段。CSA 像保留较细的段落摘要，再挑相关段落；HCA 像保留更粗的章节摘要，然后浏览所有章节。

【推导】100 万 Token 压缩 128 倍，大约剩 7,813 条。这个数量仍然随上下文增长，但远小于原始历史。与 CSA 的 512 条相比，它读得更多，却使用更粗的记忆。

| 机制 | 保留什么 | 长程读取方式 | 主要取舍 |
|---|---|---|---|
| SWA | 最近位置的细节 | 读局部窗口 | 单靠它不能直接回查遥远位置 |
| CSA | 较细的压缩历史 | 索引后稀疏读取 | 需要付出索引成本 |
| HCA | 更粗的压缩历史 | 读全部可见压缩条目 | 细节压缩更强 |

V4 在不同层交替使用两种全局视角。它们不是三个独立大模型，而是 Transformer 中不同的注意力组织方式。

## 6. Shared K=V MQA、RoPE 和输出投影

【官方配置】V4-Flash 有 64 个 Query 头、1 个 KV 头，head dimension=512，其中 RoPE 部分为 64 维；输出投影使用 8 组、每组低秩中间维度 1024。[S7]

### 6.1 多个 Query 头为什么可以共享 KV？

【教学简化】想象同一排书，多个读者提出不同问题。共享的是书架，不是问题。于是每个头仍能算不同的权重：

$$a_{h,j}\propto\exp(q_h^Tk_j/\sqrt d).$$

这里还进一步共享 Key 与 Value 表示：同一缓存向量承担匹配和内容读取功能。**1 个 KV 头不等于只剩 1 个注意力头。**

### 6.2 512 维与 RoPE 64 维并不矛盾

可以概念性地拆成：

$$q=[q_{\mathrm{content}};q_{\mathrm{rope}}],\quad 448+64=512.$$

部分通道携带旋转位置结构，其他通道保留内容空间。这只说明维度分工；V4 对输出的旋转处理等细节请看报告 §2.3.3，不能只按普通 RoPE 代码猜实现。

### 6.3 为什么要分组低秩输出投影？

【教学简化】多个头拼起来很宽，直接乘一个大矩阵成本很高。把头分组，先缩到较小空间，再汇合回 hidden size，相当于给每组设置一个信息瓶颈。它优化的是投影计算，和历史 Token 压缩是两个不同维度。

## 7. MoE：专家怎么选，Hash 路由又是什么？

【官方配置】V4-Flash：256 个路由专家，选 6 个，另有 1 个共享专家；V4.1-Flash：384 个路由专家，仍选 6 个，另有 1 个共享专家。[S7][S8]

【教学简化】忽略残差与缩放，某 Token 的计算可写作：

$$y=E_{\mathrm{shared}}(x)+\sum_{e\in\mathrm{TopK}(x)}g_eE_e(x).$$

若临时只选两个专家，结果分别为 $[2,0]$ 和 $[0,4]$，权重为 0.75、0.25，共享专家输出 $[1,1]$：

$$y=[1,1]+0.75[2,0]+0.25[0,4]=[2.5,2].$$

专家是可学习的 FFN，不应在没有实验证据时断言“第 3 个专门数学、第 8 个专门旅游”。

V4 前 3 层的 Hash 路由按 Token ID 查静态专家选择表，路由权重仍可由学习分数生成。它和 Engram 完全不同：**Hash MoE 查出“让哪些计算网络工作”；Engram 查出“取回哪些记忆向量”。** [S9]

## 8. mHC：为什么残差连接变成四条路？

标准残差近似写成 $x'=x+F(x)$。扩展成多路以后，先从多条残差流读入，再把子层结果写回多条流。

【教学简化】设只有两条流，$x_1=2,x_2=6$。读入权重为 $[0.75,0.25]$：

$$u=0.75\times2+0.25\times6=3.$$

若 $F(u)=4$，残差混合矩阵为：

$$B=\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix},\quad C=\begin{bmatrix}0.5\\0.5\end{bmatrix},$$

则：

$$X'=BX+CF(u)=[2.8,5.2]^T+[2,2]^T=[4.8,7.2]^T.$$

这里 $B$ 的元素非负、各行各列和为 1，是双随机矩阵。它让多路信息能够混合，同时约束残差混合本身的放大行为；并不等于整个神经网络都不可能放大信号。

【官方机制】V4 用 mHC 对扩展残差连接施加约束；V4.1 的 Single-Pass mHC 把输入混合系数移用前一个子块生成的系数，使计算可以更好地融合，减少激活读写。[S1][S2]

可以把它理解为：先准备好下一站的分流规则，让数据到站时不用等待“现算规则”再重复搬运。**Single-Pass 说的是残差混合实现，不是整个模型只推理一次。**

## 9. V4.1 的核心：因果 Encoder–Decoder

### 9.1 不要与翻译模型的 Encoder–Decoder 混淆

原始翻译 Transformer 可以先双向编码完整源句，再生成目标句。V4.1 的文本主干仍有因果约束，Encoder 处理某位置时不能偷看未来。

真正重要的改变是：**上半部分的全局 KV 从下半部分末层表示构造，而不是必须先完整运行上半部分所有历史位置。** [S2，§2.2]

<div style="padding:16px;border:1px solid #becddd;border-radius:10px;background:#f7faff;line-height:1.8;color:#17263b">
<div style="padding:9px;background:#e5dff3;text-align:center">文本 Embedding／视觉 Embedding</div>
<div style="text-align:center">↓</div>
<div style="padding:9px;background:#deecf8"><b>20 层因果 Encoder</b><br>处理历史与新输入，产生 Encoder 末层表示。</div>
<div style="text-align:center">↓</div>
<div style="padding:9px;background:#fff0d7"><b>从 Encoder 末层投影 Decoder 的全局 KV</b><br>跨层共享这些缓存；局部 SWA 状态另行维护和回放。</div>
<div style="text-align:center">↓</div>
<div style="padding:9px;background:#e1f0e4"><b>20 层 Decoder</b><br>生成时读取共享全局 KV；每层保留自己的 Query 和局部窗口。</div>
<div style="text-align:center">↓ 下一 Token → 继续自回归</div>
</div>

### 9.2 为什么 Prefill 只激活 8B，Decode 却是 16B？

【教学简化】假设输入 10,000 Token，局部窗口为 128，且把各层成本临时视为相同。

完整经过 40 层的工作量单位约为：

$$10,000\times40=400,000.$$

CED 的粗略单位是：

$$10,000\times20+128\times20=202,560.$$

后者约为前者的 50.6%。这只是结构直觉，省略全局 KV 投影、首个预测位置、回放和不同层成本。短输入时固定开销占比会更高。

【官方机制】长提示词的大部分位置可绕过完整 Decoder 计算，Decoder 的局部 SWA 通过尾部回放准备；生成新 Token 时仍运行更完整的主干。因此官方分别给出 Prefill 8B 与 Decode 16B。[S3]

## 10. CSA2：把“存储共享”和“索引共享”拆开

最容易误读的是 Full、Reindex、Reuse 三个名字。

| 模式 | 全局 main KV／indexer K | Top-K 位置 | 主 Query、局部 SWA KV |
|---|---|---|---|
| Full | 本层生成 | 本层选 | 本层计算 |
| Reindex | 复用前面来源层 | 本层重选 | 本层计算 |
| Reuse | 复用前面来源层 | 复用已有选择 | 本层计算 |

来源：[S2，§2.3]。**Full 是 CSA2 完整路径，不表示对全部历史进行稠密主注意力；Reuse 也不表示跳过整层。**

【教学简化】六张记忆卡：$c_1,\ldots,c_6$。A 层建卡并选 $\{2,4\}$；B 层共享卡但重选 $\{1,4\}$；C 层共享卡并沿用 $\{1,4\}$。B 与 C 不必重新建整套卡，C 还省去一次找卡。

即使 B、C 都读 $\{1,4\}$，它们的 Query 不同，最后权重和输出仍可不同。

### 10.1 配置里的来源层到底是谁？

【官方配置】V4.1-Flash 的 `kv_source_layer_ids` 为 `[2,8,14,20]`，`index_source_layer_ids` 为 `[2,8,14,20,24,28,32,36]`，均为零基编号。[S8]

| 零基主干层号 | 模式／角色 |
|---|---|
| 0～1 | SWA-only |
| 2、8、14 | Encoder 的 Full 来源层 |
| Encoder 其他全局层 | 复用相应来源层缓存与索引 |
| 20 | Decoder 的 Full 来源层；建立候选池 |
| 24、28、32、36 | Decoder Reindex 层 |
| Decoder 其他层 | Reuse 层 |

Encoder 全局层使用压缩率 2；Decoder 的全局缓存为压缩率 1。**压缩率 1 仍然可以稀疏读取，也仍可跨层共享。** 节省不只来自时间维压缩。

## 11. Hierarchical Sparse Indexer：再索引不必每层重扫百万历史

【官方机制】Decoder 第一个 Full 层扫描可见范围，并建立块级候选池。后面的 Reindex 层只在候选池内重新评分。[S2，§2.3.2]

【官方配置与推导】2048 块 × 每块 8 个位置 = 最多 16,384 个候选；最终 Top-K=512。[S8]

<div style="padding:16px;border:1px solid #becddd;border-radius:10px;background:#f7faff;line-height:1.8;color:#17263b">
<div style="padding:8px;background:#dfeafa">第一个 Full 层：扫描全部可见全局位置</div>
<div style="text-align:center">↓</div>
<div style="padding:8px;background:#ffedcc">自己的主注意力：选 Top-512<br>给后续层：另建最多 16,384 位置的候选池</div>
<div style="text-align:center">↓</div>
<div style="padding:8px;background:#dff0e3">后续 Reindex：在候选池内重新选 Top-512<br>后续 Reuse：直接沿用最新 Top-K</div>
</div>

【教学简化】像招聘时先做一次全公司简历初筛，再让各业务负责人从候选名单中挑人。后续负责人可以有不同偏好，但不必每次重扫全库。

候选池不是“整个模型只记住 16,384 Token”。第一个来源层仍访问全局，而且每个 Query 对应的池可不同。后续索引开销有界，不代表整网的所有计算、存储都与长度无关。

## 12. FP4 KV 与 890 bytes／Token：究竟省在哪里？

【官方机制】V4.1-Flash 全局 KV 占用约 890 bytes／Token，约为 V4-Flash 的 1/4；配合 SWA Bounded Replay，持久化缓存约为前代的 1/8。main KV 用 FP4，局部 SWA KV 保留 FP8。[S2][S3]

### 12.1 一个可手算的量化存储例子

【推导】512 维向量，若每值 4 bit，再每 16 通道配一个 8-bit scale：

$$512\times0.5+(512/16)\times1=288\ \mathrm{bytes}.$$

因此“FP4”不是刚好每元素 0.5 byte 的全部开销；scale 也要保存。更不能用 $288\times40$ 估全网缓存：多层共享缓存，还有不同压缩率、索引 Key 等项目。

### 12.2 百万 Token 与并发估算

【推导】只按官方全局缓存口径：

$$1,000,000\times890=890,000,000\ \mathrm{bytes}\approx0.829\ \mathrm{GiB}.$$

若 48 个独立会话各有 262,144 Token：

$$48\times262,144\times890/2^{30}\approx10.43\ \mathrm{GiB}.$$

这只是**全局 KV 的理论有效负载**。还要另算权重、局部 SWA、工作区、草稿模块、图捕获、页对齐、缓存副本，以及具体并行方案。它不是整台机器只要 10.43GiB 的承诺。

## 13. SWA Bounded Replay：用少量重算换少存缓存

【教学简化】一个工具 Agent 经常反复使用同一长前缀。全局缓存像长期档案，局部窗口像工作台上的最近几张便签。下次恢复任务时，可以载入档案，再重放尾部一小段来重建工作台。

【官方机制】V4.1 的 bounded replay 仅重放最近一个窗口，近似恢复缺失的局部状态；论文报告性能影响很小。它是经过验证的存储／计算折中，**不是逐元素完全精确的状态恢复保证**。[S2，§3.2.2]

需要区分三件事：运行时 HBM 缓存、主机／SSD 持久化缓存，以及命中后恢复局部状态的计算。1/4 与 1/8 对应的口径不同。

## 14. Engram：196B 参数为什么能低计算成本地查表？

### 14.1 和普通 Embedding 的区别

普通 Embedding 根据单个 Token ID 查一行。Engram 根据局部 Token 组合确定地址，再用上下文决定取回的记忆有多大作用。

【教学简化】把“上海／外滩／夜景”当成三个教学 Token；真实 tokenizer 未必这样切分。当前位置可形成 2-gram“外滩／夜景”和 3-gram“上海／外滩／夜景”。哈希函数把它们映射到不同记忆行：

$$r_2=H_2(t_{i-1},t_i),\qquad r_3=H_3(t_{i-2},t_{i-1},t_i).$$

设查回 $m=[2,1]$，上下文门控为 $g=0.2$，临时令投影为恒等映射，便可注入：

$$x'=x+gm=x+[0.4,0.2].$$

遇到不同语境，门控可能变成 0.8。**地址由 Token 组合确定，不代表记忆对所有上下文的影响一样。** 这不是外部检索原文，也不是把回答固定成词典输出。

### 14.2 196B 怎么算出来？

【官方机制】两处模块使用 2／3／4-gram；每阶 8 个哈希头，每头维度 256，表规模约 16M；模块位于零基层 1、14。[S2，§2.4.2][S8]

【推导】粗算：

$$2\times3\times8\times16,000,000\times256=196,608,000,000\approx196.6B.$$

真实表大小用不同素数，上式解释数量级。**不是每 Token 都与 196B 个值相乘，而是按地址稀疏读取其中很少一部分。**

因此表可以很大，计算却相对小；代价转移到存储、访存和传输。官方利用确定性地址进行预取，不能因此认为普通 CPU／GPU 搬运天然没有瓶颈。

### 14.3 与 Qwen 附件中的 N-gram Embedding 对照

| 角度 | Qwen 附件中的 N-gram | DeepSeek V4.1 Engram |
|---|---|---|
| 共同直觉 | 按局部 Token 组合读取参数化记忆 | 同左 |
| 本文已核实的 DeepSeek 细节 | 不把 Qwen 配置套过来 | 多阶、多头哈希、上下文门控、两处注入 |
| 与注意力区别 | 查模型参数表 | 查模型参数表；注意力则读取当前请求的上下文状态 |
| 与 MoE 区别 | 不等于专家路由 | 取回记忆向量，而非选择大 FFN |

这里的比较限定为用户附件所述机制，不构成对 Qwen 当前仓库的独立复核。

## 15. 视觉：图片怎样进入因果语言模型？

【官方机制】V4.1 的 DeepSeek-ViT 编码图像，经 3×3 pixel-unshuffle 和两层 MLP 投影得到语言主干可接收的视觉 Embedding，与文本一起进入主干。[S3][S8]

【教学简化】设视觉特征网格为 12×12，共 144 个位置。3×3 重排把每个小区域的 9 个向量拼到通道维，位置数变为 4×4=16，然后投影到语言 hidden size。

这减少了 9 倍的位置数；并不是把每九个位置简单平均，也不是把图片先翻译成一段文字。

视觉编码器可以先提取图像空间关系，语言主干仍按自己的因果序列规则生成文字。**视觉 Encoder 与 CED 的因果 Encoder 是两个不同模块。**

## 16. DSpark：草稿、验证和调度分别做什么？

【官方机制】V4.1 的 DSpark 有 3 个草稿 Transformer 块，一次计算 5 个草稿位置的基础 logits，再配合轻量 Markov head 和置信度调度验证。它在主干预训练后单独引入，而不是把传统联合预训练 MTP 原样保留。[S2，§2.4.3][S8]

【教学简化】草稿提出“建议／先／去／外滩／看看”。主模型验证后可能接受前三个，在第四个位置改成“博物馆”。有效输出来自验证规则；草稿不被无条件照收。

若某次草稿与验证共用 7ms，平均推进 3 个有效 Token，则约 429 Token/s；若原来每 Token 4ms，则为 250 Token/s。但若平均只接受 1 个，则约 143 Token/s，反而更慢。这些时间纯属教学假设。

因此加速取决于接受率、批量、主模型验证成本和调度。`block_size=5` 不等于任何时候都能五倍加速。

## 17. 两代参数速查：对应配置读字段

| 字段／概念 | V4-Flash | V4.1-Flash |
|---|---:|---:|
| 主干层数 | 43 | 40＝20＋20 |
| hidden size | 4096 | 5120 |
| Query 头／KV 头 | 64／1 | 64／1 |
| head dim／RoPE dim | 512／64 | 512／64 |
| 路由专家／激活数 | 256／6 | 384／6 |
| 共享专家 | 1 | 1 |
| expert intermediate size | 2048 | 2304 |
| SWA window | 128 | 128 |
| 主要全局注意力 | CSA＋HCA | CSA2 |
| index Top-K | 512 | 512 |
| residual streams | 4 | 4 |
| 最大位置数 | 1,048,576 | 1,048,576 |
| 辅助预测／草稿层配置 | 1 | 3 |
| Engram | 不套用 V4.1 配置 | 零基层 1、14 |

来源：[S7][S8]。两者均有 YaRN 相关位置扩展配置。`max_position_embeddings` 是配置上限，不表示任意部署都能在该长度下满足并发和延迟目标；辅助层也不要重复算进主干层数。

## 18. Muon、训练与推理：不要把不同层面的优化混在一起

【官方机制】V4 使用 Muon 优化器；V4.1 延续并调整训练方案，矩阵参数、归一化参数和大记忆表不都使用同一种更新方式。[S1，§2.4][S2，§2.5]

【教学解释】可以把模型结构理解为“机器如何连接”，优化器是“训练时如何修改零件”。Muon 不像 Attention 那样成为每次生成 Token 都要执行的一层。训练改进可能得到更好的权重，但不能把它直接填进推理缓存计算式。

同理，基准提升还与训练数据、后训练环境、推理预算和 Agent 框架有关。仅凭模型结构图不能推导某项基准一定提升多少。

## 19. 部署理解：小激活、低 KV，不等于模型很容易装下

### 19.1 权重、缓存、工作区分别估

$$M_{\mathrm{total}}\approx M_{\mathrm{weights}}+M_{\mathrm{KV}}+M_{\mathrm{workspace}}+M_{\mathrm{runtime}}.$$

【教学估计】假设“所有参数统一 1 byte”，V4-Flash 的 284B 对应 284GB 十进制；V4.1 若把 552B 主干与 196B Engram 合计，则约 748GB。**真实权重混合精度、表存放位置和元数据使实际值不同。** 这个估计只说明总存储与激活规模不能混用。

### 19.2 在 L20 一类环境里先核对什么？

重点是以下相互独立的问题：权重如何分片、Engram 是否卸载与预取、推理引擎是否实现 CED／CSA2、KV 量化布局是否受支持、SWA 回放和 DSpark 是否实际启用，以及 GPU 互联是否承受专家通信。

能保存 FP4 不等于该 GPU 对所有 FP4 算子都有原生高速支持；缓存解量化与专家矩阵乘也不是同一件事。本文不提供未经实际环境验证的“一条命令即可跑满百万上下文”方案。

### 19.3 最有用的测量项

| 测量 | 对应判断 |
|---|---|
| 长输入首次响应时间 | CED／Prefill 收益 |
| 每用户输出速度、总输出吞吐 | Decode 与调度效率 |
| 独立会话／共享前缀的显存差别 | 有效缓存与前缀复用 |
| SSD 恢复延迟、尾部回放耗时 | 持久缓存收益 |
| DSpark 接受率与实际吞吐 | 投机是否值得启用 |
| 专家通信、主机内存传输 | EP 与 Engram 瓶颈 |

## 20. 一条“上海旅行”请求的完整数据路径

【教学串讲】用户输入“安排上海两日游，预算 1000 元，晚上想看外滩夜景”，还附一张地图。

1. 文本变成 Token Embedding；地图经视觉编码与投影变成视觉 Embedding。
2. 因果 Encoder 处理输入；Engram 按局部词组读取学习过的记忆，再由上下文门控决定注入强度。
3. CSA2 的来源层建立全局缓存，其他层按模式共享或重选历史位置；局部窗口保留最近约束。
4. Encoder 末层表示为 Decoder 准备全局 KV；提示词尾部回放准备局部状态。
5. 生成时，各层 Query 从共享记忆中取信息；MoE 加工当前 Token 表示，mHC 传递并混合跨层信息。
6. 预测头产生词表概率；若启用 DSpark，先提草稿，再按验证规则接受输出。
7. 新 Token 继续进入因果计算；长前缀缓存可供后续轮次复用。

这些模块不会真的各自显式识别“预算”“夜景”标签；那只是人类对所处理信息的解释。

## 21. 读懂后的自测问题

| 问题 | 应抓住的答案 |
|---|---|
| CSA 与 HCA 的差异？ | 较细压缩后选择，与重压缩后浏览全局 |
| 稀疏读取等于只存 Top-K？ | 不等于，下一次查询可能选择别的位置 |
| CSA2 Full 等于 full attention？ | 不等于，是生成 KV 与索引的完整模式 |
| Reuse 层有没有自己的计算？ | 有自己的主 Query、SWA 和后续子层 |
| CED Encoder 是双向的吗？ | 不是，文本主干保持因果约束 |
| 8B 与 16B 是两套独立权重吗？ | 是阶段激活口径，不能由此推断只装 8B／16B |
| Engram 与 KV cache 区别？ | 前者是模型参数记忆，后者是请求上下文状态 |
| 890 bytes 包括所有显存吗？ | 不包括，是全局 KV 特定口径 |
| bounded replay 精确恢复所有状态吗？ | 是近似折中，论文报告影响很小 |
| DSpark 总能快五倍吗？ | 不能，受接受率和系统成本制约 |

## 22. 原始资料与建议阅读顺序

以下均为官方作者资料或实现方原始文档。正文编号可直接对应链接；配置链接指向 `main`，后续可能变化，复现实验应记录具体 commit。

1. **[S1] [DeepSeek-V4 技术报告](https://arxiv.org/html/2606.19348v1)**：先看 §2 架构，再看 §4.2.1 两种模型配置。题名：*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*。
2. **[S2] [DeepSeek-V4.1 技术报告](https://arxiv.org/html/2609.19969v1)**：先看 Figure 3，再看 §2.2 CED、§2.3 CSA2、§2.4 各扩展与 §3.2 缓存恢复。题名：*DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*。
3. **[S3] [V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)**：核对主干、Engram、激活口径及多模态入口。
4. **[S4] [V4-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)**：核对 Flash／Pro 与 Base／后训练版本差异。
5. **[S5] [V4-Flash-0731 官方模型卡](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/)**：不要把初版与后续投机模块版本混为一谈。
6. **[S6] [V4.1-Flash-1017 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**：本次核实的后续小版本；不据基准变化擅自推断主干重构。
7. **[S7] [V4-Flash config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json)**：逐项查层数、头数、压缩率、专家及辅助层。
8. **[S8] [V4.1-Flash config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)**：重点查 `text_config`、缓存／索引来源层、Engram、DSpark 与 `vision_config`。
9. **[S9] [Transformers 官方 V4 实现说明](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md)**：把论文术语对应到模块名、Hash 路由和缓存实现。
10. **[S10] [V4.1 官方参考实现 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py)**：注意 `ModelArgs` 默认值是小模型示例，不能当作发布模型的实际规模。
11. **[S11] [DeepSeek 官方更新记录](https://api-docs.deepseek.com/updates/)**：查询当前服务版本和路由；早期发布公告的后续安排可能已经修订。

建议学习路线：**标准 Attention／FFN → SWA → CSA／HCA → MoE／mHC → CED → CSA2 共享 → 分层索引 → Engram → DSpark → 实际缓存与部署。** 每一步都先回答“省计算、存储还是传输”，再看它付出了什么代价。
