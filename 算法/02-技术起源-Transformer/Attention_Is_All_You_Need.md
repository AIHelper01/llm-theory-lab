# 《Attention Is All You Need》：从数学计算到完整 Transformer 架构

Vaswani et al. · 2017 · Transformer 原始论文解读

> **Typora 阅读说明：**启用数学公式支持后阅读；正文使用 Markdown，视觉模块使用内嵌 HTML 与内联样式。原滑块以静态数值对照呈现。主题可能影响字体、边框和间距。

这篇论文非常适合用我们刚才讲 FFN 的方式来理解：**不只是介绍概念，而是让每一个模块都对应具体的输入向量、权重矩阵、计算过程和输出结果。**

论文在 2017 年提出 Transformer，核心突破是：不再依赖 RNN 的循环结构或 CNN 的卷积结构来完成序列信息交互，而是以 Attention 为主要机制构建 Encoder–Decoder 模型。

我们按照以下顺序展开：

1. 为什么要从 RNN、CNN 转向 Attention？
2. Embedding 和位置编码如何准备输入？
3. Self-Attention 究竟进行了哪些矩阵计算？
4. Multi-Head Attention 为什么要有多个头？
5. Attention 之后，FFN 如何继续加工特征？
6. Encoder 和 Decoder 如何协作完成翻译？
7. 为什么今天的 GPT 采用 Decoder-only 架构？

## 一、从 CNN 和 RNN 讲起：Transformer 解决了什么问题？

假设有一句话：

`The cat chased the mouse.`

我们的目标是让模型理解每个 Token 与其他 Token 的关系。

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">三种序列信息交互方式</div><p><b>RNN：按时间顺序传递状态</b><br>每个状态依赖前一步状态，因此存在顺序依赖。</p><hr style="border:0;border-top:1px solid #eee;"><p><b>CNN：通过卷积窗口聚合局部信息</b><br>可以并行计算，但固定大小的卷积核通常需要多层才能扩大信息交互范围。</p><hr style="border:0;border-top:1px solid #eee;"><p><b>Self-Attention：直接计算 Token 之间的相关性</b><br>每个位置可直接融合所有允许关注的位置的信息。</p></div>


这是原论文 Encoder 中的双向 Self-Attention 示例。因果 Decoder 需要遮挡未来 Token。

Self-Attention 的关键优势是：**任意两个允许交互的 Token 可以在同一层直接建立信息联系**，而且同一层不同位置的注意力计算能够并行执行。

这不意味着 Attention 没有计算成本。标准全注意力的注意力分数矩阵大小随序列长度呈二次增长；相关计算也具有二次项。

## 二、Embedding：先把 Token 转成向量

模型并不直接处理文字，而是处理数字。我们从三个 Token 开始：

`The cat chased`

为了方便手算，假设 Embedding 之后的向量是：

$$
x_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
x_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
x_3=\begin{bmatrix}1\\1\end{bmatrix}
$$

组合成一个矩阵（每行对应一个 Token）：

$$
X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
$$

| Token | 维度 1 | 维度 2 |
|---|---:|---:|
| The | 1 | 0 |
| cat | 0 | 1 |
| chased | 1 | 1 |

全部为教学用假设数值，真实模型的 Embedding 是训练得到的高维向量。

### 位置编码（Positional Encoding）

仅靠 Token Embedding，Attention 本身不知道谁在前、谁在后。所以原始 Transformer 在输入时加入位置编码：

$$X_{\text{input}}=Embedding+PositionalEncoding$$

原论文使用不同频率的正弦与余弦函数生成位置编码，让模型获得序列顺序信息：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

原论文还会将 Token Embedding 乘以 $\sqrt{d_{model}}$ 后再加位置编码。为了先理解 Attention 的计算，接下来的手算直接把前面的 $X$ 当作已经准备好的输入向量，暂不单独计算位置编码和该缩放步骤。

## 三、Self-Attention：最关键的 Q、K、V 矩阵计算

这是整篇论文最核心的数学公式：

$$\boxed{Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

看起来复杂，但实际上只有五步。

### 第一步：生成 Q、K、V

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$$

在真实 Transformer 中，$W_Q,W_K,W_V$ 都是通过训练学习到的权重矩阵。为了让我们专注于 Attention 的原理，先假设：

$$W_Q=W_K=W_V=\begin{bmatrix}1&0\\0&1\end{bmatrix}$$

那么：

$$Q=K=V=X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$$

这只是一个特意简化的数值例子，真实模型的 Q、K、V 通常并不相同。

### 第二步：计算 Token 之间的匹配分数

我们重点处理 `chased`。它的 Query 为：

$$q_3=[1,1]$$

其他 Token 的 Key 为：

$$k_1=[1,0],\quad k_2=[0,1],\quad k_3=[1,1]$$

现在分别做点积：

$$q_3k_1^T=1,\quad q_3k_2^T=1,\quad q_3k_3^T=2$$

得到匹配分数：

$$Scores=[1,1,2]$$

可以直观理解为：在这个假设的向量空间中，`chased` 与三个 Key 的点积分别为 1、1、2。

完整矩阵也可以一次算出来：

$$QK^T=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}$$

每一行代表一个 Query，每一列代表一个 Key。

### 第三步：Scaled Dot-Product

因为 Key 的维度是 2：

$$\sqrt{d_k}=\sqrt2\approx1.414$$

因此：

$$ScaledScores=[0.7071,\ 0.7071,\ 1.4142]$$

为什么要除以 $\sqrt{d_k}$？因为当向量维度增大时，点积数值可能变大，使 Softmax 输出过于集中，影响训练时的梯度。缩放能够缓解这个问题。

### 第四步：Softmax 得到 Attention Weights

$$A=Softmax([0.7071,0.7071,1.4142])$$

$$Softmax(s)_i=\frac{e^{s_i}}{\sum_j e^{s_j}}$$

得到近似结果：

$$A\approx[0.2483,\ 0.2483,\ 0.5035]$$

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">chased 对三个 Token 的注意力权重</div><div style="margin:12px 0;">The <b>0.2483</b><div style="height:14px;background:#f0f2f5;border-radius:7px;margin-top:6px;"><div style="width:24.83%;height:14px;background:#5a9fcd;border-radius:7px;"></div></div></div><div style="margin:12px 0;">cat <b>0.2483</b><div style="height:14px;background:#f0f2f5;border-radius:7px;margin-top:6px;"><div style="width:24.83%;height:14px;background:#5a9fcd;border-radius:7px;"></div></div></div><div style="margin:12px 0;">chased <b>0.5035</b><div style="height:14px;background:#f0f2f5;border-radius:7px;margin-top:6px;"><div style="width:50.35%;height:14px;background:#5a9fcd;border-radius:7px;"></div></div></div></div>


这就是 Attention 的核心：**根据 Q 和 K 的匹配程度，决定从各个 Token 的 V 中获取多少信息。**

注意，这个数值例子中 `chased` 对自身的权重最大，不代表真实模型永远更关注自身。实际的权重由训练学到的投影和上下文决定。显示值经过四舍五入，相加可能与 1 略有差异。

### 第五步：对 Value 加权求和

$$v_1=[1,0],\quad v_2=[0,1],\quad v_3=[1,1]$$

$$z_3=0.2483v_1+0.2483v_2+0.5035v_3$$

$$z_3\approx[0.2483+0+0.5035,\ 0+0.2483+0.5035]$$

$$\boxed{z_3\approx[0.7517,\ 0.7517]}$$

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">计算流程</div><table style="width:100%;border:0;border-collapse:separate;border-spacing:6px;background:white;"><tbody><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>输入 chased</b><br><span style="color:#667;">[1, 1]</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>匹配与归一化</b><br><span style="color:#667;">[0.2483, 0.2483, 0.5035]</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Value 加权求和</b><br><span style="color:#667;">[0.7517, 0.7517]</span></td></tr></tbody></table></div>


`chased` 原来的输入向量是 `[1, 1]`，经过 Attention 后，得到约 `[0.752, 0.752]`。这个新向量中，已经融入了 `The`、`cat` 和 `chased` 三个位置的 Value 信息。

这就是我们之前一直在说的“Attention 融合多个 Token 信息”，现在它不再是一个抽象的概念，而是一次实际的向量加权计算。

## 四、对照实验：自己改变 Attention 的计算结果

保留刚才的 Key 和 Value 向量，调整当前 Query 的两个数值，观察 Attention 权重与输出怎样变化。原对话中的滑块范围为 −2 到 2、步长 0.1；这里保留初始状态和一个修改后的状态。

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">Attention 实验台 · 2 维 · 3 Tokens</div><p>固定 K = V = [[1,0], [0,1], [1,1]]</p><table style="width:100%;text-align:left;border-spacing:8px;"><tr><th>Query</th><th>缩放分数</th><th>权重</th><th>输出</th></tr><tr><td>[1, 1]</td><td>[0.7071, 0.7071, 1.4142]</td><td>[0.2483, 0.2483, 0.5035]</td><td>[0.7517, 0.7517]</td></tr><tr><td>[2, 0]</td><td>[1.4142, 0, 1.4142]</td><td>[0.4458, 0.1084, 0.4458]</td><td>[0.8916, 0.5542]</td></tr></table></div>


你可以尝试将 Query 修改为 `[2, 0]`。此时它与 `The` 和 `chased` 的 Key 更匹配，与 `cat` 的 Key 相对没那么匹配，最终 Attention 权重也会发生变化。

这也是 Attention 的重要特征：**不同 Token 可以根据自己的 Query，动态选择需要从上下文中融合的信息。**

## 五、Multi-Head Attention：为什么不只用一个 Attention？

前面计算的是一个 Attention Head。但原始 Transformer Base 使用 8 个 Attention Heads。

为什么需要多个头？我们仍然使用：

`The cat chased the mouse.`

模型可以学习到多种不同的关系。例如：

- 一些头可能更关注动作和执行者之间的关系。
- 一些头可能更关注动作和宾语之间的关系。
- 一些头可能更关注局部位置或语法模式。

不过，不同 Head 的功能不是人为指定的，而且不保证每个 Head 都对应一种清晰的语言学关系。上述宾语关系可以出现在双向 Encoder 中；因果 Decoder 的 `chased` 位置不能关注后面的 `mouse`。

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">计算流程</div><table style="width:100%;border:0;border-collapse:separate;border-spacing:6px;background:white;"><tbody><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Input X</b><br><span style="color:#667;">同一输入</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Head 1 … Head 8</b><br><span style="color:#667;">不同的 Q/K/V 投影</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Concat</b><br><span style="color:#667;">拼接各头输出</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Linear Wₒ</b><br><span style="color:#667;">映射回模型维度</span></td></tr></tbody></table></div>


数学上：

$$head_i=Attention(XW_i^Q,XW_i^K,XW_i^V)$$

$$MultiHead=Concat(head_1,\ldots,head_h)W^O$$

原始 Transformer Base 的配置为：

| 参数 | 数值 |
|---|---:|
| Hidden Dimension | 512 |
| Attention Heads | 8 |
| 每个 Head 的 Q/K/V 维度 | 64 |
| 拼接后维度 | 512 |

$$8\times64=512$$

它的意义不是简单地让 Attention 计算八遍，而是让模型能够在不同的投影子空间中，同时学习不同的信息关联方式。

## 六、接下来就是 FFN：你前面已经掌握的部分

现在我们把 Multi-Head Attention 的输出交给 FFN。继续使用二维向量，不过为了便于计算，我们直接使用前面单个 Attention Head 算出的结果，并暂时省略残差、归一化及输出投影：

$$z=[0.752,\ 0.752]^T$$

这里切换为列向量写法，所以矩阵乘在向量左侧；前面 Attention 使用的是行向量约定，两种写法等价。

设定 FFN 第一层权重矩阵：

$$W_1=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

那么：

$$W_1z=\begin{bmatrix}1.504\\0\end{bmatrix}$$

经过 ReLU：

$$h=ReLU(W_1z)=\begin{bmatrix}1.504\\0\end{bmatrix}$$

再经过第二层权重：

$$W_2=\begin{bmatrix}0.5&0\\0&0.5\end{bmatrix}$$

输出：

$$FFN(z)=W_2h=\begin{bmatrix}0.752\\0\end{bmatrix}$$

Attention 已经把三个 Token 的信息融合成 `[0.752, 0.752]`。FFN 又通过权重矩阵和非线性激活，把这两个特征转换成 `[0.752, 0]`。

这里 FFN 的两个矩阵也是教学假设，不代表真实模型一定会把第二个特征清零。

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">计算流程</div><table style="width:100%;border:0;border-collapse:separate;border-spacing:6px;background:white;"><tbody><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Attention</b><br><span style="color:#667;">[0.752, 0.752] · 融合 Token 信息</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Linear → ReLU → Linear</b><br><span style="color:#667;">非线性特征变换</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>FFN 输出</b><br><span style="color:#667;">[0.752, 0]</span></td></tr></tbody></table></div>


原始论文中，FFN 的中间层维度从 512 扩展到 2048，再映射回 512：

$$512\rightarrow2048\rightarrow512$$

使用论文的行向量约定：

$$FFN(x)=\max(0,xW_1+b_1)W_2+b_2$$

这就是 Transformer 中 Attention 与 FFN 的分工。FFN 对每个 Token 分别执行相同的变换，**不在此步骤直接混合不同 Token**。

### 还不能忽略 Residual + LayerNorm

原始 Transformer 使用 Post-Norm：

$$Output=LayerNorm(x+Sublayer(x))$$

它会应用于 Attention 和 FFN 子层周围。作用可以简单理解为：

- Residual：保留原输入，同时加入模块学到的新信息。
- LayerNorm：对特征进行归一化处理，帮助稳定网络训练。

因此，真实的 Encoder Layer 并不是单纯地将 Attention 的输出直接传入 FFN。完整计算还包括残差连接和归一化操作。原论文在子层输出加入残差前还使用 Dropout，教学示意中省略。

## 七、将所有模块组合成完整 Encoder

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">Encoder · 完整结构</div><div style="text-align:center;line-height:2;">Input Tokens：The cat chased the mouse<br>↓<br>Embedding + Positional Encoding<br>↓<div style="border:2px solid #8bb8d1;background:#f4f9fc;border-radius:12px;padding:14px;"><b>Encoder Layer × 6</b><br>Multi-Head Self-Attention：跨 Token 信息融合<br>↓<br>Add &amp; Norm<br>↓<br>Feed-Forward Network：每个 Token 内部的非线性特征变换<br>↓<br>Add &amp; Norm</div>↓<br><b>Encoder Output</b><br>包含上下文信息的 Token 表示序列</div></div>


原始 Transformer Base 包含 6 层 Encoder，每层都有 Multi-Head Attention 和 FFN。

在 Encoder 中，每个 Token 都可以关注输入序列中的其他 Token，因此它属于双向上下文建模（实际使用时还会遮挡 Padding 等无效位置）。

## 八、Decoder：它为什么比 Encoder 多一个 Attention？

原始 Transformer 是为机器翻译等序列到序列任务设计的，因此包括 Encoder 和 Decoder 两个部分。例如：

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">机器翻译：Encoder 与 Decoder</div><table style="width:100%;border-collapse:separate;border-spacing:8px;"><tbody><tr><td style="background:#eef5fc;padding:18px;border:0;border-radius:10px;"><b>Encoder</b><p>输入：I love AI.</p><p>Self-Attention → FFN（含 Add &amp; Norm）</p>生成原文的上下文表示</td></tr><tr><td style="text-align:center;border:0;background:white;">↓ Encoder 输出作为 Cross-Attention 的 K、V 来源</td></tr><tr><td style="background:#edf7f2;padding:18px;border:0;border-radius:10px;"><b>Decoder</b><p>输入：已生成的中文 Token</p>Masked Self-Attention<br>↓ Add &amp; Norm<br>Cross-Attention ← Encoder 输出<br>↓ Add &amp; Norm<br>FFN<br>↓ Add &amp; Norm<p>预测下一个中文 Token</p></td></tr></tbody></table></div>


Decoder 中有两种不同的 Attention。

### 1. Masked Self-Attention：只关注已经生成的内容

假设目标句子是：

`我 喜欢 人工智能`

在自回归生成时，处理 `喜欢` 所在的位置，不能提前读取后面的 `人工智能`。用矩阵可以直观表示 Mask：

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">Causal Attention Mask</div><table style="width:100%;border-spacing:5px;border-collapse:separate;text-align:center;border:0;table-layout:fixed;margin:0;background:white;"><tr><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td></tr><tr><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td></tr><tr><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#f9ecec;color:#ed2727;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">×</td></tr><tr><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td><td style="background:#d8ede6;color:#00a44d;border-radius:9px;height:46px;font-size:20px;border:0;padding:0;text-align:center;vertical-align:middle;">✓</td></tr></table><div style="margin-top:10px;"><span style="color:#00a44d;">✓</span> 允许关注　 <span style="color:#ed2727;">×</span> 未来位置被遮挡</div></div>


行表示当前 Query 位置，列表示 Key 位置。第 1 个位置只能关注自己，第 4 个位置可以关注此前全部位置及自身。

数学上，把未来位置的 Attention 分数设置为 $-\infty$，再执行 Softmax。这些位置对应的注意力权重就变为 0：

$$Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V$$

$$M=\begin{bmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{bmatrix}$$

训练时，Decoder 的目标输入会向右移一位，例如输入 `<BOS> 我 喜欢`，分别预测 `我 喜欢 人工智能`。遮罩让各位置只能利用相应的前缀信息。

### 2. Cross-Attention：读取 Encoder 的信息

这是原始 Decoder 与大多数纯文本 Decoder-only LLM 的重要区别。Cross-Attention 的 Q、K、V 来源不同：

$$Q=X_{\text{Decoder}}W_Q$$

$$K=X_{\text{Encoder}}W_K,\quad V=X_{\text{Encoder}}W_V$$

可以理解为：

- Decoder 的 Query：我现在需要从原文中获取什么信息？
- Encoder 的 Key：原文各位置有哪些信息可供匹配？
- Encoder 的 Value：原文各位置实际提供哪些信息？

例如预测“人工智能”时，Decoder 可以通过 Cross-Attention 关注原文的 `AI`。

**Masked Self-Attention 负责利用已经生成的目标文本，Cross-Attention 负责读取源文本。**

## 九、最后如何预测下一个 Token？

在经过多层 Decoder 处理后，模型会得到最后一个位置的隐藏向量。接下来：

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">计算流程</div><table style="width:100%;border:0;border-collapse:separate;border-spacing:6px;background:white;"><tbody><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Decoder Hidden State</b><br><span style="color:#667;">最后位置的隐藏向量</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Linear</b><br><span style="color:#667;">投影到词表维度</span></td></tr><tr><td style="border:0;background:white;text-align:center;padding:0;color:#8795a1;">↓</td></tr><tr><td style="border:1px solid #dce6ed;border-radius:10px;padding:14px;text-align:center;background:#eef4fa;line-height:1.7;"><b>Softmax</b><br><span style="color:#667;">生成概率分布</span></td></tr></tbody></table></div>


| 候选 Token | 示例概率 |
|---|---:|
| 人工智能 | 70% |
| 技术 | 20% |
| 其他候选合计 | 10% |

以上是人为设置的概率，仅展示输出层机制。实际 Token 划分由分词器决定，“人工智能”在这里被简化为一个 Token。

模型依据概率分布选择或采样下一个 Token，然后继续生成，直到遇到结束标记或其他停止条件。

## 十、原始 Transformer 与今天 GPT 的关系

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">原始 Transformer → Decoder-only</div><p><b>原始论文：</b>Encoder（双向 Self-Attention）＋ Decoder（Masked Self-Attention ＋ Cross-Attention）</p><hr style="border:0;border-top:1px solid #eee;"><p><b>GPT 的 Decoder-only 思路：</b>Causal Self-Attention → FFN，多层重复；不保留原始 Encoder–Decoder Cross-Attention。</p></div>


GPT 并不是直接把原始 Transformer 的右半边原封不动地拿过来，而是采用以因果 Self-Attention 为核心的 Decoder-only 架构，并在后续模型中持续演进。

这意味着它不保留原始 Encoder–Decoder 之间的 Cross-Attention 子层；残差连接、归一化、输入输出模块仍然存在，具体实现也可能不同。

## 十一、论文中的关键参数

| 参数 | Transformer Base |
|---|---|
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Hidden Dimension | 512 |
| Attention Heads | 8 |
| Head Dimension | 64 |
| FFN Intermediate Dimension | 2048 |
| Positional Encoding | Sinusoidal（论文也比较了可学习位置嵌入） |
| Attention 类型 | Scaled Dot-Product |
| 主要归一化形式 | Post-Norm |

这些参数对应原始论文的 Base 配置。

## 十二、把整篇论文浓缩成一个计算流程

<div style="border:1px solid #e7e7e7;border-radius:18px;padding:18px;margin:20px 0;background:#fff;color:#24272b;font-family:Arial,sans-serif;"><div style="font-weight:600;margin-bottom:12px;">完整计算流程</div><div style="text-align:center;line-height:2;">文本输入<br>↓<br>Embedding + Positional Encoding<br>↓<div style="border:1px solid #c5d9e7;border-radius:12px;background:#f5f9fc;padding:16px;"><b>Transformer Layer</b><br>Attention：通过 QKᵀ / √dₖ 计算权重，再融合 V<br>↓<br>Residual + LayerNorm<br>↓<br>FFN：Linear → ReLU → Linear<br>↓<br>Residual + LayerNorm</div>↓ 多层重复<br>Output Projection + Softmax</div><p style="color:#667;font-size:13px;">原论文的基本子层示意；Decoder 还包含 Cross-Attention，完整序列到序列结构见第八节。</p></div>


到这里，应该可以建立完整的理解：

- Embedding 负责把离散 Token 变为可计算向量。
- Positional Encoding 注入顺序信息。
- Attention 计算不同 Token 的匹配程度，动态融合信息。
- Multi-Head Attention 从多个投影子空间学习关系。
- FFN 对每个 Token 的上下文特征进行非线性变换。
- Residual 和 LayerNorm 帮助信息传递与稳定训练。
- Encoder 形成输入序列表示，Decoder 利用这些表示进行自回归生成。

**《Attention Is All You Need》的标题并不是说模型只需要 Attention、不需要 FFN，而是指不再需要依赖 RNN 或 CNN 来完成序列建模。**

这也是整篇论文最值得记住的创新。

原始资料：[Attention Is All You Need（论文）](https://arxiv.org/abs/1706.03762) · [The Illustrated Transformer（图解教程）](https://jalammar.github.io/illustrated-transformer/)
