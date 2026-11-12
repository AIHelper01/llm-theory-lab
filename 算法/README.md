# 算法 · 大模型知识系统梳理

[返回项目首页](../README.md)

围绕 **Transformer 全流程 → 核心模块原理 → 推理优化与技术演进 → 代表模型架构** 整理的中文学习笔记。

关注每项技术改动了哪个模块、解决什么问题、数学计算如何变化，以及收益和代价。内容包括视频关键画面解读、公式推导与手算、可运行数学复算、架构对比和原始资料索引。

## 当前覆盖范围

已覆盖从文本输入到自回归生成的核心计算链：

```text
文本 → 分词 → Token ID → Embedding → 位置机制
     → Attention → 残差与归一化 → FFN / MoE → 堆叠多层
     → 输出投影 → 概率与采样 → 追加 Token → 继续生成
```

不同模型的模块顺序与实现可能不同，上图用于建立整体联系。训练基础已涉及目标右移、交叉熵、梯度传播及参数更新；完整训练工程、量化、分布式系统和训练后对齐尚未形成独立系列。

## 第一次怎么读

| 阶段 | 首读 | 选读 |
| --- | --- | --- |
| 01–02 · 整体流程 | [Transformer 全流程](01-技术前要-深度学习基础/Transformer全流程/Transformer_从输入到自回归生成.md) | [从机器学习到深度学习](01-技术前要-深度学习基础/机器学习与深度学习/从机器学习到深度学习_B站解读.md) · [原论文解读](02-技术起源-Transformer/Attention_Is_All_You_Need.md) |
| 03–04 · 输入表示 | [BPE](03-技术拆解-分词与向量化/BPE分词/BPE分词_从相邻合并到Embedding.md) → [Embedding](03-技术拆解-分词与向量化/Embedding/Token与Embedding_从strawberry计数到向量查表.md) → [RoPE 基础](04-技术拆解-位置编码优化/RoPE基础/RoPE_从二维旋转到相对位置点积.md) | [正弦位置编码](04-技术拆解-位置编码优化/正弦位置编码/正弦位置编码_从序列编号到多频率坐标.md) · [YaRN](04-技术拆解-位置编码优化/YaRN/YaRN_B站解读_从RoPE分频缩放到注意力温度.md) |
| 05 · 注意力 | [QKV](05-技术拆解-注意力机制/01-QKV基础/01-QKV_从三种投影到注意力读取.md) → [多头注意力](05-技术拆解-注意力机制/02-多头注意力/02-多头注意力_从独立投影到拼接融合.md) → [KV Cache 基础](05-技术拆解-注意力机制/03-KVCache基础/03-KVCache基础_从因果不变性到Prefill与Decode.md) | [Prompt Cache 番外](05-技术拆解-注意力机制/03-KVCache基础/03-番外_PromptCache与跨请求前缀复用.md) · [MQA、GQA、MLA](05-技术拆解-注意力机制/05-KVCache进阶与MLA/05-KVCache_MQA_GQA_MLA_从显存公式到低秩吸收.md) |
| 06–09 · 层内计算与输出 | [残差连接](06-技术拆解-残差链接/残差连接_从ResNet到Transformer.md) → [归一化](07-技术拆解-归一化/归一化_从LayerNorm到RMSNorm.md) → [FFN](08-技术拆解-FNN层/FFN/FFN_从非线性特征到SwiGLU与参数记忆.md) → [输出与采样](09-技术拆解-输出/输出层与采样_从Logits到Temperature与TopP.md) | [MoE](08-技术拆解-FNN层/MoE/MoE_从稠密FFN到稀疏路由与负载均衡.md) |
| 10–11 · 架构与服务 | [哈佛 LLM Serving Basics 逐页解读](11-哈佛课程解读/llm_serving_basics_1/逐页解读.md) | [Qwen 架构](10-技术代表/Qwen3.8-Flash-Next_架构深度解读.md) · [DeepSeek 架构](10-技术代表/DeepSeek-V4_V4.1_架构深度解读.md) |

## 跑一遍最小 Decoder-only 生成

在 `算法` 目录运行 [decoder_demo.py](01-技术前要-深度学习基础/Transformer全流程/decoder_demo.py)，只需要 Python 标准库。

```bash
python "01-技术前要-深度学习基础/Transformer全流程/decoder_demo.py"
```

脚本用固定教学权重和 5 个字符的词表演示单层、单头模型。输入「我爱」先转成 Token ID，再经过 Embedding、RoPE、因果注意力、残差、RMSNorm 和 FFN。Prefill 写入提示词的 K/V，并用最后位置的 logits 采样第一个新 Token。随后每个 Decode 步只处理刚采样的 Token，追加一项 K/V，再采样下一项。

终端会打印 logits、采样概率、Token 和缓存长度。脚本逐步将缓存解码的 logits 与整段重算比较，并检查缓存长度。

## 内容导航

### 1. 整体流程与核心模块

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| 深度学习基础 · Transformer 全流程 | [Transformer：从输入到自回归生成](01-技术前要-深度学习基础/Transformer全流程/Transformer_从输入到自回归生成.md) | 35 张视频截图串联 Encoder–Decoder、交叉注意力、目标右移、损失、生成循环、KV cache 与 Decoder-only |
| Transformer 基础 | [Attention Is All You Need](02-技术起源-Transformer/Attention_Is_All_You_Need.md) | 从 Embedding、Attention、FFN 到 Encoder–Decoder 与 Decoder-only |
| 分词与向量化 · BPE | [BPE 分词：从相邻合并到 Embedding](03-技术拆解-分词与向量化/BPE分词/BPE分词_从相邻合并到Embedding.md) | 23 张视频截图、相邻对统计与三轮合并手算、字节级编码、Token ID、Embedding 及可运行教学代码 |
| 分词与向量化 · Embedding | [Token 与 Embedding：从 strawberry 计数到向量查表](03-技术拆解-分词与向量化/Embedding/Token与Embedding_从strawberry计数到向量查表.md) | 27 张视频截图、ID 与向量的区别、查表及梯度手算、字符计数、词表容量与代码核对 |
| 注意力基础 · QKV | [Q、K、V：从三种投影到注意力读取](05-技术拆解-注意力机制/01-QKV基础/01-QKV_从三种投影到注意力读取.md) | 31 张视频截图、投影与点积手算、softmax、Value 汇总、K / V 受控对比及因果矩阵验证 |
| 注意力基础 · 多头 | [多头注意力：从独立投影到拼接融合](05-技术拆解-注意力机制/02-多头注意力/02-多头注意力_从独立投影到拼接融合.md) | 34 张视频截图、双头手算、投影后拆头、输出融合、张量形状与参数预算 |
| 位置编码基础 · 正弦 | [从序列编号到多频率坐标](04-技术拆解-位置编码优化/正弦位置编码/正弦位置编码_从序列编号到多频率坐标.md) | 24 张视频截图、位置相加、正余弦公式、坐标手算、位移矩阵与外推边界 |
| 位置编码基础 · RoPE | [从二维旋转到相对位置点积](04-技术拆解-位置编码优化/RoPE基础/RoPE_从二维旋转到相对位置点积.md) | 24 张视频截图、旋转矩阵、相对位移推导、多频率、KV 缓存及长上下文边界 |
| 位置编码 · B 站推导 | [为什么不用序列编号](04-技术拆解-位置编码优化/正弦位置编码/正弦位置编码_B站推导_为什么不用序列编号.md) → [从旋转矩阵到相对位置](04-技术拆解-位置编码优化/RoPE基础/RoPE_B站推导_从旋转矩阵到相对位置.md) | 算法魔法师的两期视频，分别推导正弦编码的位移矩阵与 RoPE 的相对位置点积，共 17 张截图 |
| 位置编码 · YaRN | [RoPE 分频缩放与注意力温度](04-技术拆解-位置编码优化/YaRN/YaRN_B站解读_从RoPE分频缩放到注意力温度.md) | 6 张视频截图、NTK-by-parts、温度缩放及 4K 到 128K 的实验条件 |
| 深度学习基础 · 激活函数 | [激活函数：从非线性到 SwiGLU](01-技术前要-深度学习基础/激活函数/激活函数_从非线性到SwiGLU.md) | 两条视频的 30 张截图、函数与梯度手算、论文作者、FFN / SwiGLU 结构与计算预算 |
| FFN 原理 | [FFN：从非线性特征到 SwiGLU 与参数记忆](08-技术拆解-FNN层/FFN/FFN_从非线性特征到SwiGLU与参数记忆.md) | 26 张视频截图、升降维与矩阵手算、几何解释、SwiGLU、参数预算与记忆解释 |
| 残差连接 | [残差连接：从 ResNet 到 Transformer](06-技术拆解-残差链接/残差连接_从ResNet到Transformer.md) | 作者与核心论文、退化问题、梯度推导、视频图示及 FFN / MoE 中的残差结构 |
| 归一化 | [归一化：从 LayerNorm 到 RMSNorm](07-技术拆解-归一化/归一化_从LayerNorm到RMSNorm.md) | 作者与核心论文、BN / LN / RMSNorm、公式手算、梯度及 Pre-Norm / Post-Norm |
| 归一化 · B 站解读 | [从 BN 到 LN 与 RMSNorm](07-技术拆解-归一化/归一化_B站解读_从BN到LN与RMSNorm.md) | 10 张视频截图、归一化统计轴、LN 与 RMSNorm 手算，以及 BN 优化机制的论文讨论 |
| 深度学习基础 · 梯度传播 | [梯度消失与爆炸：从链式法则到残差连接](01-技术前要-深度学习基础/梯度消失与爆炸/梯度消失与爆炸_从链式法则到残差连接.md) | 22 张视频截图、梯度连乘、神经元与矩阵反传手算、Sigmoid / ReLU、初始化及残差路径 |
| 输出层与采样 | [从 Logits 到 Temperature、Top-k 与 Top-p](09-技术拆解-输出/输出层与采样_从Logits到Temperature与TopP.md) | 28 张视频截图、LM Head、softmax 与温度手算、过滤及抽样、随机种子与复现边界 |

### 2. 推理优化与技术演进

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| KV Cache 基础 | [KV Cache：从因果不变性到 Prefill / Decode](05-技术拆解-注意力机制/03-KVCache基础/03-KVCache基础_从因果不变性到Prefill与Decode.md) | 29 张视频截图、为什么只存 KV、历史表示不变的证明、99% 的计数口径与双层缓存等价验证 |
| KV Cache 番外 | [Prompt Cache 与跨请求前缀复用](05-技术拆解-注意力机制/03-KVCache基础/03-番外_PromptCache与跨请求前缀复用.md) | 9 张视频截图、前缀命中、工具定义与上下文压缩的取舍 |
| MHA、MQA 与 GQA | [从多头计算到 KV 共享](05-技术拆解-注意力机制/04-MHA-MQA-GQA/04-MHA-MQA-GQA_从多头计算到KV共享.md) | 8 张视频截图、查询头与 KV 头的共享方式、缓存容量算例 |
| KV Cache 专题 | [KV Cache、MQA、GQA 与 MLA](05-技术拆解-注意力机制/05-KVCache进阶与MLA/05-KVCache_MQA_GQA_MLA_从显存公式到低秩吸收.md) | 28 张视频截图、缓存容量手算、KV 共享、MLA 低秩吸收与解耦 RoPE、可运行数学复算 |
| MoE 原理 | [MoE：从稠密 FFN 到稀疏路由与负载均衡](08-技术拆解-FNN层/MoE/MoE_从稠密FFN到稀疏路由与负载均衡.md) | 30 张视频截图、Top-k 手算、负载均衡、细粒度与共享专家、参数及通信成本 |
| FFN 与专家模型 | [MoE 核心论文与技术演进](08-技术拆解-FNN层/MoE_核心论文与技术演进.md) | 专家路由、激活参数、负载均衡及相关论文 |
| 位置编码 | [位置编码论文与技术演进](04-技术拆解-位置编码优化/Transformer_位置编码论文与技术演进.md) | Sinusoidal、相对位置表示、T5、RoPE、ALiBi |
| RoPE 专题 | [RoPE 原理与 Qwen / DeepSeek 架构解读](04-技术拆解-位置编码优化/RoPE_原理与Qwen_DeepSeek架构解读.md) | 旋转位置编码的数学机制及架构应用 |
| 注意力机制 | [Attention 技术演进](05-技术拆解-注意力机制/06-Attention技术演进/06-Attention_技术演进_Transformer至Qwen3.8与DeepSeekV4.1.md) | MHA、MQA、GQA、FlashAttention、MLA、线性与稀疏注意力等 |
| 注意力机制 · 视频补充 | [百万上下文与注意力演进](05-技术拆解-注意力机制/06-Attention技术演进/06-视频补充_百万上下文与注意力演进.md) | 10 张视频截图，比较 KV 头共享、MLA、DSA 与 CSA 对缓存和读取的影响 |

### 3. 代表模型与资料索引

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| Qwen | [Qwen 技术优化路线与官方资料索引](10-技术代表/Qwen_技术优化路线与官方资料索引.md) | 从模块改动理解模型架构，附原始资料入口 |
| DeepSeek | [DeepSeek 技术优化路线与官方资料索引](10-技术代表/DeepSeek_技术优化路线与官方资料索引.md) | 注意力、KV 压缩、MoE 等优化路线与资料入口 |
| Qwen 架构专题 | [Qwen3.8-Flash-Next 架构深度解读](10-技术代表/Qwen3.8-Flash-Next_架构深度解读.md) | 整体结构、模块公式、手算示例与部署含义 |
| DeepSeek 架构专题 | [DeepSeek V4 / V4.1 架构深度解读](10-技术代表/DeepSeek-V4_V4.1_架构深度解读.md) | 分代架构对比、模块机制、计算示例与部署含义 |

## 目录结构

```text
.
├── README.md
├── 01-技术前要-深度学习基础/
│   ├── 激活函数/
│   │   ├── 激活函数_从非线性到SwiGLU.md
│   │   └── assets/              # 两条参考视频的 30 张关键画面
│   ├── 梯度消失与爆炸/
│   │   ├── 梯度消失与爆炸_从链式法则到残差连接.md
│   │   └── assets/              # 22 张参考视频画面
│   └── Transformer全流程/
│       ├── Transformer_从输入到自回归生成.md
│       └── assets/              # 35 张参考视频画面
├── 02-技术起源-Transformer/
├── 03-技术拆解-分词与向量化/
│   ├── BPE分词/
│   │   ├── BPE分词_从相邻合并到Embedding.md
│   │   ├── bpe_toy.py           # 仅用标准库的手算复现
│   │   └── assets/              # 23 张参考视频画面
│   └── Embedding/
│       ├── Token与Embedding_从strawberry计数到向量查表.md
│       ├── embedding_math.py   # 查表、梯度与容量核对
│       └── assets/              # 27 张参考视频画面
├── 04-技术拆解-位置编码优化/
│   ├── 正弦位置编码/            # 独立视频解读、复算代码与 24 张截图
│   ├── RoPE基础/                # 独立视频解读、复算代码与 24 张截图
│   └── YaRN/                    # RoPE 长度扩展与注意力温度
├── 05-技术拆解-注意力机制/
│   ├── 01-QKV基础/
│   ├── 02-多头注意力/
│   ├── 03-KVCache基础/
│   ├── 04-MHA-MQA-GQA/
│   ├── 05-KVCache进阶与MLA/
│   └── 06-Attention技术演进/
├── 06-技术拆解-残差链接/
├── 07-技术拆解-归一化/
├── 08-技术拆解-FNN层/
│   ├── MoE_核心论文与技术演进.md
│   ├── FFN/                    # FFN 视频解读、复算代码与 26 张截图
│   └── MoE/                    # MoE 视频解读、复算代码与 30 张截图
├── 09-技术拆解-输出/
│   ├── 输出层与采样_从Logits到Temperature与TopP.md
│   ├── sampling_math.py        # 概率、候选过滤与采样核对
│   └── assets/                 # 28 张参考视频画面
├── 10-技术代表/
└── 11-哈佛课程解读/
```

## 阅读方式

- **GitHub：** 通过上方导航直接浏览；图片使用仓库内相对路径。
- **本地：** 克隆后使用 Typora 或支持 LaTeX 公式的 Markdown 阅读器打开，无需安装项目依赖或执行构建。
- **公式与图示：** 部分笔记包含内嵌 HTML、SVG 和样式，GitHub 可能过滤部分展示效果；建议在支持这些格式的本地阅读器中查看。
- **图片素材：** 移动文档时，请同时保留对应的 `assets/` 目录，并修正相关文档链接。
- **数学复算：** 配套 `.py` 脚本仅使用 Python 标准库，进入脚本所在目录后执行 `python3 脚本名.py`。这些脚本核对教学数值与等价关系，不作为真实模型的性能或质量评测。

```bash
git clone https://github.com/AIHelper01/llm-theory-lab.git
cd llm-theory-lab
```

## 资料与使用说明

本仓库是学习整理，不是相关模型团队或课程机构的官方文档。论文、模型报告、公告与代码入口保留在各篇笔记中；引用与课件截图的权利归原作者或机构所有。

涉及模型版本、产品状态、性能数据和架构配置时，请结合文章标注的整理日期与原始资料阅读。性能倍数依赖硬件、精度、上下文长度、批次和测量方式；教学简化公式不能直接替代实际部署测试。

欢迎通过 Issue 指出公式、引用、链接或表述问题，并提供对应的原始资料与修正建议。
