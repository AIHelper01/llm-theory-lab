# LLM Theory Lab · 大模型知识系统梳理

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

## 建议阅读顺序

1. **先看整体：** 从《Transformer：从输入到自回归生成》建立完整流程印象，再对照原始 Transformer 论文解读。
2. **拆开输入与注意力：** BPE → Token / Embedding → QKV → 多头注意力 → 正弦位置编码 → RoPE。
3. **理解层内计算与训练基础：** 激活函数 → FFN → 残差连接 → 归一化 → 梯度消失与爆炸。
4. **接到生成与稀疏计算：** 输出层与采样 → KV Cache 基础 → MQA / GQA / MLA → MoE。
5. **进入技术演进与具体架构：** 阅读位置编码、Attention、MoE 的演进综述，再对照 Qwen 与 DeepSeek 架构及官方资料。

**基础篇**以视频画面、公式和小算例解释机制；**演进综述**比较不同方法；**模型专题**将模块映射到具体架构。遇到重复概念时，可按当前问题选择对应层次阅读。

## 内容导航

### 1. 整体流程与核心模块

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| 深度学习基础 · Transformer 全流程 | [Transformer：从输入到自回归生成](技术前要-深度学习基础/Transformer全流程/Transformer_从输入到自回归生成.md) | 35 张视频截图串联 Encoder–Decoder、交叉注意力、目标右移、损失、生成循环、KV cache 与 Decoder-only |
| Transformer 基础 | [Attention Is All You Need](技术起源-Transformer/Attention_Is_All_You_Need.md) | 从 Embedding、Attention、FFN 到 Encoder–Decoder 与 Decoder-only |
| 分词与向量化 · BPE | [BPE 分词：从相邻合并到 Embedding](技术拆解-分词与向量化/BPE分词/BPE分词_从相邻合并到Embedding.md) | 23 张视频截图、相邻对统计与三轮合并手算、字节级编码、Token ID、Embedding 及可运行教学代码 |
| 分词与向量化 · Embedding | [Token 与 Embedding：从 strawberry 计数到向量查表](技术拆解-分词与向量化/Embedding/Token与Embedding_从strawberry计数到向量查表.md) | 27 张视频截图、ID 与向量的区别、查表及梯度手算、字符计数、词表容量与代码核对 |
| 注意力基础 · QKV | [Q、K、V：从三种投影到注意力读取](技术拆解-注意力机制/QKV基础/QKV_从三种投影到注意力读取.md) | 22 张视频截图、投影与点积手算、softmax、Value 汇总、K / V 受控对比及因果矩阵验证 |
| 注意力基础 · 多头 | [多头注意力：从独立投影到拼接融合](技术拆解-注意力机制/多头注意力/多头注意力_从独立投影到拼接融合.md) | 22 张视频截图、双头手算、投影后拆头、输出融合、张量形状与参数预算 |
| 位置编码基础 · 正弦 | [从序列编号到多频率坐标](技术拆解-位置编码优化/正弦位置编码/正弦位置编码_从序列编号到多频率坐标.md) | 24 张视频截图、位置相加、正余弦公式、坐标手算、位移矩阵与外推边界 |
| 位置编码基础 · RoPE | [从二维旋转到相对位置点积](技术拆解-位置编码优化/RoPE基础/RoPE_从二维旋转到相对位置点积.md) | 24 张视频截图、旋转矩阵、相对位移推导、多频率、KV 缓存及长上下文边界 |
| 深度学习基础 · 激活函数 | [激活函数：从非线性到 SwiGLU](技术前要-深度学习基础/激活函数/激活函数_从非线性到SwiGLU.md) | 两条视频的 30 张截图、函数与梯度手算、论文作者、FFN / SwiGLU 结构与计算预算 |
| FFN 原理 | [FFN：从非线性特征到 SwiGLU 与参数记忆](技术拆解-FNN层/FFN/FFN_从非线性特征到SwiGLU与参数记忆.md) | 26 张视频截图、升降维与矩阵手算、几何解释、SwiGLU、参数预算与记忆解释 |
| 残差连接 | [残差连接：从 ResNet 到 Transformer](技术拆解-残差链接/残差连接_从ResNet到Transformer.md) | 作者与核心论文、退化问题、梯度推导、视频图示及 FFN / MoE 中的残差结构 |
| 归一化 | [归一化：从 LayerNorm 到 RMSNorm](技术拆解-归一化/归一化_从LayerNorm到RMSNorm.md) | 作者与核心论文、BN / LN / RMSNorm、公式手算、梯度及 Pre-Norm / Post-Norm |
| 深度学习基础 · 梯度传播 | [梯度消失与爆炸：从链式法则到残差连接](技术前要-深度学习基础/梯度消失与爆炸/梯度消失与爆炸_从链式法则到残差连接.md) | 22 张视频截图、梯度连乘、神经元与矩阵反传手算、Sigmoid / ReLU、初始化及残差路径 |
| 输出层与采样 | [从 Logits 到 Temperature、Top-k 与 Top-p](技术拆解-输出/输出层与采样_从Logits到Temperature与TopP.md) | 28 张视频截图、LM Head、softmax 与温度手算、过滤及抽样、随机种子与复现边界 |

### 2. 推理优化与技术演进

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| KV Cache 基础 | [KV Cache：从因果不变性到 Prefill / Decode](技术拆解-注意力机制/KVCache/KVCache基础_从因果不变性到Prefill与Decode.md) | 29 张视频截图、为什么只存 KV、历史表示不变的证明、99% 的计数口径与双层缓存等价验证 |
| KV Cache 专题 | [KV Cache、MQA、GQA 与 MLA](技术拆解-注意力机制/KVCache/KVCache_MQA_GQA_MLA_从显存公式到低秩吸收.md) | 28 张视频截图、缓存容量手算、KV 共享、MLA 低秩吸收与解耦 RoPE、可运行数学复算 |
| MoE 原理 | [MoE：从稠密 FFN 到稀疏路由与负载均衡](技术拆解-FNN层/MoE/MoE_从稠密FFN到稀疏路由与负载均衡.md) | 30 张视频截图、Top-k 手算、负载均衡、细粒度与共享专家、参数及通信成本 |
| FFN 与专家模型 | [MoE 核心论文与技术演进](技术拆解-FNN层/MoE_核心论文与技术演进.md) | 专家路由、激活参数、负载均衡及相关论文 |
| 位置编码 | [位置编码论文与技术演进](技术拆解-位置编码优化/Transformer_位置编码论文与技术演进.md) | Sinusoidal、相对位置表示、T5、RoPE、ALiBi |
| RoPE 专题 | [RoPE 原理与 Qwen / DeepSeek 架构解读](技术拆解-位置编码优化/RoPE_原理与Qwen_DeepSeek架构解读.md) | 旋转位置编码的数学机制及架构应用 |
| 注意力机制 | [Attention 技术演进](技术拆解-注意力机制/Attention_技术演进_Transformer至Qwen3.8与DeepSeekV4.1.md) | MHA、MQA、GQA、FlashAttention、MLA、线性与稀疏注意力等 |

### 3. 代表模型与资料索引

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| Qwen | [Qwen 技术优化路线与官方资料索引](技术代表/Qwen_技术优化路线与官方资料索引.md) | 从模块改动理解模型架构，附原始资料入口 |
| DeepSeek | [DeepSeek 技术优化路线与官方资料索引](技术代表/DeepSeek_技术优化路线与官方资料索引.md) | 注意力、KV 压缩、MoE 等优化路线与资料入口 |
| Qwen 架构专题 | [Qwen3.8-Flash-Next 架构深度解读](技术代表/Qwen3.8-Flash-Next_架构深度解读.md) | 整体结构、模块公式、手算示例与部署含义 |
| DeepSeek 架构专题 | [DeepSeek V4 / V4.1 架构深度解读](技术代表/DeepSeek-V4_V4.1_架构深度解读.md) | 分代架构对比、模块机制、计算示例与部署含义 |

## 目录结构

```text
.
├── README.md
├── 技术前要-深度学习基础/
│   ├── 激活函数/
│   │   ├── 激活函数_从非线性到SwiGLU.md
│   │   └── assets/              # 两条参考视频的 30 张关键画面
│   ├── 梯度消失与爆炸/
│   │   ├── 梯度消失与爆炸_从链式法则到残差连接.md
│   │   └── assets/              # 22 张参考视频画面
│   └── Transformer全流程/
│       ├── Transformer_从输入到自回归生成.md
│       └── assets/              # 35 张参考视频画面
├── 技术拆解-分词与向量化/
│   ├── BPE分词/
│   │   ├── BPE分词_从相邻合并到Embedding.md
│   │   ├── bpe_toy.py           # 仅用标准库的手算复现
│   │   └── assets/              # 23 张参考视频画面
│   └── Embedding/
│       ├── Token与Embedding_从strawberry计数到向量查表.md
│       ├── embedding_math.py   # 查表、梯度与容量核对
│       └── assets/              # 27 张参考视频画面
├── 技术起源-Transformer/
├── 技术拆解-位置编码优化/
│   ├── 正弦位置编码/            # 独立视频解读、复算代码与 24 张截图
│   └── RoPE基础/                # 独立视频解读、复算代码与 24 张截图
├── 技术拆解-注意力机制/
│   ├── QKV基础/                # 单头注意力解读、复算代码与 22 张截图
│   ├── 多头注意力/             # 双头手算、拼接融合与 22 张截图
│   └── KVCache/
│       ├── KVCache基础_从因果不变性到Prefill与Decode.md
│       ├── kv_cache_equivalence.py  # 双层因果注意力的缓存等价验证
│       ├── KVCache_MQA_GQA_MLA_从显存公式到低秩吸收.md
│       ├── kv_cache_math.py     # 缓存容量与矩阵吸收复算
│       └── assets/              # 进阶篇 28 张；kv-basics/ 为基础篇 29 张
├── 技术拆解-FNN层/
│   ├── MoE_核心论文与技术演进.md
│   ├── FFN/                    # FFN 视频解读、复算代码与 26 张截图
│   └── MoE/                    # MoE 视频解读、复算代码与 30 张截图
├── 技术拆解-残差链接/
├── 技术拆解-归一化/
├── 技术拆解-输出/
│   ├── 输出层与采样_从Logits到Temperature与TopP.md
│   ├── sampling_math.py        # 概率、候选过滤与采样核对
│   └── assets/                 # 28 张参考视频画面
└── 技术代表/
```

`技术拆解-FNN层` 是当前实际目录名，对应内容为 FFN / MoE 专题；残差连接与归一化已分别归入独立目录。

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
