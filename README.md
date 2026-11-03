# LLM Theory Lab · 大模型知识系统梳理

围绕 **Transformer 原理 → 核心模块演进 → 代表模型架构 → 推理服务系统** 整理的中文学习笔记。

关注每项技术改动了哪个模块、解决什么问题、数学计算如何变化，以及收益和代价。内容包括论文解读、公式推导、架构对比、原始资料索引与课程逐页讲解。

## 内容导航

| 主题 | 文档 | 内容 |
| --- | --- | --- |
| Transformer 基础 | [Attention Is All You Need](技术起源-Transformer/Attention_Is_All_You_Need.md) | 从 Embedding、Attention、FFN 到 Encoder–Decoder 与 Decoder-only |
| 位置编码 | [位置编码论文与技术演进](技术演进-位置编码优化/Transformer_位置编码论文与技术演进.md) | Sinusoidal、相对位置表示、T5、RoPE、ALiBi |
| RoPE 专题 | [RoPE 原理与 Qwen / DeepSeek 架构解读](技术演进-位置编码优化/RoPE_原理与Qwen_DeepSeek架构解读.md) | 旋转位置编码的数学机制及架构应用 |
| 注意力机制 | [Attention 技术演进](技术演进-注意力机制/Attention_技术演进_Transformer至Qwen3.8与DeepSeekV4.1.md) | MHA、MQA、GQA、FlashAttention、MLA、线性与稀疏注意力等 |
| FFN 与专家模型 | [MoE 核心论文与技术演进](技术演进-FNN层/MoE_核心论文与技术演进.md) | 专家路由、激活参数、负载均衡及相关论文 |
| Qwen | [Qwen 技术优化路线与官方资料索引](技术代表/Qwen_技术优化路线与官方资料索引.md) | 从模块改动理解模型架构，附原始资料入口 |
| DeepSeek | [DeepSeek 技术优化路线与官方资料索引](技术代表/DeepSeek_技术优化路线与官方资料索引.md) | 注意力、KV 压缩、MoE 等优化路线与资料入口 |
| LLM 推理服务 | [LLM Serving Basics I 逐页解读](哈佛课程解读/llm_serving_basics_1/逐页解读.md) | 36 页原课件截图与中文讲解，覆盖请求生命周期、模型结构和系统成本 |

## 建议阅读顺序

1. **建立基础：** 阅读 Transformer 原始论文解读，理解 token、向量、Q/K/V、注意力、FFN 和残差连接。
2. **拆解模块：** 阅读位置编码、RoPE、Attention 和 MoE 专题，对照输入输出形状理解公式。
3. **连接具体模型：** 阅读 Qwen 与 DeepSeek 技术路线，将架构设计映射到基础模块。
4. **理解部署成本：** 阅读 LLM Serving 课程解读，把参数量、FLOPs、KV cache、显存、吞吐和延迟联系起来。

也可以从课程逐页解读入手，遇到不熟悉的概念时再回到对应专题。

## 目录结构

```text
.
├── README.md
├── 技术起源-Transformer/
├── 技术演进-位置编码优化/
├── 技术演进-注意力机制/
├── 技术演进-FNN层/
├── 技术代表/
└── 哈佛课程解读/
    └── llm_serving_basics_1/
        ├── 逐页解读.md
        └── assets/              # 36 张整页课件截图
```

`技术演进-FNN层` 是当前实际目录名，对应内容为 FFN / MoE 专题。

## 阅读方式

- **GitHub：** 通过上方导航直接浏览；图片使用仓库内相对路径。
- **本地：** 克隆后使用 Typora 或支持 LaTeX 公式的 Markdown 阅读器打开，无需安装项目依赖或执行构建。
- **公式与图示：** 部分笔记包含内嵌 HTML、SVG 和样式，GitHub 可能过滤部分展示效果；建议在支持这些格式的本地阅读器中查看。
- **课件截图：** 移动课程解读文档时，请同时保留相邻的 `assets/` 目录。

```bash
git clone https://github.com/AIHelper01/llm-theory-lab.git
cd llm-theory-lab
```

## 资料与使用说明

本仓库是学习整理，不是相关模型团队或课程机构的官方文档。论文、模型报告、公告与代码入口保留在各篇笔记中；引用与课件截图的权利归原作者或机构所有。

涉及模型版本、产品状态、性能数据和架构配置时，请结合文章标注的整理日期与原始资料阅读。性能倍数依赖硬件、精度、上下文长度、批次和测量方式；教学简化公式不能直接替代实际部署测试。

欢迎通过 Issue 指出公式、引用、链接或表述问题，并提供对应的原始资料与修正建议。
