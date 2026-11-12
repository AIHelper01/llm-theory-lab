# 推理实践

第一阶段从加载模型并完成一次生成开始。先跟踪分词、Prefill、Decode 和采样，再测量首 Token 延迟、生成速度和显存占用。单请求跑通后，再比较批处理和服务端调度。

可以先运行算法篇的教学脚本，对照缓存解码和整段重算的结果。推理服务的请求生命周期和调度见课程笔记。 [最小 Decoder-only 示例](../算法/01-技术前要-深度学习基础/Transformer全流程/decoder_demo.py) · [哈佛 LLM Serving Basics](../算法/11-哈佛课程解读/llm_serving_basics_1/逐页解读.md)
