"""用固定教学权重演示单层 Decoder-only Transformer 的生成过程。"""

from math import cos, exp, isclose, sin, sqrt
from random import Random


VOCAB = ("我", "爱", "猫", "狗", "。")
EMBEDDINGS = ((1, 0.2), (0.2, 1), (-0.7, 0.6), (0.5, -0.8), (-0.6, -0.7))
W_Q = ((0.5, 0.2), (-0.1, 0.8))
W_K = ((0.4, -0.3), (0.2, 0.7))
W_V = ((0.6, 0.1), (0.3, -0.4))
W_O = ((0.8, -0.2), (0.1, 0.7))
W_UP = ((0.7, 0.2), (-0.3, 0.5), (0.4, -0.6))
W_DOWN = ((0.3, -0.1, 0.5), (-0.2, 0.6, 0.1))


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def project(weights, vector):
    return [dot(row, vector) for row in weights]


def softmax(scores):
    peak = max(scores)
    weights = [exp(score - peak) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def rms_norm(vector):
    scale = sqrt(dot(vector, vector) / len(vector) + 1e-5)
    return [value / scale for value in vector]


def rotate(vector, position):
    # 头维度为 2，只有一对旋转通道，角频率为 1。
    x, y = vector
    c, s = cos(position), sin(position)
    return [c * x - s * y, s * x + c * y]


def forward(token_ids, cache):
    """返回每个新位置的词表 logits，并把旋转后的 K 和原始 V 写入缓存。"""
    start = len(cache)
    inputs = [EMBEDDINGS[token_id] for token_id in token_ids]
    queries = []
    for position, state in enumerate(inputs, start):
        norm = rms_norm(state)
        queries.append(rotate(project(W_Q, norm), position))
        cache.append((rotate(project(W_K, norm), position), project(W_V, norm)))

    logits = []
    for position, (state, query) in enumerate(zip(inputs, queries), start):
        # 因果掩码只允许读到当前位置，即使 Prefill 已经算出了后续 K/V。
        visible = cache[: position + 1]
        scores = [dot(query, key) / sqrt(len(query)) for key, _ in visible]
        weights = softmax(scores)
        context = [sum(w * value[i] for w, (_, value) in zip(weights, visible))
                   for i in range(len(state))]
        hidden = [a + b for a, b in zip(state, project(W_O, context))]

        expanded = [max(0.0, x) for x in project(W_UP, rms_norm(hidden))]
        hidden = [a + b for a, b in zip(hidden, project(W_DOWN, expanded))]
        # 复用 Embedding 权重作输出投影，得到词表大小的 logits。
        logits.append(project(EMBEDDINGS, rms_norm(hidden)))
    return logits


def demo():
    prompt = "我爱"
    prefix = [VOCAB.index(char) for char in prompt]
    cache = []
    rng = Random(0)
    print(f"输入 {prompt!r} → Token IDs {prefix}")
    logits = forward(prefix, cache)[-1]
    print(f"Prefill 处理 {len(prefix)} 个位置，KV Cache 长度 {len(cache)}")

    for step in range(3):
        probabilities = softmax(logits)
        token_id = rng.choices(range(len(VOCAB)), weights=probabilities)[0]
        print(f"采样 {step + 1}: logits={[round(x, 4) for x in logits]}, "
              f"概率={[round(p, 4) for p in probabilities]}, "
              f"Token {token_id} ({VOCAB[token_id]})")
        prefix.append(token_id)
        if step == 2:
            break
        logits = forward([token_id], cache)[-1]
        full_logits = forward(prefix, [])[-1]
        error = max(abs(cached - full) for cached, full in zip(logits, full_logits))
        assert all(isclose(a, b, rel_tol=0, abs_tol=1e-12)
                   for a, b in zip(logits, full_logits))
        assert len(cache) == len(prefix)
        print(f"Decode 只处理 1 个位置，KV Cache 长度 {len(cache)}，"
              f"与整段重算的 logits 最大误差 {error:.3g}")

    print("生成结果:", "".join(VOCAB[token_id] for token_id in prefix))


if __name__ == "__main__":
    demo()
