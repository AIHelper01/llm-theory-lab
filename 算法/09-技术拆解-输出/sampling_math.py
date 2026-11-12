"""输出采样教学复算，仅用标准库，不调用真实模型。"""
from math import exp, isclose, log
from random import Random


def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError('教学公式要求 temperature > 0；贪心应另行选择。')
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    values = [exp(x - peak) for x in scaled]
    return [x / sum(values) for x in values]


def renormalize(probabilities, keep):
    mass = sum(probabilities[i] for i in keep)
    return [p / mass if i in keep else 0.0 for i, p in enumerate(probabilities)]


def top_k(probabilities, k):
    if k < 1:
        raise ValueError('k 必须至少为 1。')
    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    return renormalize(probabilities, set(order[:k]))


def top_p(probabilities, cutoff):
    if not 0 < cutoff <= 1:
        raise ValueError('cutoff 必须在 (0, 1] 内。')
    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    keep, cumulative = set(), 0.0
    for i in order:
        keep.add(i)
        cumulative += probabilities[i]
        if cumulative >= cutoff:
            break
    return renormalize(probabilities, keep)


def sample_with_uniform(probabilities, u):
    if not 0 <= u < 1:
        raise ValueError('u 必须在 [0, 1) 内。')
    cumulative = 0.0
    for i, p in enumerate(probabilities):
        cumulative += p
        if u < cumulative:
            return i
    # 仅处理累计概率浮点舍入误差，不返回被过滤的尾部候选。
    return max(i for i, p in enumerate(probabilities) if p > 0)


def equal_vectors(a, b):
    return len(a) == len(b) and all(isclose(x, y, abs_tol=1e-12) for x, y in zip(a, b))


if __name__ == '__main__':
    logits = [log(6), log(3), 0.0]
    p = softmax(logits)
    assert equal_vectors(p, [0.6, 0.3, 0.1])
    assert equal_vectors(p, softmax([x + 1000 for x in logits]))
    assert equal_vectors(softmax(logits, 0.5), [36 / 46, 9 / 46, 1 / 46])
    assert equal_vectors(top_k(p, 2), [2 / 3, 1 / 3, 0])
    assert equal_vectors(top_p(p, 0.8), [2 / 3, 1 / 3, 0])
    assert equal_vectors(top_p([0.6, 0.3, 0.1], 0.6), [1, 0, 0])
    assert equal_vectors(top_p([0.95, 0.03, 0.02], 0.9), [1, 0, 0])
    assert equal_vectors(top_p([0.4, 0.35, 0.25], 0.9), [0.4, 0.35, 0.25])
    assert sum(x > 0 for x in top_p(p, 0.85)) == 2
    assert sum(x > 0 for x in top_p(softmax(logits, 2), 0.85)) == 3
    assert sample_with_uniform([0.6, 0.3, 0.1], 0.72) == 1
    assert sample_with_uniform([0.6, 0.3, 0.1], 0.6) == 1
    assert sample_with_uniform([0.6, 0.3, 0.1], 0.95) == 2
    rng_a, rng_b = Random(42), Random(42)
    a = [sample_with_uniform(p, rng_a.random()) for _ in range(20)]
    b = [sample_with_uniform(p, rng_b.random()) for _ in range(20)]
    assert a == b
    assert (1e16 + -1e16) + 1 == 1
    assert 1e16 + (-1e16 + 1) == 0
    for temperature in [0.5, 1, 2]:
        print(f'T={temperature}:', [round(x, 6) for x in softmax(logits, temperature)])
    print('Top-p=0.8:', top_p(p, 0.8))
    print('固定种子抽样前 10 个 ID:', a[:10])
    print('softmax、温度、候选边界、随机种子与浮点例子核对通过。')
