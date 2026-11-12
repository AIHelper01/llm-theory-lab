"""MoE 路由、加权输出和参数预算的教学复算。"""
from math import exp, isclose, log


def softmax(scores):
    values = [exp(x - max(scores)) for x in scores]
    return [x / sum(values) for x in values]


def topk_weights(probabilities, k):
    selected = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))[:k]
    denominator = sum(probabilities[i] for i in selected)
    return selected, [probabilities[i] / denominator for i in selected]


if __name__ == '__main__':
    p = softmax([log(x) for x in [0.5, 0.25, 0.15, 0.1]])
    selected, weights = topk_weights(p, 2)
    assert selected == [0, 1]
    assert all(isclose(a, b) for a, b in zip(weights, [2 / 3, 1 / 3]))
    outputs = [[3, 0], [0, 6]]
    routed = [sum(w * value[j] for w, value in zip(weights, outputs)) for j in range(2)]
    assert all(isclose(a, b) for a, b in zip(routed, [2, 2]))
    combined = [a + 0.5 * b for a, b in zip(routed, [1, -1])]
    assert all(isclose(a, b) for a, b in zip(combined, [2.5, 1.5]))
    def balance(f, probabilities):
        return len(f) * sum(a * b for a, b in zip(f, probabilities))
    assert balance([0.5, 0.5], [0.5, 0.5]) == 1
    assert balance([1, 0], [1, 0]) == 2
    d, m, experts, active = 4, 8, 8, 2
    per_expert, router = 3 * d * m, d * experts
    total = (experts + 1) * per_expert + router
    activated = (active + 1) * per_expert + router
    assert (total, activated) == (896, 320)
    q = 4
    assert experts * 3 * d * m == (q * experts) * 3 * d * (m // q)
    assert active * 3 * d * m == (q * active) * 3 * d * (m // q)
    print('Top-2 专家:', selected, '权重:', weights)
    print('路由输出:', routed, '加入共享门控:', combined)
    print('示例总参数:', total, '激活参数:', activated)
    print('路由、均衡对比及细粒度参数等式核对通过。')
