"""双层因果注意力的缓存等价验证，仅用标准库，不是模型性能测试。"""
from math import exp, isclose, log, sqrt

LAYERS = [
    ([[0.5, 0.2], [-0.1, 0.8]], [[0.4, -0.3], [0.2, 0.7]],
     [[0.6, 0.1], [0.3, -0.4]]),
    ([[0.7, -0.2], [0.1, 0.4]], [[0.2, 0.5], [-0.3, 0.6]],
     [[0.4, -0.1], [0.2, 0.9]]),
]


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def softmax(scores):
    largest = max(scores)
    values = [exp(score - largest) for score in scores]
    return [value / sum(values) for value in values]


def attend(query, keys, values):
    scores = [sum(a * b for a, b in zip(query, key)) / sqrt(len(query)) for key in keys]
    weights = softmax(scores)
    return [sum(w * v[i] for w, v in zip(weights, values)) for i in range(len(values[0]))]


def full_forward(inputs):
    """所有位置同时计算，显式截断每行的可见范围。"""
    states = [list(x) for x in inputs]
    for wq, wk, wv in LAYERS:
        queries = [matvec(wq, x) for x in states]
        keys = [matvec(wk, x) for x in states]
        values = [matvec(wv, x) for x in states]
        outputs = []
        for i, query in enumerate(queries):
            attention = attend(query, keys[:i + 1], values[:i + 1])
            outputs.append([x + a for x, a in zip(states[i], attention)])
        states = outputs
    return states


def cached_step(token_input, caches):
    """新位置逐层处理，每层只保存历史 Key 与 Value。"""
    state = list(token_input)
    for (wq, wk, wv), (keys, values) in zip(LAYERS, caches):
        query = matvec(wq, state)
        keys.append(matvec(wk, state))
        values.append(matvec(wv, state))
        attention = attend(query, keys, values)
        state = [x + a for x, a in zip(state, attention)]
    return state


def max_error(left, right):
    assert len(left) == len(right)
    return max(abs(a - b) for xs, ys in zip(left, right) for a, b in zip(xs, ys))


if __name__ == '__main__':
    inputs = [[1.0, 0.2], [-0.5, 0.7], [0.3, -0.4], [0.8, 1.1]]
    caches = [([], []) for _ in LAYERS]
    incremental = []
    for i, token_input in enumerate(inputs):
        incremental.append(cached_step(token_input, caches))
        error = max_error(full_forward(inputs[:i + 1]), incremental)
        assert error < 1e-12
        assert all(len(keys) == len(values) == i + 1 for keys, values in caches)
        print(f'前缀长度 {i + 1}: 全量/缓存最大误差 {error:.3g}')

    changed_future = inputs[:-1] + [[20.0, -30.0]]
    assert max_error(full_forward(inputs)[:-1], full_forward(changed_future)[:-1]) < 1e-12
    weights = softmax([0, log(2), log(3)])
    assert all(isclose(a, b) for a, b in zip(weights, [1 / 6, 2 / 6, 3 / 6]))
    result = attend([1], [[0], [log(2)], [log(3)]], [[1, 0], [0, 3], [2, 2]])
    assert isclose(result[0], 7 / 6) and isclose(result[1], 2)
    repeated = sum(range(1000, 2000))
    cached_positions = 2000
    assert repeated == 1499500 and repeated / cached_positions == 749.75
    per_token = 2 * 80 * 64 * 128 * 2
    assert per_token == 2621440
    assert per_token * 4096 == 10 * 2**30
    print(f'教学位置计数比: {repeated / cached_positions:.2f}')
    print(f'位置处理次数下降: {(1 - cached_positions / repeated) * 100:.4f}%')
    print('因果不变性、缓存等价、注意力手算及容量验证全部通过。')
