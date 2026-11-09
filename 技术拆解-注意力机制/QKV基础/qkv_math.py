"""单头 QKV 教学复算，不加载真实模型权重。"""
from math import exp, sqrt, isclose


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(values):
    peak = max(values)
    raw = [exp(x - peak) for x in values]
    return [x / sum(raw) for x in raw]


def attend(query, keys, values):
    scores = [dot(query, key) / sqrt(len(query)) for key in keys]
    weights = softmax(scores)
    result = [sum(w * row[j] for w, row in zip(weights, values))
              for j in range(len(values[0]))]
    return weights, result


def close(a, b):
    return all(isclose(x, y, abs_tol=1e-6) for x, y in zip(a, b))


if __name__ == '__main__':
    x = [0, 1, 1, 1]
    wq = [[0, 0], [1, 0], [0, 1], [0, -1]]
    query = [sum(x[i] * wq[i][j] for i in range(4)) for j in range(2)]
    assert query == [1, 0]
    keys, values = [[2, 1], [1, 2], [0, 1]], [[2, 0], [0, 2], [1, 1]]
    assert [dot(query, key) for key in keys] == [2, 1, 0]
    weights, output = attend(query, keys, values)
    assert close(weights, [0.575975, 0.283995, 0.140029])
    assert close(output, [1.291980, 0.708020])
    changed_weights, changed_value_output = attend(query, keys, [[4, 1], values[1], values[2]])
    assert close(weights, changed_weights)
    assert close(changed_value_output, [2.443931, 1.283995])
    assert close([b - a for a, b in zip(output, changed_value_output)],
                 [weights[0] * 2, weights[0]])
    new_weights, changed_key_output = attend(query, [[0, 0], keys[1], keys[2]], values)
    assert close(new_weights, [0.248255, 0.503490, 0.248255])
    assert close(changed_key_output, [0.744765, 1.255235])
    assert dot([1, 0], [2, 1]) == 2
    assert dot([1, 3], [0, 1]) == 3
    # 完整因果矩阵与逐个 Query 的可见前缀计算一致。
    queries = [[1, 2], [2, -2], [1, 0]]
    matrix = [softmax([dot(q, key) / sqrt(2) if j <= i else float('-inf')
                       for j, key in enumerate(keys)]) for i, q in enumerate(queries)]
    for i, q in enumerate(queries):
        assert isclose(sum(matrix[i]), 1)
        assert all(w == 0 for w in matrix[i][i + 1:])
        full_result = [sum(w * row[j] for w, row in zip(matrix[i], values)) for j in range(2)]
        _, prefix_result = attend(q, keys[:i + 1], values[:i + 1])
        assert close(full_result, prefix_result)
    print('Q:', query)
    print('读取权重:', [round(x, 6) for x in weights])
    print('加权输出:', [round(x, 6) for x in output])
    print('只改 V:', [round(x, 6) for x in changed_value_output])
    print('只改 K:', [round(x, 6) for x in changed_key_output])
    print('投影、读取、受控变化和因果矩阵验证全部通过。')
