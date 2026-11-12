"""FFN / SwiGLU 教学手算复现，仅用标准库。"""
from math import exp, isclose


def rowmul(row, matrix):
    return [sum(row[i] * matrix[i][j] for i in range(len(row)))
            for j in range(len(matrix[0]))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


if __name__ == '__main__':
    x = [1, -2]
    w1, b1 = [[1, 0, 1], [0, 1, -1]], [0, 1, 0]
    w2, b2 = [[1, 0], [0, 1], [1, -1]], [0.5, 0]
    pre = add(rowmul(x, w1), b1)
    active = [max(0, value) for value in pre]
    output = add(rowmul(active, w2), b2)
    assert pre == [1, -1, 3] and active == [1, 0, 3]
    assert output == [4.5, -3] and add(x, output) == [5.5, -5]
    # 无激活的两层与合并后的仿射变换一致。
    merged_w = [rowmul(row, w2) for row in w1]
    merged_b = add(rowmul(b1, w2), b2)
    assert add(rowmul(pre, w2), b2) == add(rowmul(x, merged_w), merged_b)
    for value in [-3, -1, 0, 2]:
        assert max(0, value) + max(0, -value) == abs(value)
    gated = [(g / (1 + exp(-g))) * u for g, u in zip([0, 1], [2, 3])]
    assert isclose(gated[0], 0) and isclose(gated[1], 2.193175735890015)
    count = 3 * 5120 * 17408 * 64
    assert count == 17112760320
    print('FFN 输出:', output, '含残差:', add(x, output))
    print('SwiGLU 示例:', gated)
    print('示例全层矩阵参数:', f'{count:,}')
    print('仿射合并、ReLU 折叠、门控与参数预算核对通过。')
