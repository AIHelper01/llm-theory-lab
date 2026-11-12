"""多头注意力的两头算例及投影等价验证，仅用标准库。"""
from math import exp, isclose


def softmax(scores):
    values = [exp(x - max(scores)) for x in scores]
    return [x / sum(values) for x in values]


def rowmul(row, matrix):
    return [sum(row[i] * matrix[i][j] for i in range(len(row)))
            for j in range(len(matrix[0]))]


def close(a, b):
    return len(a) == len(b) and all(isclose(x, y, abs_tol=1e-6) for x, y in zip(a, b))


if __name__ == '__main__':
    a, b, c = softmax([2, 1, 0])
    head1 = rowmul([a, b, c], [[2, 0], [0, 1], [0, 0]])
    head2 = rowmul(softmax([0, 2, 1]), [[0, 0], [0, 2], [1, 0]])
    assert close(head1, [1.330482, 0.244728])
    assert close(head2, [0.244728, 1.330482])
    joined = head1 + head2
    wo = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]]
    output = rowmul(joined, wo)
    assert close(output, [1.575210, 1.575210, 1.085753, -1.085753])
    part1, part2 = rowmul(head1, wo[:2]), rowmul(head2, wo[2:])
    assert close(output, [x + y for x, y in zip(part1, part2)])
    # 将不同头的矩阵按列拼接，再投影、拆分，与分别投影一致。
    w1 = [[1, 0], [0, 1], [1, -1], [2, 1]]
    w2 = [[0, 1], [1, 0], [-1, 2], [1, 1]]
    combined_w = [left + right for left, right in zip(w1, w2)]
    for x in [[1, 2, 3, 4], [0, 1, -1, 2]]:
        fused = rowmul(x, combined_w)
        assert close(fused[:2], rowmul(x, w1))
        assert close(fused[2:], rowmul(x, w2))
    dim = 512
    for heads in [2, 8]:
        width = dim // heads
        assert 3 * dim * (heads * width) + (heads * width) * dim == 1048576
    print('头 1:', [round(x, 6) for x in head1])
    print('头 2:', [round(x, 6) for x in head2])
    print('拼接:', [round(x, 6) for x in joined])
    print('输出投影:', [round(x, 6) for x in output])
    print('双头读取、拼接融合、分块等价与参数预算核对通过。')
