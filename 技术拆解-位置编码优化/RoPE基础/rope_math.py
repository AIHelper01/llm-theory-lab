"""RoPE 的相邻维度配对教学实现，不用于替换真实检查点布局。"""
from math import cos, sin, pi, isclose


def rotate2(vector, angle):
    x, y = vector
    return [cos(angle) * x - sin(angle) * y,
            sin(angle) * x + cos(angle) * y]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rope(vector, position, base=10000):
    dim = len(vector)
    assert dim > 0 and dim % 2 == 0
    result = []
    for i in range(dim // 2):
        result.extend(rotate2(vector[2 * i:2 * i + 2], position * base ** (-2 * i / dim)))
    return result


if __name__ == '__main__':
    theta = pi / 6
    q, k = [1, 0], [1, 0]
    score = dot(rotate2(q, theta), rotate2(k, 3 * theta))
    assert isclose(score, 0.5)
    assert isclose(dot(rotate2(q, 6 * theta), rotate2(k, 8 * theta)), score)
    changed = dot(rotate2(q, theta), rotate2([0, 1], 3 * theta))
    assert isclose(changed, -sin(pi / 3))
    assert isclose(dot(q, rotate2(k, 2 * theta)), dot(q, rotate2(k, 14 * theta)))
    # 非单位、不同内容向量，验证范数、共同旋转和相对位移。
    for q, k in [([2, -3], [0.7, 1.2]), ([0.3, 0.4], [-0.5, 0.2])]:
        assert isclose(dot(rotate2(q, 0.8), rotate2(q, 0.8)), dot(q, q))
        assert isclose(dot(rotate2(q, 0.8), rotate2(k, 0.8)), dot(q, k), abs_tol=1e-12)
        assert isclose(dot(rotate2(q, 2 * theta), rotate2(k, 7 * theta)),
                       dot(q, rotate2(k, 5 * theta)), abs_tol=1e-12)
    q, k = [1, 0, 1, 0], [0.2, -0.4, 0.3, 0.8]
    assert isclose(dot(rope(q, 2), rope(k, 7)), dot(q, rope(k, 5)), abs_tol=1e-12)
    assert isclose(dot(rope(q, 9), rope(q, 9)), dot(q, q))
    print('单频位置 1 / 3 的点积:', round(score, 6))
    print('改变 Key 内容方向后的点积:', round(changed, 6))
    print('四维位置 1:', [round(x, 6) for x in rope([1, 0, 1, 0], 1)])
    print('范数、共同旋转、相对位置及多频率恒等式核对通过。')
