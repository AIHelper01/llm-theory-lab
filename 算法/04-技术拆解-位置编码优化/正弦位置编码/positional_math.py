"""正弦位置编码的基础恒等式复算，仅用标准库。"""
from math import cos, sin, isclose


def encoding(position, dim=4, base=10000):
    assert dim > 0 and dim % 2 == 0
    result = []
    for i in range(dim // 2):
        angle = position * base ** (-2 * i / dim)
        result.extend([sin(angle), cos(angle)])
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


if __name__ == '__main__':
    assert encoding(0) == [0, 1, 0, 1]
    assert all(isclose(x, y, abs_tol=1e-6) for x, y in zip(
        encoding(1), [0.841471, 0.540302, 0.010000, 0.999950]))
    content = [0.2, -0.1, 0.5, 0.3]
    combined = [x + p for x, p in zip(content, encoding(1))]
    assert all(isclose(x, y, abs_tol=1e-6) for x, y in zip(
        combined, [1.041471, 0.440302, 0.510000, 1.299950]))
    for dim in [4, 8, 64]:
        for position in [0, 1, 20, 1000]:
            assert isclose(dot(encoding(position, dim), encoding(position, dim)), dim / 2)
        p, delta = 7, 3
        original, shifted = encoding(p, dim), encoding(p + delta, dim)
        rebuilt = []
        for i in range(dim // 2):
            angle = delta * 10000 ** (-2 * i / dim)
            a, b = original[2 * i:2 * i + 2]
            rebuilt.extend([cos(angle) * a + sin(angle) * b,
                            -sin(angle) * a + cos(angle) * b])
        assert all(isclose(x, y, abs_tol=1e-12) for x, y in zip(rebuilt, shifted))
        q = 19
        expected_dot = sum(cos((p - q) * 10000 ** (-2 * i / dim)) for i in range(dim // 2))
        assert isclose(dot(encoding(p, dim), encoding(q, dim)), expected_dot, abs_tol=1e-12)
    for p in [0, 1, 2]:
        print(f'PE({p}):', [round(x, 6) for x in encoding(p)])
    print('位置 1 的相加结果:', [round(x, 6) for x in combined])
    print('坐标、范数、位移矩阵与点积恒等式全部通过。')
