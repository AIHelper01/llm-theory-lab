"""KV 裸缓存容量及 MLA 线性吸收的教学复算；不是推理性能测试。"""
from math import isclose


def kv_bytes(tokens, layers, kv_heads, key_dim, value_dim, element_bytes):
    return tokens * layers * kv_heads * (key_dim + value_dim) * element_bytes


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


if __name__ == '__main__':
    per_token = {
        name: kv_bytes(1, 80, heads, 128, 128, 2)
        for name, heads in [('MHA', 64), ('GQA', 8), ('MQA', 1)]
    }
    assert per_token == {'MHA': 2621440, 'GQA': 327680, 'MQA': 40960}
    total = kv_bytes(50 * 2500, 80, 64, 128, 128, 2)
    assert total == 327680000000
    assert per_token['MHA'] / per_token['GQA'] == 8
    assert per_token['MHA'] / per_token['MQA'] == 64
    assert kv_bytes(1, 64, 4, 256, 256, 2) == 262144
    assert kv_bytes(1, 16, 4, 256, 256, 2) == 65536
    latent_cache = (512 + 64) * 2
    assert latent_cache == 1152
    assert isclose((512 + 64) / (2 * 128 * 128), 0.017578125)

    # Key 吸收：q * (U_K c) == (U_K^T q)^T c。
    latents = [(1, 2), (3, 0)]
    key_up, query = (2, 1), 2
    expanded_scores = [query * dot(key_up, c) for c in latents]
    absorbed_query = [query * x for x in key_up]
    absorbed_scores = [dot(absorbed_query, c) for c in latents]
    assert expanded_scores == absorbed_scores == [8, 12]

    # Value 吸收：给定权重，先上投影再加权 == 先加权再上投影。
    weights, value_up = (0.25, 0.75), (1, -1)
    expanded_output = sum(a * dot(value_up, c) for a, c in zip(weights, latents))
    weighted_latent = [sum(a * c[i] for a, c in zip(weights, latents)) for i in range(2)]
    absorbed_output = dot(value_up, weighted_latent)
    assert isclose(expanded_output, absorbed_output)
    assert isclose(absorbed_output, 2)
    for name, size in per_token.items():
        print(f'{name}: {size:,} bytes/token = {size / 1024:,.0f} KiB/token')
    print(f'50 × 2500 Token MHA: {total / 1e9:.2f} GB = {total / 2**30:.2f} GiB')
    print('MLA Key 分数:', absorbed_scores, 'Value 输出:', absorbed_output)
    print('全部容量与吸收等式核对通过。')
