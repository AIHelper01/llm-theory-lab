"""Embedding 教学数值核对；不调用或模拟真实模型分词器。"""
from math import isclose


if __name__ == '__main__':
    word = 'strawberry'
    positions = [i for i, ch in enumerate(word, 1) if ch == 'r']
    assert positions == [3, 8, 9]
    for pieces in [[word], ['straw', 'berry'], list(word)]:
        assert ''.join(pieces) == word
        assert sum(piece.count('r') for piece in pieces) == 3

    table = [[0.1, 0.2, -0.1], [-0.2, 0.3, 0.0],
             [0.4, -0.2, 0.6], [0.0, 0.5, 0.2]]
    ids = [2, 0, 2]
    lookup = [table[token_id][:] for token_id in ids]
    one_hot = [[int(i == token_id) for i in range(len(table))] for token_id in ids]
    selected = [[sum(row[i] * table[i][j] for i in range(len(table)))
                 for j in range(3)] for row in one_hot]
    assert lookup == selected
    upstream = [[1, 0, -1], [-0.5, 1, 0.5], [0.5, 2, 0]]
    gradient = [[0.0] * 3 for _ in table]
    for token_id, row in zip(ids, upstream):
        for j, value in enumerate(row):
            gradient[token_id][j] += value
    assert gradient[2] == [1.5, 2, -1]
    assert gradient[0] == [-0.5, 1, 0.5]
    updated = [x - 0.1 * g for x, g in zip(table[2], gradient[2])]
    assert all(isclose(x, y) for x, y in zip(updated, [0.25, -0.4, 0.7]))
    parameters = 248320 * 5120
    assert parameters == 1271398400
    weight_bytes = parameters * 2
    assert weight_bytes == 2542796800
    print('r 的数量和位置:', len(positions), positions)
    print('直接查表与 one-hot 选行相同:', lookup)
    print('重复 ID 累加梯度:', gradient[2])
    print('第 2 行 SGD 更新:', [round(x, 6) for x in updated])
    print(f'示例表权重: {weight_bytes / 1e9:.3f} GB = {weight_bytes / 2**30:.3f} GiB')
    print('全部教学数值核对通过。')
