"""字符级 BPE 教学复现；不包含生产 Tokenizer 的预处理及特殊 Token。"""
from collections import Counter


def merge(tokens, pair):
    """从左到右进行非重叠替换。"""
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(''.join(pair))
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


def train(corpus, steps):
    sequences = {tuple(text): frequency for text, frequency in corpus.items()}
    rules = []
    totals = [sum(len(tokens) * n for tokens, n in sequences.items())]
    for _ in range(steps):
        counts = Counter()
        for tokens, n in sequences.items():
            for pair in zip(tokens, tokens[1:]):
                counts[pair] += n
        if not counts:
            break
        # 频次优先，平局时按符号对的字典序选择，以便结果稳定。
        pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        rules.append(pair)
        updated = Counter()
        for tokens, n in sequences.items():
            updated[merge(tokens, pair)] += n
        sequences = updated
        totals.append(sum(len(tokens) * n for tokens, n in sequences.items()))
    return rules, totals


def encode(text, rules):
    tokens = tuple(text)
    ranks = {pair: rank for rank, pair in enumerate(rules)}
    while True:
        candidates = [pair for pair in zip(tokens, tokens[1:]) if pair in ranks]
        if not candidates:
            return list(tokens)
        tokens = merge(tokens, min(candidates, key=ranks.__getitem__))


if __name__ == '__main__':
    rules, totals = train({'abc': 4, 'abd': 2, 'ab': 1}, 3)
    assert rules == [('a', 'b'), ('ab', 'c'), ('ab', 'd')]
    assert totals == [20, 13, 9, 7]
    assert encode('abcd', rules) == ['abc', 'd']
    assert merge(tuple('aaa'), ('a', 'a')) == ('aa', 'a')
    assert encode('abc', [('b', 'c'), ('a', 'b')]) == ['a', 'bc']
    assert '咖'.encode('utf-8').hex(' ').upper() == 'E5 92 96'
    assert '啡'.encode('utf-8').hex(' ').upper() == 'E5 95 A1'
    print('合并规则:', rules)
    print('加权 Token 总数:', totals)
    print('abcd 编码:', encode('abcd', rules))
    print('全部教学示例核对通过。')
