"""Constrained search with original, unrenormalized token log probabilities."""
import numpy as np


class SIDTrie:
    def __init__(self, item_ids, codes):
        self.children, self.descendants, self.leaves = {}, {}, {}
        self.length = codes.shape[1]
        for item, code in zip(item_ids, codes):
            key = tuple(map(int, code))
            if key in self.leaves:
                raise ValueError('SID is not unique')
            self.leaves[key] = int(item)
            for depth in range(len(key)+1):
                prefix = key[:depth]
                self.descendants.setdefault(prefix, set()).add(int(item))
                if depth < len(key):
                    self.children.setdefault(prefix, set()).add(key[depth])

    def allowed(self, prefix, excluded):
        return [token for token in sorted(self.children.get(tuple(prefix), []))
                if not self.descendants[tuple(prefix)+(token,)].issubset(excluded)]


def beam_search(log_probabilities, trie, width, k, excluded=None):
    if width < k or k < 1:
        raise ValueError('Require beam width >= k >= 1')
    excluded = set() if excluded is None else set(excluded)
    beams = [((), 0.)]
    for _ in range(trie.length):
        scores = np.asarray(log_probabilities([prefix for prefix, _ in beams]))
        if scores.ndim != 2 or len(scores) != len(beams) or not np.isfinite(scores).all():
            raise ValueError('Invalid decoder log probabilities')
        expanded = []
        for row, (prefix, value) in enumerate(beams):
            for token in trie.allowed(prefix, excluded):
                expanded.append((prefix+(token,), value+float(scores[row, token])))
        expanded.sort(key=lambda pair: (-pair[1], pair[0]))
        beams = expanded[:width]
        if not beams:
            return []
    result = [(trie.leaves[code], score) for code, score in beams]
    return sorted(result, key=lambda pair: (-pair[1], pair[0]))[:k]


def exact_search(log_probabilities, trie, k, excluded=None):
    """Tiny-catalog oracle. No pruning; used to verify scoring and beam behavior."""
    return beam_search(log_probabilities, trie, max(k, len(trie.leaves)), k, excluded)
