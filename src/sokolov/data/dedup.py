"""Vectorized 64-bit pHash Hamming search; O(Q*R) time, O(chunk*R) memory."""
from __future__ import annotations

CHUNK = 256


def phash64(image):
    """64-bit perceptual hash (imagehash.phash, hash_size=8) as an unsigned int."""
    import imagehash
    return int(str(imagehash.phash(image)), 16)


def _popcount_table():
    import numpy as np
    return np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hamming(left, right):
    """Pairwise Hamming distances (bits) between two uint64 hash vectors."""
    import numpy as np
    left, right = np.asarray(left, dtype=np.uint64), np.asarray(right, dtype=np.uint64)
    xor = np.bitwise_xor(left[:, None], right[None, :])
    return _popcount_table()[xor.view(np.uint8)].reshape(*xor.shape, 8).sum(-1, dtype=np.int32)


def nearest(queries, references):
    """Distance and index of the nearest reference for each query (-1/65 if none)."""
    import numpy as np
    queries = np.asarray(queries, dtype=np.uint64)
    distance = np.full(len(queries), 65, dtype=np.int32)
    index = np.full(len(queries), -1, dtype=np.int64)
    if not len(references) or not len(queries):
        return distance, index
    for start in range(0, len(queries), CHUNK):
        block = hamming(queries[start:start+CHUNK], references)
        index[start:start+CHUNK] = block.argmin(1)
        distance[start:start+CHUNK] = block.min(1)
    return distance, index


def close_pairs(hashes, max_distance):
    """All pairs i<j with distance <= max_distance, as (i, j, d), in i/j order."""
    import numpy as np
    hashes = np.asarray(hashes, dtype=np.uint64)
    pairs = []
    for start in range(0, len(hashes), CHUNK):
        block = hamming(hashes[start:start+CHUNK], hashes)
        rows, cols = np.nonzero(block <= max_distance)
        keep = cols > rows + start
        pairs += [(int(i)+start, int(j), int(block[i, j])) for i, j in zip(rows[keep], cols[keep])]
    return pairs
