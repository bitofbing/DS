import numpy as np
from itertools import combinations


def jaccard_similarity(A, B):
    """
    计算两个焦元之间的 Jaccard 相似度
    """
    if A == B == frozenset():  # 空集的特殊情况
        return 1.0
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    return intersection / union if union > 0 else 0.0


def build_jaccard_matrix(focal_elements):
    """
    构建 Jaccard 矩阵 D
    """
    n = len(focal_elements)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = jaccard_similarity(focal_elements[i], focal_elements[j])
    return D


def jousselme_distance(m1, m2):
    """
    计算两个 BPA 之间的 Jousselme 距离
    """
    # 获取所有焦元
    focal_elements = list(set(m1.keys()).union(set(m2.keys())))
    focal_elements = [frozenset(f) for f in focal_elements]

    # 构建 Jaccard 矩阵
    D = build_jaccard_matrix(focal_elements)

    # 构建 BPA 向量
    m1_vec = np.array([m1.get(f, 0) for f in focal_elements])
    m2_vec = np.array([m2.get(f, 0) for f in focal_elements])

    # 计算距离
    diff = m1_vec - m2_vec
    distance = np.sqrt(0.5 * np.dot(diff.T, np.dot(D, diff)))
    return distance


# 示例证据体
evidence = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 1
    {('A', 'C'): 0.8, ('B',): 0.2}  # 证据体 2
]

# 将证据体转换为适合的格式
evidence_frozenset = []
for ev in evidence:
    ev_frozenset = {frozenset(k): v for k, v in ev.items()}
    evidence_frozenset.append(ev_frozenset)

# 计算 Jousselme 距离
distance = jousselme_distance(evidence_frozenset[0], evidence_frozenset[1])
print("Jousselme Distance:", distance)