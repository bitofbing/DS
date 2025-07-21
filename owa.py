import numpy as np
from scipy.optimize import minimize
from string import ascii_uppercase

def compute_singleton_relation(i, j):
    """
    通用单例元素关系计算
    支持: 字母(A-Z)、数字、或混合输入
    关系公式: 1 - exp(-|a - b|)
    """

    # 字母到数字的自动映射 (A->1, B->2, ..., Z->26)
    def to_numeric(x):
        if isinstance(x, str) and x.isalpha() and x in ascii_uppercase:
            return ascii_uppercase.index(x) + 1
        return int(x)  # 如果是数字字符串或数字

    a = to_numeric(i)
    b = to_numeric(j)
    return 0 if a == b else 1 - np.exp(-abs(a - b))

def compute_singleton_set(A, B, gamma):
    """计算单例到集合的关系"""
    r_values = []
    for b in B:
        r_values.append(compute_singleton_relation(A, b))
    return owa_aggregation(r_values, gamma, force_binary=False)


def build_relation_matrix(focal_elements, gamma=0.8):
    """构建关系矩阵（适配字母型证据体）"""
    n = len(focal_elements)
    R_o = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # 仅计算上三角
            A, B = focal_elements[i], focal_elements[j]

            # Case 1: 单例-单例关系
            if len(A) == 1 and len(B) == 1:
                a = next(iter(A))  # 获取集合中的唯一元素
                b = next(iter(B))
                R_o[i, j] = compute_singleton_relation(a, b)

            # Case 2: 单例-集合关系
            elif len(A) == 1 and len(B) > 1:
                a = next(iter(A))
                R_o[i, j] = compute_singleton_set(a, B, gamma)

            # Case 3: 集合-集合关系
            elif len(A) > 1 and len(B) > 1:
                r_values = []
                for a in A:
                    r_values.append(compute_singleton_set(a, B, gamma))
                R_o[i, j] = owa_aggregation(r_values, gamma, force_binary=False)

    # 对称填充下三角
    R_o = np.triu(R_o) + np.triu(R_o, 1).T
    return R_o


def owa_aggregation(values, gamma, force_binary=False):
    """OWA聚合函数（保持不变）"""
    if len(values) == 1:
        return values[0]

    if force_binary:  # 二元情况直接计算
        sorted_v = sorted(values, reverse=True)
        return gamma * sorted_v[0] + (1 - gamma) * sorted_v[1]

    sorted_values = np.array(sorted(values, reverse=True))
    n = len(sorted_values)

    def entropy(w):
        return -np.sum(w * np.log(w + 1e-10))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.sum([(n - k - 1) / (n - 1) * w[k] for k in range(n)]) - gamma}
    ]
    bounds = [(0.001, 1) for _ in range(n)]
    res = minimize(entropy, x0=np.ones(n) / n, bounds=bounds,
                   constraints=constraints, options={'ftol': 1e-10})

    weights = res.x / np.sum(res.x)
    return np.dot(sorted_values, weights)


def d_OWA(m1, m2, gamma=0.8):
    """计算两个证据体之间的距离"""
    # 获取所有焦元（保持顺序一致性）
    focal_elements = sorted(list(set(m1.keys()).union(set(m2.keys()))), key=lambda x: (len(x), x))

    # 构建关系矩阵
    R_o = build_relation_matrix(focal_elements, gamma)

    # 转换为向量（保持相同顺序）
    m1_vec = np.array([m1.get(f, 0) for f in focal_elements])
    m2_vec = np.array([m2.get(f, 0) for f in focal_elements])

    # 计算距离
    diff = m1_vec - m2_vec
    M = np.eye(len(R_o)) - R_o
    distance = np.sqrt(0.5 * diff.T @ M @ diff)
    return distance


def owa_similarity(evidence, gamma=0.8):
    """计算证据体列表的相似度矩阵"""
    # 转换证据体格式
    evidence_frozenset = [{frozenset(k): v for k, v in ev.items()} for ev in evidence]

    n = len(evidence_frozenset)
    similarity_matrix = np.eye(n)  # 对角线为1

    for i in range(n):
        for j in range(i + 1, n):
            distance = d_OWA(evidence_frozenset[i], evidence_frozenset[j], gamma)
            similarity = 1 - distance
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    return similarity_matrix


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例证据体
    evidence = [
        {('A',): 0.6, ('B',): 0.3, ('A', 'B'): 0.1},  # 证据体1
        {('A',): 0.5, ('B',): 0.4, ('A', 'B'): 0.1}  # 证据体2
    ]

    # 计算相似度矩阵
    sim_matrix = owa_similarity(evidence)
    print("相似度矩阵:")
    print(sim_matrix)

    # 计算具体距离
    m1 = {frozenset(k): v for k, v in evidence[0].items()}
    m2 = {frozenset(k): v for k, v in evidence[1].items()}
    distance = d_OWA(m1, m2)
    print(f"\n证据体间距离: {distance:.4f}")
    print(f"证据体间相似度: {1 - distance:.4f}")