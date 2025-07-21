import numpy as np
from scipy.optimize import minimize


# 动态计算单例关系矩阵
def compute_singleton_relation(i, j):
    return 1 - np.exp(-abs(i - j))  # 使用修正后的公式

# 计算单例到集合
def compute_singleton_set(A, B, gamma):
    r_values = []
    for b in B:
        r_values.append(compute_singleton_relation(A, b))
    return owa_aggregation(r_values, gamma, force_binary=False)

def build_relation_matrix(gamma=0.8):  # 使用调整后的gamma值
    elements = [1, 2, 3, 4, (1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]
    n = len(elements)
    R_o = np.zeros((n, n))

    is_singleton = lambda x: isinstance(x, int)
    for i in range(n):
        for j in range(i, n):  # 仅计算上三角
            A, B = elements[i], elements[j]

            # Case 1: 单例-单例关系
            if is_singleton(A) and is_singleton(B):
                R_o[i, j] = compute_singleton_relation(A, B)

            # Case 2: 单例-集合关系
            elif is_singleton(A) and not is_singleton(B):
                R_o[i, j] = compute_singleton_set(A, B, gamma)

            # Case 3: 集合-集合关系
            elif not is_singleton(A) and not is_singleton(B):
                r_values = []
                for a in A:
                    r_values.append(compute_singleton_set(a, B, gamma))
                R_o[i, j] = owa_aggregation(r_values, gamma, force_binary=False)
    # 对称填充下三角
    R_o = np.triu(R_o) + np.triu(R_o, 1).T
    return R_o


def owa_aggregation(values, gamma, force_binary=False):
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
    bounds = [(0.001, 1) for _ in range(n)]  # 避免0权重
    res = minimize(entropy, x0=np.ones(n) / n, bounds=bounds,
                   constraints=constraints, options={'ftol': 1e-10})

    weights = res.x / np.sum(res.x)  # 强制归一化
    return np.dot(sorted_values, weights)


def d_OWA(m1, m2, R_o):
    """
    基于关系矩阵的距离计算
    公式: d = sqrt(0.5 * (m1-m2)^T (I - R_o) (m1-m2))

    参数:
        m1, m2: 质量函数向量（需与R_o相同的焦元顺序）
        R_o: 关系矩阵
    """
    diff = m1 - m2
    M = np.eye(len(R_o)) - R_o  # (I - R_o)
    distance = np.sqrt(0.5 * diff.T @ M @ diff)
    return distance

# 验证
if __name__ == "__main__":
    R_o = build_relation_matrix()

    # 检查单例关系
    print("单例关系验证:")
    print(f"r(1,2) = {R_o[0, 1]:.4f} (预期: 0.6321)")
    print(f"r(1,3) = {R_o[0, 2]:.4f} (预期: 0.8647)")
    print(f"r(1,4) = {R_o[0, 3]:.4f} (预期: 0.9502)")

    # 检查关键集合关系
    print("\n集合关系验证:")
    print(f"r(1,{{1,3}}) = {R_o[0, 7]:.4f} (预期: 0.7390)")
    print(f"r({{1,2}},{{3,4}}) = {R_o[4, 6]:.4f} (预期: 0.9101)")

    # 完整矩阵输出
    print("\n完整关系矩阵:")
    print(np.round(R_o, 4))