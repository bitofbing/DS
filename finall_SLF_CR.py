import numpy as np
from scipy.optimize import minimize



def normalize_evidence(evidence_list):
    """
    归一化证据，使所有证据具有相同的事件键 {'A', 'B', 'C'}
    - 输入: evidence_list (列表), 包含多个证据体，每个证据体是一个字典，键为元组，值为概率。
    - 输出: 归一化后的证据列表，每个证据体具有相同的事件键 {'A', 'B', 'C'}。
    """
    all_hypotheses = {'A', 'B', 'C'}  # 确保所有证据具有相同的键
    normalized = []

    for ev in evidence_list:
        new_ev = {h: 0 for h in all_hypotheses}  # 初始化为 0
        for key, prob in ev.items():
            for h in key:  # 元组键可能包含多个事件
                new_ev[h] += prob / len(key)  # 平均分配概率
        normalized.append(new_ev)

    return normalized


def bhattacharyya_distance(p, q):
    """
    计算 Bhattacharyya 距离
    - 输入: p, q (列表或数组), 两个概率分布。
    - 输出: Bhattacharyya 距离，用于衡量两个概率分布之间的相似性。
    """
    p, q = np.array(p), np.array(q)
    return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)  # 避免 log(0)

def compute_similarity_matrix(evidence):
    """
    计算相似度矩阵 S
    - 输入: evidence (列表), 归一化后的证据列表。
    - 输出: 相似度矩阵 S，表示证据之间的相似性。
    """
    n = len(evidence)
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            p = list(evidence[i].values())
            q = list(evidence[j].values())

            distance = bhattacharyya_distance(p, q)
            similarity = np.exp(-distance)  # 转化为相似度

            S[i, j] = similarity
            S[j, i] = similarity  # 矩阵对称

    return S

def new_compute_weight_from_similarity(S, lambda_entropy=0.1):
    """
    新的权重计算方法，引入熵约束
    - 输入: S (矩阵), 相似度矩阵；lambda_entropy (浮点数), 熵约束的权重。
    - 输出: 权重 R，表示每个证据的可靠性。
    """
    S_avg = np.mean(S, axis=1)
    def objective(w):
        return -np.dot(w, S_avg) + lambda_entropy * np.sum(w * np.log(w + 1e-8))  # 加熵约束

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(S_avg)

    # w0 = np.ones(len(S_avg)) / len(S_avg)
    w0 = np.exp(S_avg) / np.sum(np.exp(S_avg))
    result = minimize(objective, w0, bounds=bounds, constraints=constraints)

    return result.x

def soft_likelihood_function(omega, evidence, R, alpha):
    """
    软似然函数 (Soft Likelihood Function, SLF)
    - 输入: omega (字符串), 当前假设；evidence (列表), 归一化后的证据列表；R (数组), 权重；alpha (浮点数), 乐观系数。
    - 输出: 软似然值 L，表示当前假设的软似然估计。
    """
    N = len(evidence)
    prob_values = [ev.get(omega, 0) for ev in evidence]
    R_values = R / np.sum(R)  # 归一化可靠性度

    # 排序索引
    sorted_indices = np.argsort([prob_values[i] * R_values[i] for i in range(N)])[::-1]

    # 计算权重向量
    w = []
    for i in range(N):
        if i == 0:
            w.append(R_values[sorted_indices[i]] ** ((1 - alpha) / alpha))
        else:
            sum_k = np.sum([R_values[sorted_indices[k]] for k in range(i + 1)])
            sum_k_prev = np.sum([R_values[sorted_indices[k]] for k in range(i)])
            w.append(sum_k ** ((1 - alpha) / alpha) - sum_k_prev ** ((1 - alpha) / alpha))

    # 计算软似然值
    L = 0
    for i in range(N):
        prod = 1
        for k in range(i + 1):
            prod *= prob_values[sorted_indices[k]]
        L += w[i] * prod

    return L

def fuse_evidence(evidence, R, alpha=0.1):
    """
    计算融合概率：
    - 使用软似然函数进行最终的概率估计
    - 输入: evidence (列表), 归一化后的证据列表；R (数组), 权重；alpha (浮点数), 乐观系数。
    - 输出: 融合后的概率分布。
    """
    hypotheses = {'A', 'B', 'C'}
    fused = {}

    for omega in hypotheses:
        L = soft_likelihood_function(omega, evidence, R, alpha)
        fused[omega] = L

    # 归一化
    total_L = sum(fused.values()) + 1e-10
    fused = {k: v / total_L for k, v in fused.items()}

    return fused

# 你的证据体
evidence = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('B',): 0.5, ('C',): 0.5},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A', 'C'): 0.8, ('B',): 0.2}
]

# evidence = [
#     {('A',): 0.6, ('B',): 0.3, ('A','B'): 0.1},  # 证据体 1
#     {('A',): 0.5, ('B',): 0.4, ('A','B'): 0.1}  # 证据体 2
# ]

# evidence = [
#     {('A',): 0.5, ('B',): 0.2, ('C',): 0.3},
#     {('B',): 0.9, ('C',): 0.1},
#     {('A',): 0.55, ('B',): 0.1, ('A','C'): 0.35},
#     {('A',): 0.55, ('B',): 0.1, ('A','C'): 0.35},
#     {('A',): 0.6, ('B',): 0.1, ('A', 'C'): 0.3}
# ]
# 处理证据体格式
normalized_evidence = normalize_evidence(evidence)

# 计算相似度矩阵
S = compute_similarity_matrix(normalized_evidence)

# 计算权重 R
R =new_compute_weight_from_similarity(S)

# 计算融合概率
alpha = 0.1  # 乐观系数
fused_probabilities = fuse_evidence(normalized_evidence, R, alpha)

print("Optimal R:", R)
print("Fused probabilities (after SLF):", fused_probabilities)