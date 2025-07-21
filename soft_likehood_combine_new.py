import numpy as np

def normalize_evidence(evidence_list):
    """ 归一化证据，使所有证据具有相同的事件键 {'A', 'B', 'C'} """
    all_hypotheses = {'A', 'B', 'C'}
    normalized = []

    for ev in evidence_list:
        new_ev = {h: 0 for h in all_hypotheses}
        for key, prob in ev.items():
            for h in key:
                new_ev[h] += prob / len(key)  # 平均分配概率
        normalized.append(new_ev)

    return normalized

def bhattacharyya_distance(p, q):
    """ 计算 Bhattacharyya 距离 """
    p, q = np.array(p), np.array(q)
    return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)  # 避免 log(0)

def compute_similarity_matrix(evidence):
    """ 计算相似度矩阵 S """
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

def compute_weight_from_similarity(S, gamma=2):
    """ 计算权重 R（优化版，指数调整） """
    S_avg = np.mean(S, axis=1)  # 每行的均值
    R = S_avg ** gamma  # 通过指数调整权重
    return R / np.sum(R)  # 归一化

def soft_likelihood_fusion(evidence, R, alpha=0.8):
    """
    SLF-CR 融合方法（优化版）
    - R: 计算的相似性权重
    - alpha: 排序权重参数
    """
    fused = {}

    for omega in evidence[0]:  # 遍历所有假设
        P_raw = np.array([evidence[i][omega] for i in range(len(evidence))])

        # **排序加权**
        sorted_indices = np.argsort(-P_raw)  # 从大到小排序
        rank_weights = np.array([(i + 1) ** (-alpha) for i in range(len(P_raw))])
        rank_weights /= np.sum(rank_weights)  # 归一化

        # **计算融合概率**
        fused[omega] = np.sum(R * rank_weights * P_raw)

    # **归一化**
    total = sum(fused.values()) + 1e-10
    fused = {k: v / total for k, v in fused.items()}

    return fused

# 你的证据体
evidence = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('B',): 0.5, ('C',): 0.5},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A', 'C'): 0.8, ('B',): 0.2}
]

# 处理证据体格式
normalized_evidence = normalize_evidence(evidence)

# 计算相似度矩阵
S = compute_similarity_matrix(normalized_evidence)

# 计算权重 R（优化版）
R = compute_weight_from_similarity(S, gamma=2)

# 计算融合概率（SLF-CR 方法）
fused_probabilities = soft_likelihood_fusion(normalized_evidence, R, alpha=0.1)

print("Optimal R:", R)
print("Fused probabilities (optimized SLF-CR):", fused_probabilities)
