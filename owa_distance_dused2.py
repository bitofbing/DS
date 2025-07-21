import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from string import ascii_uppercase
from scipy.special import softmax
from scipy.stats import entropy

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "lines.linewidth": 2,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    'axes.labelweight': 'bold',
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",  # 更稳定跨平台字体
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "text.antialiased": True
})

def deng_entropy(mass_func):
    """
    计算邓熵（Deng Entropy）
    参数：
    - mass_func: 证据体字典 {frozenset: probability}
    - normalized: 是否归一化到[0,1]范围
    返回：
    - 邓熵值（默认归一化）
    """
    entropy = 0.0
    max_entropy = np.log2(sum(2 ** len(A) - 1 for A in mass_func.keys()))

    for A, m_A in mass_func.items():
        if m_A > 0:
            card = len(A)
            entropy -= m_A * np.log2(m_A / (2 ** card - 1))
        # 考虑假设空间复杂度（假设数量）
    hypothesis_complexity = 1 - 1 / len(mass_func)
    return entropy * hypothesis_complexity

def normalize_evidence(evidence_list):
    """
    归一化证据，使所有证据具有相同的事件键 {'A', 'B', 'C'}
    - 输入: evidence_list (列表), 包含多个证据体，每个证据体是一个字典，键为元组，值为概率。
    - 输出: 归一化后的证据列表，每个证据体具有相同的事件键 {'A', 'B', 'C'}。
    """
    # all_hypotheses = {'A', 'B', 'C'}  # 确保所有证据具有相同的键
    all_hypotheses = {'A', 'B', 'C', 'D', 'E'}  # 确保所有证据具有相同的键
    normalized = []

    for ev in evidence_list:
        new_ev = {h: 0 for h in all_hypotheses}  # 初始化为 0
        for key, prob in ev.items():
            for h in key:  # 元组键可能包含多个事件
                new_ev[h] += prob / len(key)  # 平均分配概率
        normalized.append(new_ev)

    return normalized

    # 字母到数字的自动映射 (A->1, B->2, ..., Z->26)
def to_numeric(x):
    if isinstance(x, str) and x.isalpha() and x in ascii_uppercase:
        return ascii_uppercase.index(x) + 1
    return int(x)  # 如果是数字字符串或数字

def compute_singleton_relation(i, j, alpha = 0.8):
    """
    通用单例元素关系计算
    支持: 字母(A-Z)、数字、或混合输入
    关系公式: 1 - exp(-|a - b|)
    """
    a = to_numeric(i)
    b = to_numeric(j)
    return alpha * (1 - np.exp(-abs(a - b))) + (1 - alpha) * np.abs(a - b)

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

def owa_similarity_matrix(evidence, gamma=0.8):
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

def compute_credibility_from_similarity(S):
    """
    计算归一化的可信度权重 Crd
    - 输入: S (相似度矩阵)
    - 输出: Crd (可信度权重向量)
    """
    # 计算支持度 Sup(m_i)
    Sup = np.sum(S, axis=1) - np.diag(S)  # 去掉自身对自身的支持度

    # 计算可信度 Crd 并归一化
    Crd = Sup / np.sum(Sup) if np.sum(Sup) != 0 else np.ones_like(Sup) / len(Sup)  # 避免除零

    return Crd

def soft_likelihood_function(omega, evidence, R_values, alpha):
    """
    软似然函数 (Soft Likelihood Function, SLF)
    - 输入: omega (字符串), 当前假设；evidence (列表), 归一化后的证据列表；R (数组), 权重；alpha (浮点数), 乐观系数。
    - 输出: 软似然值 L，表示当前假设的软似然估计。
    """
    N = len(evidence)
    prob_values = [ev.get(omega, 0) for ev in evidence]
    # R_values = R / np.sum(R)  # 归一化可靠性度

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

def compute_deng_entropy2(mass_function):
    """
    计算邓熵（Deng Entropy）
    - 输入: mass_function (字典), 表示证据的质量函数
    - 输出: 邓熵值
    """
    entropy = 0
    total_mass = sum(mass_function.values())
    for m in mass_function.values():
        if m > 0:
            p = m / total_mass
            entropy -= p * np.log2(p)
    return entropy


def compute_deng_entropy(mass_function):
    """改进的邓熵计算，考虑复合假设"""
    masses = np.array(list(mass_function.values()))
    masses = masses / np.sum(masses)

    # 计算基本概率分配的邓熵
    deng_entropy = -np.sum(masses * np.log2(masses + 1e-10))

    # 考虑假设空间复杂度（假设数量）
    hypothesis_complexity = 1 - 1 / len(mass_function)
    return deng_entropy * hypothesis_complexity


def compute_evidence_quality(evidence):
    """综合质量评估（邓熵+信息量+确定性）"""
    prob = np.array(list(evidence.values()))
    prob = prob / np.sum(prob)

    # 1. 邓熵指标（取反，因为熵越低质量越高）
    deng = deng_entropy(evidence)
    deng_score = 1 - (deng / np.log2(len(evidence)))  # 归一化

    # 2. 信息量指标
    info_score = 1 - entropy(prob) / np.log2(len(prob))

    # 3. 确定性指标
    certainty = np.max(prob)

    # 三重加权综合（可调参数）
    return 0.3 * deng_score + 0.4 * info_score + 0.3 * certainty


def compute_belief_from_similarity(S, evidence_list, alpha=0.75):
    """
    增强版置信度计算（三级融合）
    参数：
    - alpha: 质量主导系数 (0-1)
    返回：
    - belief_weights: 最终权重
    - metadata: 各阶段计算结果
    """
    n = len(evidence_list)
    np.fill_diagonal(S, 0)  # 确保无自相似

    # 第一阶段：基础可信度
    Crd = compute_credibility_from_similarity(S)

    # 第二阶段：质量评估
    Quality = np.array([compute_evidence_quality(ev) for ev in evidence_list])
    Q_norm = Quality / np.sum(Quality)

    # 第三阶段：邓熵加权一致性
    deng_weights = np.array([1 - deng_entropy(ev) / np.log2(len(ev) + 1e-10) for ev in evidence_list])
    Consistency = np.sum(S * deng_weights, axis=1)
    C_norm = Consistency / np.sum(Consistency)

    # 动态融合 (带可信度校准)
    combined = (alpha * Q_norm + (1 - alpha) * C_norm) * Crd  # 加入可信度平滑因子
    # combined = (alpha * Q_norm + (1 - alpha) * C_norm)  # 加入可信度平滑因子
    belief_weights = combined / np.sum(combined)

    return belief_weights

def compute_belief_from_similarity2(S, evidence_list):
    """
    基于相似度矩阵和邓熵计算综合置信度权重
    - 输入:
        S (n×n矩阵): 证据间的相似度矩阵
        evidence_list (列表): 证据的质量函数列表
    - 输出:
        Bel (向量): 综合置信度权重
        Sim (向量): 相似度支持度
        Ent (向量): 邓熵向量
    """
    n = len(evidence_list)

    # 1. 计算基于相似度的支持度 (去掉自相似)
    Sim = np.sum(S, axis=1) - np.diag(S)
    Sim = Sim / np.sum(Sim) if np.sum(Sim) > 0 else np.ones(n) / n

    # 2. 计算各证据的邓熵
    Ent = np.zeros(n)
    for i, ev in enumerate(evidence_list):
        total_mass = sum(ev.values())
        for m in ev.values():
            if m > 0:
                p = m / total_mass
                Ent[i] -= p * np.log2(p + 1e-10)  # 避免log(0)

    # 3. 邓熵归一化（熵越大权重应越小）
    Ent_norm = 1 - (Ent - np.min(Ent)) / (np.max(Ent) - np.min(Ent) + 1e-10)

    # 4. 综合置信度计算（相似度支持度 × 熵权重）
    Bel = Sim * Ent_norm
    Bel = Bel / np.sum(Bel) if np.sum(Bel) > 0 else np.ones(n) / n

    return Bel, Sim, Ent


def similarity_based_soft_likelihood(omega, evidence_list, R_values, alpha):
    """
    基于相似度矩阵的软似然函数
    - 输入:
        omega: 待评估假设
        evidence_list: 证据列表（每个证据为质量函数字典）
        S: 相似度矩阵
        alpha: 乐观系数
    - 输出:
        L: 软似然值
    """
    # 获取各证据对假设的支持度
    prob_values = [ev.get(omega, 0) for ev in evidence_list]

    # 按加权支持度降序排序
    sorted_indices = np.argsort([p * R for p, R in zip(prob_values, R_values)])[::-1]

    # 计算权重向量
    w = []
    for i in range(len(evidence_list)):
        if i == 0:
            w.append(R_values[sorted_indices[i]] ** ((1 - alpha) / alpha))
        else:
            sum_k = np.sum([R_values[sorted_indices[k]] for k in range(i + 1)])
            sum_k_prev = np.sum([R_values[sorted_indices[k]] for k in range(i)])
            w.append(sum_k ** ((1 - alpha) / alpha) - sum_k_prev ** ((1 - alpha) / alpha))

    # 计算软似然值
    L = 0
    for i in range(len(evidence_list)):
        prod = np.prod([prob_values[sorted_indices[k]] for k in range(i + 1)])
        L += w[i] * prod

    return L

def fuse_evidence(evidence, R, alpha=0.1):
    """
    计算融合概率：
    - 使用软似然函数进行最终的概率估计
    - 输入: evidence (列表), 归一化后的证据列表；R (数组), 权重；alpha (浮点数), 乐观系数。
    - 输出: 融合后的概率分布。
    """
    # hypotheses = {'A', 'B', 'C'}
    hypotheses = {'A', 'B', 'C','D','E'}
    fused = {}

    for omega in hypotheses:
        L = similarity_based_soft_likelihood(omega, evidence, R, alpha)
        fused[omega] = L

    print("fused_result_before_nor:", fused)
    # 归一化
    total_L = sum(fused.values()) + 1e-10
    fused = {k: v / total_L for k, v in fused.items()}

    return fused

def main_function(evidence):
    # 处理证据体格式
    normalized_evidence = normalize_evidence(evidence)

    # 计算相似度矩阵
    # S = compute_similarity_matrix_jousselme(normalized_evidence, "exponential")
    S = owa_similarity_matrix(evidence)

    # 计算权重 R
    print(S)
    # 计算综合置信度权重
    R_values = compute_belief_from_similarity(S, evidence)
    # 乘积融合
    # product = R * R_cre
    # R_fused = product / np.sum(product)
    # R_final = R_fused ** 2 / np.sum(R_fused ** 2)
    # # 指数加权参数
    # beta, gamma = 2.0, 2.0  # 可调整放大效果
    #
    # # 计算指数加权融合
    # W_final = (R ** beta) * (R_cre ** gamma)
    # alpha = 0.8  # 相似度权重占比80%
    # w_fused = alpha * R + (1 - alpha) * R_cre
    # w_fused /= np.sum(w_fused)  # 归一化

    # 计算融合概率
    alpha = 0.1  # 乐观系数
    fused_probabilities = fuse_evidence(normalized_evidence, R_values, alpha)
    # fused_probabilities_cre = fuse_evidence(normalized_evidence, R_cre, alpha)

    print("Optimal R:", R_values)
    print("Fused probabilities (after SLF):", fused_probabilities)
    return fused_probabilities

# 你的证据体
evidenceList = []
evidence5 = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('B',): 0.5, ('C',): 0.5},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
    {('A', 'C'): 0.8, ('B',): 0.2}
]
# evidenceList.append(evidence5)
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

evidence1 = [
    {('A',): 0.0437, ('B',): 0.3346, ('C',): 0.2916, ('A', 'B'): 0.0437, ('A', 'C'): 0.0239, ('B', 'C'): 0.2385, ('A', 'B', 'C'): 0.0239},  # 证据体 1 (Sepal Length)
    {('A',): 0.0865, ('B',): 0.2879, ('C',): 0.1839, ('A', 'B'): 0.0863, ('A', 'C'): 0.0865, ('B', 'C'): 0.1825, ('A', 'B', 'C'): 0.0863},  # 证据体 2 (Sepal Width)
    {('A',): 1.4e-09, ('B',): 0.6570, ('C',): 0.1726, ('A', 'B'): 1.3e-09, ('A', 'C'): 1.4e-11, ('B', 'C'): 0.1704, ('A', 'B', 'C'): 1.4e-11},  # 证据体 3 (Petal Length)
    {('A',): 8.20e-06, ('B',): 0.6616, ('C',): 0.1692, ('A', 'B'): 8.20e-06, ('A', 'C'): 3.80e-06, ('B', 'C'): 0.1692, ('A', 'B', 'C'): 3.80e-06}   # 证据体 4 (Petal Width)
]
# evidenceList.append(evidence1)
# 应用一的数据
# evidence2 = [
#     {('A',): 0.40, ('B',): 0.28, ('C',): 0.30, ('A', 'C'): 0.02},  # S₁
#     {('A',): 0.01, ('B',): 0.90, ('C',): 0.08, ('A', 'C'): 0.01},  # S₂
#     {('A',): 0.63, ('B',): 0.06, ('C',): 0.01, ('A', 'C'): 0.30},  # S₃
#     {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.30},  # S₄
#     {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.30}   # S₅
# ]
evidence2 = [
    {('A',): 0.40, ('B',): 0.28, ('C',): 0.30, ('A', 'C'): 0.01, ('C', 'A'): 0.01},  # S₁
    {('A',): 0.01, ('B',): 0.90, ('C',): 0.08, ('A', 'C'): 0.005, ('C', 'A'): 0.005},  # S₂
    {('A',): 0.63, ('B',): 0.06, ('C',): 0.01, ('A', 'C'): 0.02, ('C', 'A'): 0.01},  # S₃
    {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.02, ('C', 'A'): 0.01},  # S₄
    {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.02, ('C', 'A'): 0.02}   # S₅
]
evidenceList.append(evidence2)
#
# # 应用二的数据
evidence3 = [
    {('A',): 0.7, ('B',): 0.1, ('A','B','C'): 0.2},  # 证据体 m1
    {('A',): 0.7, ('A','B','C'): 0.3},               # 证据体 m2
    {('A',): 0.65, ('B',): 0.15, ('A','B','C'): 0.20},  # 证据体 m3
    {('A',): 0.75, ('C',): 0.05, ('A','B','C'): 0.20},  # 证据体 m4
    {('B',): 0.20, ('C',): 0.80}                      # 证据体 m5
]
# evidenceList.append(evidence3)
evidence4 = [
    {('A',): 0.6, ('B',): 0.15, ('C',): 0.15, ('D',): 0, ('E',): 0.1},  # 证据体 m1
    {('A',): 0.001, ('B',): 0.45, ('C',): 0.15, ('D',): 0.24 ,('E',): 0.159},               # 证据体 m2
    {('A',): 0.55, ('B',): 0.1, ('C',): 0.1, ('D',): 0.15 ,('E',): 0.1},  # 证据体 m3
    {('A',): 0.8, ('B',): 0.1, ('C',): 0.05, ('D',): 0 ,('E',): 0.05},  # 证据体 m3
]
# evidenceList.append(evidence4)
for evidence in evidenceList:
    main_function(evidence)