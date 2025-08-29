import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from string import ascii_uppercase
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import itertools
from collections import defaultdict

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

def compute_rps_entropy(evidence):
    """
    严格遵循Example 4.1计算方法的RPS熵

    参数:
        evidence: 单个RPS证据体（字典格式，键为元组，值为质量）

    返回:
        entropy: 计算得到的熵值
    """
    total_entropy = 0.0

    # 预计算F(i)值（根据Example 4.1的算法）
    def compute_F(i):
        return sum(math.perm(i, k) for k in range(i + 1))

    # 计算每个证据项的贡献
    for items, mass in evidence.items():
        if mass <= 0:
            continue  # 跳过零质量项

        i = len(items)  # 当前组合长度
        F_i = compute_F(i)

        # 特别注意：Example中F(1)-1=2-1=1，但按定义F(1)=1! =1 → 需要确认
        # 根据Example 4.1的实际计算，F(1)=2, F(2)=5, F(3)=16
        # 这表明原定义可能有调整，这里采用示例中的值

        log_arg = mass / (F_i - 1)
        term = mass * math.log(log_arg)  # term本身是负的
        total_entropy -= term  # 负负得正
        # print(f"组合 {items}: mass={mass}, log参数={log_arg:.4f}, term={term:.4f}, 贡献={-term:.4f}")
    # hypothesis_complexity = 1 - 1 / len(evidence)
    #     print(f"证据 {evidence}的总rps熵: total_entropy={total_entropy}")
    # 标准化到0-1范围（熵值越小质量越高）
    max_possible_entropy = math.log(len(evidence) * 10)  # 经验值
    normalize_entropy = min(total_entropy / max_possible_entropy, 1.0)
    print(f"证据 {evidence}的标准化熵: normalize_entropy={normalize_entropy}")
    return normalize_entropy

def normalize_evidence(evidence_list):
    """
    归一化证据，使所有证据具有相同的事件键 {'A', 'B', 'C'}
    - 输入: evidence_list (列表), 包含多个证据体，每个证据体是一个字典，键为元组，值为概率。
    - 输出: 归一化后的证据列表，每个证据体具有相同的事件键 {'A', 'B', 'C'}。
    """
    all_hypotheses = {'A', 'B', 'C'}  # 确保所有证据具有相同的键
    # all_hypotheses = {'A', 'B', 'C', 'D', 'E'}  # 确保所有证据具有相同的键
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
    # evidence_frozenset = [{frozenset(k): v for k, v in ev.items()} for ev in evidence]
    evidence_frozenset = evidence.copy()
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

# def compute_evidence_quality(evidence):
#     """综合质量评估（邓熵+信息量+确定性）"""
#     prob = np.array(list(evidence.values()))
#     prob = prob / np.sum(prob)
#
#     # 1. 邓熵指标（取反，因为熵越低质量越高）
#     deng = deng_entropy(evidence)
#     deng_score = 1 - (deng / np.log2(len(evidence)))  # 归一化
#
#     # 2. 信息量指标
#     info_score = 1 - entropy(prob) / np.log2(len(prob))
#
#     # 3. 确定性指标
#     certainty = np.max(prob)
#
#     # 三重加权综合（可调参数）
#     return 0.4 * deng_score + 0.4 * info_score + 0.2 * certainty
#
#
# def compute_belief_from_similarity(S, evidence_list, alpha=0.8):
#     """
#     增强版置信度计算（三级融合）
#     参数：
#     - alpha: 质量主导系数 (0-1)
#     返回：
#     - belief_weights: 最终权重
#     - metadata: 各阶段计算结果
#     """
#     n = len(evidence_list)
#     np.fill_diagonal(S, 0)  # 确保无自相似
#
#     # 第一阶段：基础可信度
#     Crd = compute_credibility_from_similarity(S)
#
#     # 第二阶段：质量评估
#     Quality = np.array([compute_evidence_quality(ev) for ev in evidence_list])
#     Q_norm = Quality / np.sum(Quality)
#
#     # 第三阶段：邓熵加权一致性
#     deng_weights = np.array([1 - deng_entropy(ev) / np.log2(len(ev) + 1e-10) for ev in evidence_list])
#     Consistency = np.sum(S * deng_weights, axis=1)
#     C_norm = Consistency / np.sum(Consistency)
#
#     # 动态融合 (带可信度校准)
#     combined = (alpha * Q_norm + (1 - alpha) * C_norm) * Crd  # 加入可信度平滑因子
#     # combined = (alpha * Q_norm + (1 - alpha) * C_norm)  # 加入可信度平滑因子
#     belief_weights = combined / np.sum(combined)
#
#     return belief_weights

def compute_evidence_quality(evidence):
    """综合质量评估（RPS熵+信息量+确定性）"""
    prob = np.array(list(evidence.values()))
    prob = prob / (np.sum(prob) + 1e-10)

    # 1. RPS熵指标（取反，因为熵越低质量越高）
    rps_entropy = compute_rps_entropy(evidence)
    rps_score = 1- rps_entropy  # 已标准化

    # 2. 信息量指标（香农熵）
    # info_score = 1 - entropy(prob) / math.log(len(prob) + 1e-10)

    # 3. 确定性指标
    certainty = np.max(prob)

    # 三重加权综合（可调参数）
    return 0.7 * rps_score + 0.3 * certainty

def compute_belief_from_similarity(S, evidence_list, alpha=0.8):
    """
    增强版置信度计算（三级融合）
    参数：
    - S: 相似度矩阵
    - evidence_list: 证据体列表
    - alpha: 质量主导系数 (0-1)
    返回：
    - belief_weights: 最终权重
    """
    n = len(evidence_list)
    np.fill_diagonal(S, 0)  # 确保无自相似

    # 第一阶段：基础可信度
    Crd = compute_credibility_from_similarity(S)

    # 第二阶段：质量评估（使用改进的compute_evidence_quality）
    Quality = np.array([compute_evidence_quality(ev) for ev in evidence_list])
    Q_norm = Quality / (np.sum(Quality) + 1e-10)

    # 第三阶段：RPS熵加权一致性
    rps_weights = np.array([1 - compute_rps_entropy(ev) for ev in evidence_list])
    Consistency = np.sum(S * rps_weights.reshape(-1, 1), axis=1)
    C_norm = Consistency / (np.sum(Consistency) + 1e-10)

    # 动态融合（带可信度校准）
    # combined = (alpha * Q_norm + (1 - alpha) * C_norm) * Crd
    combined = (Q_norm + C_norm) * Crd
    belief_weights = combined / (np.sum(combined) + 1e-10)

    return belief_weights


def optimized_entropy_fusion(S, evidence_list):
    """集成所有优化技术的完整实现"""
    # 1. 计算熵特征
    entropies = np.array([compute_rps_entropy(ev) for ev in evidence_list])
    entropy_weights = np.exp(-entropies / 0.3)
    entropy_std = np.std(entropies)

    # 2. 改进的可信度计算
    raw_cred = compute_credibility_from_similarity(S)
    # cred = raw_cred * (1 - 0.2 * entropies)  # 熵惩罚

    # 3. 增强的一致性计算
    weighted_S = S * np.sqrt(entropy_weights[:, None] * entropy_weights[None, :])
    consistency = np.sum(weighted_S, axis=1)

    # 4. 动态融合（考虑熵分布特性）
    alpha = 0.4 * (1 - min(entropy_std, 0.4))
    fused = alpha * raw_cred + (1 - alpha) * consistency

    # 5. 结果后处理
    return np.clip(fused, 0, 1) / (np.sum(fused) + 1e-10)

def optimized_entropy_fusion_v2(S, evidence_list):
    """
    改进的熵融合方案（cred作为证据权重）
    参数：
        S: 相似度矩阵 (n×n)
        evidence_list: 证据体列表 [dict1, dict2,...]
    返回：
        融合后的权重向量
    """
    # 1. 计算初始可信度权重
    cred = compute_credibility_from_similarity(S)
    cred_weights = cred / (np.sum(cred) + 1e-10)

    # 2. 构建加权证据体
    fused_evidence = {}
    # 收集所有可能的命题
    all_propositions = set().union(*[set(ev.keys()) for ev in evidence_list])

    # 加权融合证据
    for prop in all_propositions:
        fused_evidence[prop] = sum(ev.get(prop, 0) * w
                                   for ev, w in zip(evidence_list, cred_weights))

    # 3. 计算融合后证据的熵值
    fused_entropy = compute_rps_entropy(fused_evidence)

    # 4. 熵加权一致性计算
    entropies = np.array([compute_rps_entropy(ev) for ev in evidence_list])
    entropy_weights = np.exp(-entropies / 0.3)  # 温度系数0.3
    weighted_S = S * np.sqrt(entropy_weights[:, None] * entropy_weights[None, :])
    consistency = np.sum(weighted_S, axis=1)

    # 5. 动态融合（基于融合熵调整权重）
    alpha = 0.5 * (1 - min(fused_entropy, 0.5))  # 融合熵越低，cred权重越大
    final_weights = alpha * cred + (1 - alpha) * consistency

    return consistency / (np.sum(consistency) + 1e-10)

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
    hypotheses = {'A', 'B', 'C','D'}
    # hypotheses = {
    # ('A',),
    # ('B',),
    # ('C',),
    # ('B', 'A'),
    # ('A', 'B'),
    # ('A', 'C'),
    # ('C', 'A', 'B'),
    # ('B', 'C', 'A'),
    # ('B', 'A', 'C'),
    # ('A', 'C', 'B'),
    # ('A', 'B', 'C')}
    # hypotheses = {'A', 'B', 'C','D','E'}
    fused = {}

    for omega in hypotheses:
        L = similarity_based_soft_likelihood(omega, evidence, R, alpha)
        fused[omega] = L

    print("fused_result_before_nor:", fused)
    # 归一化
    total_L = sum(fused.values()) + 1e-10
    fused = {k: v / total_L for k, v in fused.items()}

    return fused


def calculate_OPT(pmf_data, hypotheses={'A', 'B', 'C','D'}):
    """
    直接基于假设集的OPT计算
    参数：
    - pmf_data: SLF融合后的概率分布字典 {tuple: prob}
    - hypotheses: 预设的假设集合
    返回：
    - 标准化后的OPT概率分布 {class: prob}
    """
    # 初始化概率分布
    P_OPT = {h: 0.0 for h in hypotheses}

    # 按命题长度排序处理（单元素优先）
    sorted_items = sorted(pmf_data.items(),
                          key=lambda x: (len(x[0]), x[0]))

    for items, mass in sorted_items:
        items_set = set(items)  # 转换为集合避免顺序影响
        if len(items_set) == 1:
            # 单元素直接分配
            theta = next(iter(items_set))  # 获取唯一元素
            P_OPT[theta] += mass
        else:
            # 多元素按OPT规则分配
            last_element = items[-1]  # 保持原顺序的最后一个元素
            for theta in items_set:
                if theta != last_element:
                    P_OPT[theta] += mass / (len(items_set) - 1)

    # 标准化处理
    total = sum(P_OPT.values())
    return {k: v / total for k, v in P_OPT.items()}

def batch_OPT(data_list):
    return [calculate_OPT(d) for d in data_list]

def main_function(evidence):
    # 处理证据体格式
    # normalized_evidence = normalize_evidence(evidence)
    OPT_evidence = batch_OPT(evidence)

    # 计算相似度矩阵
    S = owa_similarity_matrix(evidence)

    # 计算权重 R
    print(S)
    # 计算综合置信度权重
    R_values = compute_belief_from_similarity(S, evidence)
    # 计算融合概率
    alpha = 0.1  # 乐观系数
    fused_probabilities = fuse_evidence(OPT_evidence, R_values, alpha)

    print("Optimal R:", R_values)
    print("Fused probabilities (after SLF):", fused_probabilities)
    # 执行计算
    # 执行计算
    # P_OPT = calculate_OPT(fused_probabilities)
    #
    # # 结果输出
    # print("OPT概率分布（直接假设集）：")
    # for hypo, prob in sorted(P_OPT.items()):
    #     print(f"P_OPT({hypo}) = {prob:.6f} ({prob * 100:.2f}%)")

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
# evidenceList.append(evidence2)
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

evidence5 = [
    {  # 传感器1
        ('A',): 0.31,
        ('B',): 0.0,
        ('C',): 0.29,
        ('A','C',): 0.0,
        ('C','A',): 0.0,
        ('A', 'B', 'C'): 0.0167,
        ('A', 'C', 'B'): 0.0167,
        ('B', 'A', 'C'): 0.0167,
        ('B', 'C', 'A'): 0.0167,
        ('C', 'A', 'B'): 0.3167,
        ('C', 'B', 'A'): 0.0167
    },
    {  # 传感器2
        ('A',): 0.0,
        ('B',): 0.8,
        ('C',): 0.2,
        ('A','C',): 0.0,
        ('C','A',): 0.0,
        ('A', 'B', 'C'): 0.0,
        ('A', 'C', 'B'): 0.0,
        ('B', 'A', 'C'): 0.0,
        ('B', 'C', 'A'): 0.0,
        ('C', 'A', 'B'): 0.0,
        ('C', 'B', 'A'): 0.0
    },
    {  # 传感器3
        ('A',): 0.27,
        ('B',): 0.07,
        ('C',): 0.21,
        ('A','C',): 0.0,
        ('C','A',): 0.0,
        ('A', 'B', 'C'): 0.025,
        ('A', 'C', 'B'): 0.025,
        ('B', 'A', 'C'): 0.025,
        ('B', 'C', 'A'): 0.025,
        ('C', 'A', 'B'): 0.325,
        ('C', 'B', 'A'): 0.025
    },
    {  # 传感器4
        ('A',): 0.25,
        ('B',): 0.05,
        ('C',): 0.3,
        ('A','C',): 0.09,
        ('C','A',): 0.31,
        ('A', 'B', 'C'): 0.0,
        ('A', 'C', 'B'): 0.0,
        ('B', 'A', 'C'): 0.0,
        ('B', 'C', 'A'): 0.0,
        ('C', 'A', 'B'): 0.3,
        ('C', 'B', 'A'): 0.0
    },
    {  # 专家
        ('A',): 0.25,
        ('B',): 0.0,
        ('C',): 0.2,
        ('A', 'C'): 0.36,
        ('C', 'A'): 0.0,
        ('A', 'B', 'C'): 0.0233,
        ('A', 'C', 'B'): 0.0733,
        ('B', 'A', 'C'): 0.0233,
        ('B', 'C', 'A'): 0.0233,
        ('C', 'A', 'B'): 0.0233,
        ('C', 'B', 'A'): 0.0233
    }
]
# evidenceList.append(evidence5)

evidence_rps = [
    {  # 第一组RPS证据
        ('A',): 0.2,
        ('B',): 0.08,
        ('C',): 0.0,
        ('B', 'A'): 0.05,
        ('A', 'B'): 0.12,
        ('A', 'C'): 0.03,
        ('C', 'A', 'B'): 0.0,
        ('B', 'C', 'A'): 0.05,
        ('B', 'A', 'C'): 0.1,
        ('A', 'C', 'B'): 0.25,
        ('A', 'B', 'C'): 0.12
    },
    {  # 第二组RPS证据
        ('A',): 0.07,
        ('B',): 0.13,
        ('C',): 0.02,
        ('B', 'A'): 0.2,
        ('A', 'B'): 0.07,
        ('A', 'C'): 0.1,
        ('C', 'A', 'B'): 0.08,
        ('B', 'C', 'A'): 0.0,
        ('B', 'A', 'C'): 0.2,
        ('A', 'C', 'B'): 0.0,
        ('A', 'B', 'C'): 0.13
    },
    {  # 第三组RPS证据
        ('A',): 0.14,
        ('B',): 0.09,
        ('C',): 0.0,
        ('B', 'A'): 0.08,
        ('A', 'B'): 0.12,
        ('A', 'C'): 0.00,
        ('C', 'A', 'B'): 0.05,
        ('B', 'C', 'A'): 0.0,
        ('B', 'A', 'C'): 0.1,
        ('A', 'C', 'B'): 0.3,
        ('A', 'B', 'C'): 0.12
    }
]
# evidenceList.append(evidence_rps)
example5_2 = [
    {  # RPS₁
        ('A',): 0.0,        # 零质量可保留或移除
        ('B', 'C'): 0.05,
        ('C', 'B'): 0.05,   # 顺序敏感保留
        ('D',): 0.9
    },
    {  # RPS₂
        ('A',): 0.9,
        ('B', 'C'): 0.05,
        ('C', 'B'): 0.05,
        ('D',): 0.0
    },
    {  # RPS₃
        ('A',): 0.8,
        ('B', 'C'): 0.05,
        ('C', 'B'): 0.05,
        ('D',): 0.1
    }
]
# evidenceList.append(example5_2)
example5_3 = [
    {  # RPS₁
        ('A',): 0.9,
        ('B', 'C'): 0.01,
        ('C', 'B'): 0.0,    # 显式保留零值
        ('D',): 0.09
    },
    {  # RPS₂
        ('A',): 0.0,
        ('B', 'C'): 0.01,
        ('C', 'B'): 0.9,    # 注意与RPS₁的顺序相反
        ('D',): 0.09
    },
    {  # RPS₃
        ('A',): 0.5,
        ('B', 'C'): 0.01,
        ('C', 'B'): 0.4,    # 顺序敏感值
        ('D',): 0.09
    }
]
# evidenceList.append(example5_3)
example5_4 = [
    {  # RPS₁
        ('A', 'B', 'C'): 0.9,
        ('A', 'C', 'B'): 0.0,    # 显式保留零值
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    },
    {  # RPS₂
        ('A', 'B', 'C'): 0.0,
        ('A', 'C', 'B'): 0.9,    # 注意顺序差异
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    },
    {  # RPS₃
        ('A', 'B', 'C'): 0.5,
        ('A', 'C', 'B'): 0.4,    # 顺序敏感值
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    }
]
# evidenceList.append(example5_4)
# 定义映射关系
index_to_label = {0: 'A', 1: 'B', 2: 'C'}

# 转换函数
def convert_to_labeled_rps(gen_rps):
    labeled_evidence = []
    for rps in gen_rps:
        converted = {}
        for items, mass in rps:
            # 转换数字索引为字母标签
            labeled_items = tuple(index_to_label[i] for i in items)
            converted[labeled_items] = mass
        labeled_evidence.append(converted)
    return labeled_evidence

gen_rps = [{((0, 1, 2), 1.5917244981330598e-07),
  ((0, 2, 1), 2.9406750290725125e-07),
  ((1, 0, 2), 2.0213860437974062e-07),
  ((1, 2), 0.09118749040651578),
  ((1, 2, 0), 9.460419907293513e-07),
  ((2,), 0.7403419597193839),
  ((2, 0, 1), 6.003605197431245e-07),
  ((2, 1), 0.16846682721585785),
  ((2, 1, 0), 1.5208771748850276e-06)},
 {((0, 1, 2), 0.01188199867148949),
  ((0, 2, 1), 0.01035662243762494),
  ((1,), 0.47885110874073056),
  ((1, 0, 2), 0.015796363493213478),
  ((1, 2), 0.22509177919736081),
  ((1, 2, 0), 0.025677447776139048),
  ((2, 0, 1), 0.012617876180362793),
  ((2, 1), 0.19619515499138454),
  ((2, 1, 0), 0.02353164851169426)},
 {((0, 1, 2), 5.169212053030185e-74),
  ((0, 2, 1), 4.396215399074331e-74),
  ((1,), 0.5816550235840346),
  ((1, 0, 2), 1.049079060904447e-73),
  ((1, 2), 0.22607603321880362),
  ((1, 2, 0), 1.454721947440344e-72),
  ((2, 0, 1), 7.66492634482424e-74),
  ((2, 1), 0.19226894319716173),
  ((2, 1, 0), 1.2497557492710257e-72)},
 {((0, 1, 2), 3.891018838666681e-27),
  ((0, 2, 1), 2.3087813150525835e-27),
  ((1,), 0.9176022547914116),
  ((1, 0, 2), 6.637943618793252e-27),
  ((1, 2), 0.051713147346844665),
  ((1, 2, 0), 1.1562906044195511e-26),
  ((2, 0, 1), 2.7440209721440796e-27),
  ((2, 1), 0.030684597861743695),
  ((2, 1, 0), 8.055665013028123e-27)}]
evidenceList.append(convert_to_labeled_rps(gen_rps))
gen_rps2 = [{((np.int64(0), np.int64(1), np.int64(2)), np.float64(0.001359736249463779)),
  ((np.int64(0), np.int64(2), np.int64(1)), np.float64(0.0014790228833903361)),
  ((np.int64(1),), np.float64(0.5353385436887967)),
  ((np.int64(1), np.int64(0), np.int64(2)), np.float64(0.0018800952025744943)),
  ((np.int64(1), np.int64(2)), np.float64(0.2144841486344294)),
  ((np.int64(1), np.int64(2), np.int64(0)), np.float64(0.0048440492199631835)),
  ((np.int64(2), np.int64(0), np.int64(1)), np.float64(0.002171180946677301)),
  ((np.int64(2), np.int64(1)), np.float64(0.23330036547890506)),
  ((np.int64(2), np.int64(1), np.int64(0)), np.float64(0.005142857695799624))},
 {((np.int64(0), np.int64(1)), np.float64(0.055610137778653805)),
  ((np.int64(0), np.int64(1), np.int64(2)), np.float64(0.01792730353286244)),
  ((np.int64(0), np.int64(2), np.int64(1)), np.float64(0.014633224363548039)),
  ((np.int64(1),), np.float64(0.6980767817513214)),
  ((np.int64(1), np.int64(0)), np.float64(0.10359027356000934)),
  ((np.int64(1), np.int64(0), np.int64(2)), np.float64(0.02406398488669137)),
  ((np.int64(1), np.int64(2), np.int64(0)), np.float64(0.036589596039018915)),
  ((np.int64(2), np.int64(0), np.int64(1)), np.float64(0.017293834222907995)),
  ((np.int64(2), np.int64(1), np.int64(0)), np.float64(0.03221486386498678))},
 {((np.int64(0), np.int64(1), np.int64(2)),
   np.float64(9.213446872810916e-110)),
  ((np.int64(0), np.int64(2), np.int64(1)),
   np.float64(1.121078380449317e-109)),
  ((np.int64(1), np.int64(0), np.int64(2)),
   np.float64(1.5999215604396331e-109)),
  ((np.int64(1), np.int64(2)), np.float64(0.13939649243334243)),
  ((np.int64(1), np.int64(2), np.int64(0)),
   np.float64(3.2558870196707563e-108)),
  ((np.int64(2),), np.float64(0.6909879487700293)),
  ((np.int64(2), np.int64(0), np.int64(1)),
   np.float64(2.3449787118105853e-109)),
  ((np.int64(2), np.int64(1)), np.float64(0.16961555879662818)),
  ((np.int64(2), np.int64(1), np.int64(0)),
   np.float64(3.9218926165327734e-108))},
 {((np.int64(0), np.int64(1), np.int64(2)),
   np.float64(1.0778812549904096e-68)),
  ((np.int64(0), np.int64(2), np.int64(1)),
   np.float64(1.8867311816777108e-68)),
  ((np.int64(1), np.int64(0), np.int64(2)),
   np.float64(1.4122510145910698e-68)),
  ((np.int64(1), np.int64(2)), np.float64(4.02655783757686e-06)),
  ((np.int64(1), np.int64(2), np.int64(0)), np.float64(7.08712766574289e-68)),
  ((np.int64(2),), np.float64(0.9999889253260626)),
  ((np.int64(2), np.int64(0), np.int64(1)),
   np.float64(3.8473205490596514e-68)),
  ((np.int64(2), np.int64(1)), np.float64(7.048116099813455e-06)),
  ((np.int64(2), np.int64(1), np.int64(0)),
   np.float64(1.103005365781949e-67))}]
evidenceList.append(convert_to_labeled_rps(gen_rps))
for evidence in evidenceList:
    main_function(evidence)