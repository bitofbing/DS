import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from string import ascii_uppercase
from scipy.special import softmax

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


def deng_entropy2(mass_func, normalized=True):
    """
    计算邓熵（Deng Entropy）
    参数：
    - mass_func: 证据体字典 {frozenset: probability}
    - normalized: 是否归一化到[0,1]范围
    返回：
    - 邓熵值（默认归一化）
    """
    entropy = 0.0
    max_entropy = np.log2(sum(2 ** len(A) - 1 for A in mass_func.keys())) if normalized else 1.0

    for A, m_A in mass_func.items():
        if m_A > 0:
            card = len(A)
            entropy -= m_A * np.log2(m_A / (2 ** card - 1))

    return entropy / max_entropy if normalized else entropy

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

def compute_singleton_relation2(i, j, alpha=0.8, beta=0.5):
    """
    单例关系计算（整合Jaccard相似度）
    参数：
    - alpha: 控制数值关系的权重 (0-1)
    - beta: 控制Jaccard相似度的权重 (0-1)
    假设：i和j都是单例集合（如'A'或('A',)等形式）
    """
    def to_numeric(x):
        """将单例元素转换为数值"""
        elem = next(iter(x)) if isinstance(x, (set, tuple, list)) else x
        if isinstance(elem, str) and elem.isalpha() and elem in ascii_uppercase:
            return ascii_uppercase.index(elem) + 1
        return int(elem)

    def to_set(x):
        """将单例转换为集合形式（用于Jaccard计算）"""
        if isinstance(x, (set, tuple, list)):
            return set(x)
        return {x}

    # 数值关系计算
    a_num = to_numeric(i)
    b_num = to_numeric(j)
    exp_diff = 1 - np.exp(-abs(a_num - b_num))
    abs_diff = abs(a_num - b_num)
    numeric_rel = alpha * exp_diff + (1 - alpha) * abs_diff

    # Jaccard相似度计算
    set_i = to_set(i)
    set_j = to_set(j)
    intersection = len(set_i & set_j)
    union = len(set_i | set_j)
    jaccard = intersection / union if union != 0 else 0

    # 融合两种度量（使用beta权重）
    return beta * (1 - jaccard) + (1 - beta) * numeric_rel

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


def d_OWA2(m1, m2, gamma=0.8, epsilon=1e-10, use_deng_entropy=True):
    """计算两个证据体之间的距离"""
    # 获取所有焦元（保持顺序一致性）
    focal_elements = sorted(list(set(m1.keys()).union(set(m2.keys()))), key=lambda x: (len(x), x))

    # 计算邓熵差异（如果启用）
    entropy_weight = 1.0
    if use_deng_entropy:
        e1 = deng_entropy(m1)
        e2 = deng_entropy(m2)
        entropy_diff = np.abs(e1 - e2)
        entropy_weight = 1 + entropy_diff  # 熵差越大，距离放大系数越大

    # 构建关系矩阵
    R_o = build_relation_matrix(focal_elements, gamma)

    # 转换为向量（保持相同顺序）
    m1_vec = np.array([m1.get(f, 0) for f in focal_elements])
    m2_vec = np.array([m2.get(f, 0) for f in focal_elements])
    # 归一化处理（可选，根据需求）
    # m1_vec = m1_vec / (np.sum(m1_vec) + epsilon)
    # m2_vec = m2_vec / (np.sum(m2_vec) + epsilon)

    # 计算距离
    diff = m1_vec - m2_vec
    M = np.eye(len(R_o)) - R_o
    distance = np.sqrt(0.5 * diff.T @ M @ diff)
    return entropy_weight * distance  # 应用邓熵修正


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

def d_OWA3(m1, m2, gamma=0.8, epsilon=1e-10, use_deng_entropy=False, entropy_power=2.0):
    """
    纯邓熵修正的距离计算（通过幂次放大熵的影响）
    参数：
    - entropy_power: 熵差的放大指数（>1时增强分化）
    - 其他参数同原始函数
    """
    # 获取焦元并计算基础距离
    focal_elements = sorted(list(set(m1.keys()).union(set(m2.keys()))), key=lambda x: (len(x), x))
    R_o = build_relation_matrix(focal_elements, gamma)

    m1_vec = np.array([m1.get(f, 0) for f in focal_elements])
    m2_vec = np.array([m2.get(f, 0) for f in focal_elements])
    diff = m1_vec - m2_vec
    M = np.eye(len(R_o)) - R_o
    base_distance = np.sqrt(0.5 * diff.T @ M @ diff)

    # 纯邓熵修正
    if use_deng_entropy:
        e1 = deng_entropy(m1)
        e2 = deng_entropy(m2)
        entropy_diff = np.abs(e1 - e2)

        # 幂次放大熵差影响
        entropy_weight = 1 + entropy_diff ** entropy_power
        return base_distance * entropy_weight

    return base_distance

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

def nonlinear_aggregation(S, alpha=1.0):
    """
    限制 log-sum-exp 放大效应，避免某些证据占据主导地位
    """
    return np.log(1 + np.sum(np.exp(alpha * S), axis=1)) / alpha

def adaptive_gaussian_transform(x):
    """
    自适应高斯变换：根据数据的均值和标准差动态调整 mu 和 sigma
    """
    mu = np.mean(x)
    sigma = np.std(x)
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-8))**2)  # 防止除零

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

def new_compute_weight_from_similarity_nonlinear(S, lambda_entropy=0.1, beta=3.5, alpha=1.0):
    S_agg = nonlinear_aggregation(S, alpha)  # 非线性聚合
    S_agg_sigmoid = 1 / (1 + np.exp(-beta * (S_agg - np.mean(S_agg))))
    def objective(w):
        return -np.dot(w, S_agg_sigmoid) + lambda_entropy * np.sum(w * np.log(w + 1e-8))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(S_agg_sigmoid)

    w0 = S_agg_sigmoid / np.sum(S_agg_sigmoid)
    # result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    # 优化
    result = minimize(
        objective,
        w0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'  # 使用 SLSQP 算法
    )
    return result.x


def deng_entropy_weights(evidence):
    """基于邓熵计算初始权重（不确定性惩罚）"""
    entropies = np.array([deng_entropy(ev) for ev in evidence])
    return np.exp(-entropies)  # 不确定性越高权重越低


def hybrid_weight_computation(S, evidence, lambda_entropy=0.2, beta=2.0, alpha=0.5):
    """
    融合相似度与邓熵的权重计算
    参数：
    - S: 相似度矩阵
    - evidence: 证据体列表
    - lambda_entropy: 熵正则化强度
    - beta: Sigmoid陡度参数
    - alpha: 相似度/熵权重平衡因子
    """
    # 1. 邓熵权重计算
    W_deng = deng_entropy_weights(evidence)

    # 2. 相似度非线性聚合（改进版）
    S_agg = np.log(1 + np.sum(np.exp(beta * S), axis=1)) / beta  # 平滑聚合

    # 3. 熵感知的Sigmoid变换
    S_sigmoid = 1 / (1 + np.exp(-beta * (S_agg - np.median(S_agg))))  # 使用中位数更鲁棒

    # 4. 多目标优化（加入熵约束）
    def objective(w):
        # 相似度拟合项 + 熵正则项 + 邓熵适配项
        fit_term = -np.dot(w, S_sigmoid)
        entropy_term = lambda_entropy * np.sum(w * np.log(w + 1e-10))
        deng_term = alpha * np.sum((w - W_deng) ** 2)  # 保持与邓熵权重的一致性
        return fit_term + entropy_term + deng_term

    # 5. 带约束优化
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.01, 1)] * len(S_sigmoid)  # 避免零权重
    w_init = (S_sigmoid + alpha * W_deng) / (1 + alpha)  # 智能初始化

    res = minimize(
        objective,
        w_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8}
    )
    return res.x

def conflict_degree(S):
    """基于相似度矩阵计算冲突指标"""
    return 1 - np.min(np.sum(S, axis=1)) / len(S)

def enhanced_weight_framework(S, evidence, tau0=0.5, k=5.0, C0=0.3):
    """
    双通道动态权重框架
    参数：
    - tau0: 基础温度系数
    - k: 融合曲线陡度
    - C0: 冲突度阈值
    """
    # 1. 计算邓熵和冲突度
    E = np.array([deng_entropy(ev) for ev in evidence])
    C = conflict_degree(S)

    # 2. 双通道权重计算
    # 通道1：相似度权重
    S_agg = np.log(1 + np.sum(np.exp(2 * S), axis=1)) / 2  # 平滑聚合
    W_sim = softmax(S_agg)

    # 通道2：邓熵权重（温度自适应）
    tau = tau0 * (1 + C)
    W_deng = softmax(-E / tau)

    # 3. 动态融合
    gamma = 1 / (1 + np.exp(-k * (C - C0)))  # 冲突越大，邓熵权重占比越高
    W_hybrid = gamma * W_sim + (1 - gamma) * W_deng

    # 4. 后处理优化
    def refine_weights(w):
        # 惩罚过高权重（防止单一证据主导）
        return np.sum(w ** 2) + 0.1 * np.sum((w - 1 / len(w)) ** 2)

    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(refine_weights, W_hybrid,
                   bounds=[(0.01, 0.5)] * len(W_hybrid),  # 限制最大权重
                   constraints=cons)
    return res.x

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


import numpy as np


def deng_entropy(mass_func):
    """计算邓熵（未归一化版本）"""
    entropy = 0.0
    for A, m_A in mass_func.items():
        if m_A > 0:
            card = len(A)
            entropy -= m_A * np.log2(m_A / (2 ** card - 1))
    return entropy


def entropy_based_discount(evidence):
    """基于邓熵计算证据折扣因子"""
    # 计算每个证据体的邓熵
    entropies = [deng_entropy(ev) for ev in evidence]
    max_entropy = max(entropies) if entropies else 1.0

    # 熵越高，折扣越大（不确定性惩罚）
    discounts = [1 - (e / (max_entropy + 1e-10)) ** 0.5 for e in entropies]  # 平方根软化折扣
    return discounts

def joint_confidence(mass_func):
    """综合冲突与不确定性的置信度度量"""
    deng_e = deng_entropy(mass_func)
    # 计算证据特异性（冲突指标）
    specificity = 1 - len([k for k in mass_func.keys() if len(k) > 1]) / len(mass_func)
    return np.exp(-deng_e) * specificity  # 联合度量

def dynamic_weight_adjustment(R_values, evidence):
    """融合相似度权重与置信度"""
    confidences = [joint_confidence(ev) for ev in evidence]
    # 非线性融合（可调参数β控制权重）
    β = 0.7
    adjusted_weights = R_values**β * np.array(confidences)**(1-β)
    return adjusted_weights / np.sum(adjusted_weights)


def enhanced_slf(omega, evidence, R_values, alpha):
    # 动态权重调整
    R_adj = dynamic_weight_adjustment(R_values, evidence)

    # 概率收集（考虑焦元包含关系）
    prob_values = [sum(v for k, v in ev.items() if omega in k) for ev in evidence]

    # 熵感知排序
    sort_metric = [p * R_adj[i] for i, p in enumerate(prob_values)]
    sorted_idx = np.argsort(sort_metric)[::-1]

    # 改进累积权重计算
    w = []
    cum_R = 0
    for i in range(len(evidence)):
        cum_R += R_adj[sorted_idx[i]]
        w.append(cum_R ** ((1 - alpha) / alpha) - (cum_R - R_adj[sorted_idx[i]]) ** ((1 - alpha) / alpha))

    # 计算似然值
    return sum(w[i] * np.prod([prob_values[sorted_idx[k]] for k in range(i + 1)])
               for i in range(len(evidence)))

def enhanced_soft_likelihood(omega, evidence, R_values, alpha):
    """
    改进的软似然函数（邓熵修正版）
    参数：
    - omega: 待评估假设（如'A'）
    - evidence: 归一化证据列表 [{A:0.5, B:0.3}, ...]
    - R_values: 原始可靠性权重数组
    - alpha: 乐观系数（0-1）
    返回：
    - 修正后的软似然值
    """
    N = len(evidence)

    # 1. 计算邓熵折扣因子
    discounts = entropy_based_discount(evidence)

    # 2. 熵修正后的权重
    R_entropy = R_values * (1 - np.array(discounts))  # 不确定性越高，有效权重越低
    print("R_entropy:", R_entropy)
    # 3. 概率值收集（考虑焦元包含关系）
    prob_values = []
    for ev in evidence:
        # 包含omega的所有焦元的概率和（如omega='A'时，统计'A'和'AB'等）
        p = sum(v for k, v in ev.items() if omega in k)
        prob_values.append(p)

    # 4. 熵感知的排序指标（概率*修正后权重）
    sort_metric = [p * R_entropy[i] for i, p in enumerate(prob_values)]
    sorted_indices = np.argsort(sort_metric)[::-1]  # 降序
    # sorted_indices = np.argsort([prob_values[i] * R_values[i] for i in range(N)])[::-1]

    # 5. 改进权重计算（考虑折扣后的累积权重）
    w = []
    for i in range(N):
        if i == 0:
            cum_weight = R_entropy[sorted_indices[i]] ** ((1 - alpha) / alpha)
        else:
            sum_k = np.sum(R_entropy[sorted_indices[:i + 1]])
            sum_k_prev = np.sum(R_entropy[sorted_indices[:i]])
            cum_weight = sum_k ** ((1 - alpha) / alpha) - sum_k_prev ** ((1 - alpha) / alpha)
        w.append(cum_weight)

    # 6. 计算修正似然值
    L = 0.0
    for i in range(N):
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
    hypotheses = {'A', 'B', 'C'}
    fused = {}

    for omega in hypotheses:
        L = enhanced_slf(omega, evidence, R, alpha)
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
    S = owa_similarity_matrix(normalized_evidence)

    # 计算权重 R
    print(S)
    R =enhanced_weight_framework(S,evidence)
    R_cre = compute_credibility_from_similarity(S)

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

    # 相似度引导的乘积融合
    lambda_ = 0.2  # 置信度微调系数
    w_fused = R * (1 + lambda_ * R_cre)
    w_fused /= np.sum(w_fused)  # 归一化

    # 计算融合概率
    alpha = 0.1  # 乐观系数
    fused_probabilities = fuse_evidence(normalized_evidence, w_fused, alpha)
    # fused_probabilities_cre = fuse_evidence(normalized_evidence, R_cre, alpha)

    print("Optimal R:", R)
    print("Optimal R_fused:", w_fused)
    print("Fused probabilities (after SLF):", fused_probabilities)
    # print("Optimal R_cre:", R_cre)
    # print("Fused probabilities (after SLF_cre):", fused_probabilities_cre)
    # Step 3-2: 多次融合证据
    # fused_result_after_slf_cre =dempster_combine(fused_probabilities,fused_probabilities_cre)
    # print("fused_result_after_slf_cre:", fused_result_after_slf_cre)
    return fused_probabilities

# 你的证据体
# evidence = [
#     {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
#     {('B',): 0.5, ('C',): 0.5},
#     {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
#     {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},
#     {('A', 'C'): 0.8, ('B',): 0.2}
# ]

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

# evidence = [
#     {('A',): 0.0437, ('B',): 0.3346, ('C',): 0.2916, ('A', 'B'): 0.0437, ('A', 'C'): 0.0239, ('B', 'C'): 0.2385, ('A', 'B', 'C'): 0.0239},  # 证据体 1 (Sepal Length)
#     {('A',): 0.0865, ('B',): 0.2879, ('C',): 0.1839, ('A', 'B'): 0.0863, ('A', 'C'): 0.0865, ('B', 'C'): 0.1825, ('A', 'B', 'C'): 0.0863},  # 证据体 2 (Sepal Width)
#     {('A',): 1.4e-09, ('B',): 0.6570, ('C',): 0.1726, ('A', 'B'): 1.3e-09, ('A', 'C'): 1.4e-11, ('B', 'C'): 0.1704, ('A', 'B', 'C'): 1.4e-11},  # 证据体 3 (Petal Length)
#     {('A',): 8.20e-06, ('B',): 0.6616, ('C',): 0.1692, ('A', 'B'): 8.20e-06, ('A', 'C'): 3.80e-06, ('B', 'C'): 0.1692, ('A', 'B', 'C'): 3.80e-06}   # 证据体 4 (Petal Width)
# ]

# 应用一的数据
# evidence = [
#     {('A',): 0.40, ('B',): 0.28, ('C',): 0.30, ('A', 'C'): 0.02},  # S₁
#     {('A',): 0.01, ('B',): 0.90, ('C',): 0.08, ('A', 'C'): 0.01},  # S₂
#     {('A',): 0.63, ('B',): 0.06, ('C',): 0.01, ('A', 'C'): 0.30},  # S₃
#     {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.30},  # S₄
#     {('A',): 0.60, ('B',): 0.09, ('C',): 0.01, ('A', 'C'): 0.30}   # S₅
# ]
#
# # 应用二的数据
evidence = [
    {('A',): 0.7, ('B',): 0.1, ('A','B','C'): 0.2},  # 证据体 m1
    {('A',): 0.7, ('A','B','C'): 0.3},               # 证据体 m2
    {('A',): 0.65, ('B',): 0.15, ('A','B','C'): 0.20},  # 证据体 m3
    {('A',): 0.75, ('C',): 0.05, ('A','B','C'): 0.20},  # 证据体 m4
    {('B',): 0.20, ('C',): 0.80}                      # 证据体 m5
]
main_function(evidence)