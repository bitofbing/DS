import matplotlib.pyplot as plt
from collections.abc import Iterable
import copy
from sklearn.model_selection import RepeatedKFold
from typing import Dict, List, Tuple, Union
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import itertools
from string import ascii_uppercase
import numpy as np

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
        term = mass * math.log2(log_arg)  # term本身是负的
        total_entropy -= term  # 负负得正
        # print(f"组合 {items}: mass={mass}, log参数={log_arg:.4f}, term={term:.4f}, 贡献={-term:.4f}")
    # hypothesis_complexity = 1 - 1 / len(evidence)
    #     print(f"证据 {evidence}的总rps熵: total_entropy={total_entropy}")
    # 标准化到0-1范围（熵值越小质量越高）
    # print("total_entropy",total_entropy)
    max_possible_entropy = math.log(len(evidence) * 10)  # 经验值
    normalize_entropy = min(total_entropy / max_possible_entropy, 1.0)
    # print(f"证据 {evidence}的标准化熵: normalize_entropy={normalize_entropy}")
    return total_entropy

# ================== 缓存字母-数字映射 ==================
_letter_cache = {ch: idx + 1 for idx, ch in enumerate(ascii_uppercase)}


def to_numeric(x):
    """字母/数字映射，使用缓存加速"""
    if isinstance(x, str) and x.isalpha() and x in _letter_cache:
        return _letter_cache[x]
    return int(x)


# ================== 单例关系计算 ==================
_relation_cache = {}


def compute_singleton_relation(i, j):
    """单例-单例关系计算，带缓存"""
    key = tuple(sorted([i, j]))
    if key in _relation_cache:
        return _relation_cache[key]
    a, b = to_numeric(i), to_numeric(j)
    val = 1 - np.exp(-abs(a - b))
    _relation_cache[key] = val
    return val


# ================== 单例-集合关系 ==================
def compute_singleton_set(A, B, gamma):
    """单例到集合的关系，向量化"""
    B_numeric = np.array([to_numeric(b) for b in B])
    a_numeric = to_numeric(A)
    # 一次性计算所有 r(a,b)
    r_values = 1 - np.exp(-np.abs(a_numeric - B_numeric))
    return owa_aggregation(r_values, gamma, force_binary=False)


# ================== OWA聚合 ==================
def owa_aggregation(values, gamma=0.8, force_binary=False):
    """解析解版本 OWA 聚合"""
    values = np.array(values)
    n = len(values)
    if n == 1:
        return values[0]
    if force_binary:
        sorted_v = np.sort(values)[::-1]
        return gamma * sorted_v[0] + (1 - gamma) * sorted_v[1]

    # ===== 解析解求权重 =====
    # 原约束: sum(w)=1, sum((n-k-1)/(n-1)*w[k]) = gamma
    # 对于任意 n，可以解出 w1..wn 的解析解
    sorted_values = np.sort(values)[::-1]
    w = np.zeros(n)
    w[0] = gamma
    if n > 1:
        w[1:] = (1 - gamma) / (n - 1)
    return np.dot(sorted_values, w)


# ================== 构建关系矩阵 ==================
def build_relation_matrix(focal_elements, gamma=0.8):
    """构建关系矩阵，向量化 + 缓存"""
    n = len(focal_elements)
    R_o = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            A, B = focal_elements[i], focal_elements[j]

            # 单例-单例
            if len(A) == 1 and len(B) == 1:
                a, b = next(iter(A)), next(iter(B))
                R_o[i, j] = compute_singleton_relation(a, b)

            # 单例-集合
            elif len(A) == 1 and len(B) > 1:
                a = next(iter(A))
                R_o[i, j] = compute_singleton_set(a, B, gamma)

            # 集合-集合
            elif len(A) > 1 and len(B) > 1:
                # 向量化: 对集合A每个元素调用 compute_singleton_set
                r_values = np.array([compute_singleton_set(a, B, gamma) for a in A])
                R_o[i, j] = owa_aggregation(r_values, gamma, force_binary=False)

    # 对称填充
    R_o = np.triu(R_o) + np.triu(R_o, 1).T
    return R_o


# ================== 证据体距离 ==================
def d_OWA(m1, m2, gamma=0.8):
    """计算两个证据体之间的距离"""
    # 获取所有焦元（保持顺序）
    focal_elements = sorted(list(set(m1.keys()).union(set(m2.keys()))), key=lambda x: (len(x), x))

    # 构建关系矩阵
    R_o = build_relation_matrix(focal_elements, gamma)

    # 转向量
    m1_vec = np.array([m1.get(f, 0) for f in focal_elements])
    m2_vec = np.array([m2.get(f, 0) for f in focal_elements])

    # 计算距离
    diff = m1_vec - m2_vec
    # M = np.eye(len(R_o)) - R_o
    M = 1 - R_o
    result = 0.5 * diff.T @ M @ diff
    # distance = np.sqrt(max(result, 0))  # 防止负数
    return result


# ================== 相似度矩阵 ==================
def compute_similarity_matrix(evidence_list, gamma=0.8):
    """
    向量化计算相似度矩阵，处理负值情况
    """
    n = len(evidence_list)
    distance_matrix = np.zeros((n, n))

    # 计算距离矩阵
    for i in range(n):
        for j in range(i + 1, n):
            distance = d_OWA(evidence_list[i], evidence_list[j], gamma)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # 计算相似度并处理边界
    similarity_matrix = 1 - distance_matrix
    # similarity_matrix = np.maximum(similarity_matrix, 0)  # 所有值不小于0

    # 确保对角线为1（自相似度）
    np.fill_diagonal(similarity_matrix, 1)

    return similarity_matrix


def compute_credibility_from_similarity(S):
    """
    计算归一化的可信度权重 Crd
    - 输入: S (相似度矩阵)
    - 输出: Crd (可信度权重向量)
    """
    # 计算支持度 Sup(m_i)
    Sup = np.sum(S, axis=1) - np.diag(S)  # 去掉自身对自身的支持度
    # Sup = np.sum(S, axis=1)
    # 计算可信度 Crd 并归一化
    Crd = Sup / np.sum(Sup) if np.sum(Sup) != 0 else np.ones_like(Sup) / len(Sup)  # 避免除零

    return Crd

def compute_consistency_with_std(S):
    """
    计算一致性：与其它源的相似度均值，并返回标准差

    参数:
        S: 相似度矩阵，对角线元素为1

    返回:
        consistency_means: 每个证据与其他证据的平均相似度
        consistency_std: 一致性标准差（反映证据间的一致性差异）
    """
    n = S.shape[0]

    # 计算每个证据与其他证据的平均相似度（排除对角线）
    consistency_means = []
    for i in range(n):
        # 获取第i个证据与其他所有证据的相似度（排除自身）
        other_similarities = np.concatenate([S[i, :i], S[i, i+1:]])
        mean_similarity = np.mean(other_similarities)
        consistency_means.append(mean_similarity)

    consistency_means = np.array(consistency_means)

    # 计算一致性标准差
    consistency_std = np.std(consistency_means)

    return consistency_means, consistency_std

def compute_certainty(evidence):
    """
    计算确定性指标（最大概率）
    参数:
        evidence: 单个RPS证据体
    返回:
        certainty: 确定性指标（0-1之间）
    """
    prob = np.array(list(evidence.values()))
    prob = prob / (np.sum(prob) + 1e-10)
    certainty = np.max(prob)
    return certainty


def geometric_mean_correction(Crd, rps_weights_norm):
    """
    使用几何平均修正Crd
    """
    # 计算熵质量指标
    # entropy_quality = 1 - rps_weights_norm
    Crd = np.asarray(Crd, dtype=float)
    rps_weights_norm = np.asarray(rps_weights_norm, dtype=float)
    # 计算几何平均：sqrt(Crd * 熵质量)
    geometric_mean = np.sqrt(np.abs(Crd * rps_weights_norm))
        # 计算 Crd 的均值和方差
    mean_crd = np.mean(Crd)
    var_crd = np.var(Crd)

    # 利用均值和方差作为全局修正系数
    # 均值增强稳定性，方差增强区分度
    correction_factor = np.sqrt(mean_crd / (var_crd + 1e-8))
    print("correction_factor",correction_factor)
    # 应用修正
    adjusted = geometric_mean * correction_factor
    # 归一化
    adjusted_Crd = adjusted / np.sum(adjusted)

    return adjusted_Crd

def geometric_mean_correction_with_stats(Crd, rps_weights_norm, eps=1e-12):
    """
    用几何均值 + Crd 的均值/方差构造逐元素修正因子（无条件判断版本）。
    返回归一化后的向量（和为1）。
    """
    Crd = np.asarray(Crd, dtype=float)
    rps = np.asarray(rps_weights_norm, dtype=float)

    # 基础几何项（你之前用 abs 的版本效果不错）
    # 在乘法内加 eps 保证非负且避免 sqrt(0) 对数问题
    geometric_mean = np.sqrt(np.abs(Crd * rps))

    # 全局统计量（自适应，无手动参数）
    mu = Crd.mean()
    sigma = Crd.std() + eps  # 防止除零

    # 标准化偏差 z（可正可负）
    # z = (Crd - mu) / sigma
    Delta_Cred_var = mu - Crd * (sigma / (mu + 1e-10))
    # 自适应系数 k：随分散度增加而增强调整力（无硬超参）
    k = 1.0 + (sigma / (mu + eps))

    M = k * Delta_Cred_var

    # 逐元素修正因子（z 越小 -> 因子越小）
    factor = np.exp(-M)

    # 应用逐元素修正并归一化（确保输出是概率向量）
    adjusted = geometric_mean * factor
    W = adjusted / (adjusted.sum() + eps)

    return W

def geometric_mean_correction_with_stats2(Crd, rps_weights_norm, eps=1e-12):
    """
    用几何均值 + Crd 的均值/方差构造逐元素修正因子（无条件判断版本）。
    返回归一化后的向量（和为1）。
    """
    Crd = np.asarray(Crd, dtype=float)
    rps = np.asarray(rps_weights_norm, dtype=float)

    # 全局统计量（自适应，无手动参数）
    mu = Crd.mean()
    sigma = Crd.std() + eps  # 防止除零

    # 标准化偏差 z（可正可负）
    # z = (Crd - mu) / sigma
    Delta_Cred_var = mu - Crd * (sigma / (mu + 1e-10))
    # 自适应系数 k：随分散度增加而增强调整力（无硬超参）
    k = 1 + (sigma / (mu + eps))

    M = k * Delta_Cred_var

    # 逐元素修正因子（z 越小 -> 因子越小）
    factor = np.exp(-Delta_Cred_var)
    # factor = np.power(rps,Delta_Cred_var)

    # 基础几何项（你之前用 abs 的版本效果不错）
    # 在乘法内加 eps 保证非负且避免 sqrt(0) 对数问题
    geometric_mean = np.sqrt(np.abs(Crd * factor)) * rps
    # 应用逐元素修正并归一化（确保输出是概率向量）
    # adjusted = geometric_mean * factor
    W = geometric_mean / (geometric_mean.sum() + eps)

    return W
def geometric_mean_correction_adjust(Crd, rps_weights_norm):
    """
    使用几何平均修正Crd
    """
    # 计算熵质量指标
    # entropy_quality = 1 - rps_weights_norm
    Crd = np.asarray(Crd, dtype=float)
    rps_weights_norm = np.asarray(rps_weights_norm, dtype=float)
    # 计算几何平均：sqrt(Crd * 熵质量)
        # 平移到正数域：始终减去 min(Crd)，再加上 eps
    geometric_mean = np.sqrt(np.abs(Crd * rps_weights_norm))
        # 计算 Crd 的均值和方差
    mean_crd = np.mean(Crd)
    var_crd = np.std(Crd)

    # 利用均值和方差作为全局修正系数
    # 均值增强稳定性，方差增强区分度
    correction_factor = np.sqrt(mean_crd / (var_crd + 1e-8))

    # 应用修正
    adjusted = geometric_mean * correction_factor
    # 归一化
    adjusted_Crd = adjusted / np.sum(adjusted)

    return adjusted_Crd
def compute_weights_with_both_metrics(S, evidence_list):
    n = len(evidence_list)
    # 第一步：计算基础可信度 (公式6-4)
    Crd = compute_credibility_from_similarity(S)
    # print("Crd；" , Crd)
    # 第二步：计算RPS熵和归一化熵值 (公式6-5)
    rps_weights = np.array([compute_rps_entropy(ev) for ev in evidence_list])
    # print("rps_weights；" , rps_weights)
    rps_weights_norm = rps_weights / (np.sum(rps_weights) + 1e-10)
    # print("rps_weights_norm；" , rps_weights_norm)

    # 一致性：与其它源的相似度均值
    # Consistency = np.mean(S, axis=1)
# 计算一致性均值和标准差
    consistency_means, consistency_std = compute_consistency_with_std(S)
    # print("Consistency；" , Consistency)
    # 第四步：计算可信度差值 (公式6-7)
    mean_cred = Crd.mean()
    # print("mean_cred；", mean_cred)
    Delta_Cred = mean_cred - Crd
    std_cred = Crd.std()   # 缓存标准差
    # print("std_cred；", std_cred)
    Delta_Cred_var = mean_cred - Crd * (std_cred / (mean_cred + 1e-10))
    # print("std_cred / (mean_cred + 1e-10)" , std_cred / (mean_cred + 1e-10))
    # print("Delta_Cred；" , Delta_Cred)
    # print("Delta_Cred_var；" , Delta_Cred_var)
    base_correction = np.power(rps_weights_norm, Delta_Cred_var)
    # print("base_correction；", base_correction)
    # 第六步：计算修正后的可信度 (公式6-6扩展)
    Cred_H = Crd * base_correction
    # print("Cred_H",Cred_H)
    adjusted_Crd = geometric_mean_correction_with_stats2(Crd, rps_weights_norm)
    W = adjusted_Crd
    # 第七步：最终权重归一化 (公式6-8)
    # W = Cred_H / (np.sum(Cred_H) + 1e-10)

    return W

def similarity_based_soft_likelihood(omega, evidence_list, R_values, alpha=0.1, verbose=False):
    """
    Compute the soft likelihood (SLF) for a given hypothesis omega with detailed step prints.

    Parameters
    ----------
    omega : hashable
        The hypothesis to evaluate (e.g., 'W', 'X', etc.).
    evidence_list : list of dict
        List of RPS evidence, each as a dictionary mapping hypotheses to probabilities.
    R_values : list or np.ndarray
        Reliability degrees corresponding to each evidence source.
    alpha : float, optional
        Optimism coefficient for OWA weight calculation (default 0.1).
    verbose : bool, optional
        If True, prints detailed intermediate computation steps.

    Returns
    -------
    L : float
        Computed soft likelihood for hypothesis omega.
    """
    # Step 1: Extract support values
    prob_values = np.array([ev.get(omega, 0.0) for ev in evidence_list])
    weighted_support = prob_values * R_values

    if verbose:
        print(f"\nHypothesis: {omega}")
        print(f"Original support values: {prob_values}")

        print("\nWeighted support calculation:")
        for i, (p, r, ws) in enumerate(zip(prob_values, R_values, weighted_support)):
            print(f"  Source {i+1}: prob={p:.4f} * R={r:.4f} = {ws:.4f}")
        print(f"Weighted support (all): {weighted_support}")

    # Step 2: Sort by weighted support descending
    sorted_indices = np.argsort(weighted_support)[::-1]
    sorted_prob = prob_values[sorted_indices]
    sorted_R = np.array(R_values)[sorted_indices]

    if verbose:
        print(f"\nSorted indices (by weighted support): {sorted_indices}")
        print(f"Sorted probability values: {sorted_prob}")
        print(f"Sorted reliability values: {sorted_R}")

    # Step 3: Compute OWA weights with detailed printing
    w = []
    # print("\nStep 3: Compute OWA weights")
    power = (1 - alpha) / alpha
    for i in range(len(evidence_list)):
        if i == 0:
            weight = sorted_R[i] ** power
            # print(f"  i={i}, weight = sorted_R[{i}]^{power:.1f} = {sorted_R[i]:.4f}^{power:.1f} -> {weight:.4e}")
        else:
            cum_sum = np.sum(sorted_R[:i+1])
            cum_sum_prev = np.sum(sorted_R[:i])
            # print(f"  i={i}, cumulative sum details:")
            # for j in range(i+1):
                # print(f"     cum_sum += sorted_R[{j}] = {sorted_R[j]:.4f} -> {np.sum(sorted_R[:j+1]):.4f}")
            # for j in range(i):
                # print(f"     cum_sum_prev += sorted_R[{j}] = {sorted_R[j]:.4f} -> {np.sum(sorted_R[:j+1]):.4f}")
            weight = cum_sum ** power - cum_sum_prev ** power
            # print(f"       weight = {cum_sum:.4f}^{power:.1f} - {cum_sum_prev:.4f}^{power:.1f} = {weight:.4e}")
        w.append(weight)
    w = np.array(w)

    if verbose:
        print(f"\nComputed OWA weights: {w}")

    # Step 4: Compute soft likelihood with detailed intermediate steps
    L = 0.0
    # print("\nStep 4: Compute soft likelihood")
    for i in range(len(evidence_list)):
        prod = 1.0
        # print(f"\n  Step {i+1}:")
        for j in range(i+1):
            prod *= sorted_prob[j]
            # print(f"    Multiply prob[{j}]={sorted_prob[j]:.4f} -> partial product = {prod:.4f}")
        L += w[i] * prod
        # print(f"    Multiply by w[{i}]={w[i]:.4e} -> contribution to L = {w[i] * prod:.4e}")
        # print(f"    Cumulative L = {L:.4e}")

    if verbose:
        print(f"\nFinal soft likelihood L({omega}) = {L:.4f}\n")

    return L

def similarity_based_soft_likelihood_youhua(omega, evidence_list, R_values, alpha=0.1, verbose=False):
    """
    Optimized soft likelihood computation using vectorization.
    """
    # Step 1: Extract support values
    prob_values = np.array([ev.get(omega, 0.0) for ev in evidence_list])
    R_values = np.array(R_values)
    weighted_support = prob_values * R_values

    # Step 2: Sort by weighted support descending
    sorted_indices = np.argsort(weighted_support)[::-1]
    sorted_prob = prob_values[sorted_indices]
    sorted_R = R_values[sorted_indices]

    # Step 3: Vectorized OWA weight calculation
    power = (1 - alpha) / alpha
    cum_R = np.cumsum(sorted_R)
    cum_R_prev = np.insert(cum_R[:-1], 0, 0.0)  # 前一个累积和
    w = cum_R**power - cum_R_prev**power  # 向量化权重计算

    # Step 4: Vectorized soft likelihood computation
    cumprod_prob = np.cumprod(sorted_prob)  # 累乘概率
    L = np.dot(w, cumprod_prob)  # 最终加权累乘求和

    if verbose:
        print(f"Hypothesis: {omega}")
        print(f"Sorted probabilities: {sorted_prob}")
        print(f"Sorted reliability: {sorted_R}")
        print(f"OWA weights: {w}")
        print(f"Cumulative products: {cumprod_prob}")
        print(f"Soft likelihood: {L}")

    return L

def absolute_weighted_sum_amplify(evidence_list, weights, pes):
    # weights: numpy array
    mean_w = weights.mean()
    w_amp = weights / (mean_w + 1e-12)      # >1 for w>mean, <1 for w<mean
    S = {}
    for subset in pes:
        vals = [e.get(subset, 0.0) for e in evidence_list]
        S[subset] = sum(w * v for w, v in zip(w_amp, vals))
    # 全局归一化（如果你需要概率分布）
    total = sum(S.values())
    if total > 0:
        for k in S:
            S[k] /= total
    return S

def absolute_weighted_sum_amplify_log(evidence_list, weights, pes, eps=1e-9):
    mean_w = weights.mean()
    # 对数放大：避免极端比例
    w_amp = np.log1p(weights) / np.log1p(mean_w + 1e-12)
    # ---- 双非线性权重放大 ----
    # w_amp = (np.log1p(weights) * np.sqrt(weights + 1)) / (np.log1p(mean_w) * np.sqrt(mean_w + 1e-12))
    # print("w_amp:", w_amp)
    S = {}
    for subset in pes:
        vals = [e.get(subset, 0.0) for e in evidence_list]
        S[subset] = sum(w * v for w, v in zip(w_amp, vals))

        # S[subset] = sum(w * v * subset_factor for w, v in zip(w_amp, vals))
    total = sum(S.values())
    if total > 0:
        for k in S:
            S[k] /= total
    return S


def weight_based_rps_sampling(evidence_list, weights, pes, num_samples=1000):
    """
    基于权重采样生成平均RPS
    """
    n = len(evidence_list)

    # 归一化权重作为采样概率
    # sample_probs = weights / np.sum(weights)

    # 多次采样并平均
    sampled_rps_list = []

    for _ in range(num_samples):
        # 根据权重采样一个证据
        sampled_idx = np.random.choice(n, p=weights)
        sampled_ev = evidence_list[sampled_idx]

        # 添加到采样列表
        sampled_rps_list.append(sampled_ev)

    # 计算采样结果的简单平均
    avg_rps = defaultdict(float)
    for ev in sampled_rps_list:
        for subset in pes:
            avg_rps[subset] += ev.get(subset, 0.0)

    for subset in pes:
        avg_rps[subset] /= num_samples

    # 清理极小值
    avg_rps = {k: v for k, v in avg_rps.items() if v > 1e-5}

    # 归一化
    total = sum(avg_rps.values())
    if total > 0:
        for k in avg_rps:
            avg_rps[k] /= total

    return dict(avg_rps)

def weighted_consensus_rps(evidence_list, weights, pes):
    mean_w = weights.mean()
    # 对数放大：避免极端比例
    w_amp = np.log1p(weights) / np.log1p(mean_w + 1e-12)
    n = len(evidence_list)
    S = {subset: 0.0 for subset in pes}
    for i in range(n):
        for j in range(n):
            for subset in pes:
                avg_ev = np.mean([evidence_list[i].get(subset, 0.0), evidence_list[j].get(subset, 0.0)])
                S[subset] += w_amp[i] * w_amp[j] * avg_ev
    total = sum(S.values())
    for k in S:
        S[k] /= (total + 1e-12)
    return S

def weighted_consensus_rps_geo2(evidence_list, weights, pes):
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()  # 归一化权重

    S = {}
    for subset in pes:
        vals = np.array([e.get(subset, 0.0) for e in evidence_list])
        # log 加权几何平均
        # 算术平均
        vals = np.array([e.get(subset, 0.0) for e in evidence_list])

        # 使用 soft log，避免极端小值导致强偏差
        geo = np.exp(np.sum(weights * np.log1p(vals))) -1
        S[subset] = max(geo, 0.0)

    total = sum(S.values())
    for k in S:
        S[k] /= (total + 1e-12)
    return S

def absolute_weighted_sum_amplify_log_wuzhe(evidence_list, weights, pes):
    """
    改进版：对数放大 + 自适应平滑因子（无固定 lam）
    """
    weights = np.asarray(weights, dtype=float)
    mean_w = weights.mean()
    if mean_w <= 1e-12:
        mean_w = 1e-12

    # 对数放大（避免极端值）
    w_amp = np.log1p(weights) / np.log1p(mean_w)

    # 权重归一化（保持均值=1）
    w_amp /= w_amp.mean() + 1e-12

    # 自适应平滑因子（随权重分布调整，范围 0~0.3）
    lam = np.clip(weights.std() / (mean_w + 1e-12), 0.0, 0.3)

    # 融合平滑，避免过度倾斜
    w_amp = (1 - lam) * w_amp + lam

    S = {}
    for subset in pes:
        vals = [e.get(subset, 0.0) for e in evidence_list]
        # 加极小项避免 0/NaN
        S[subset] = sum(w * v for w, v in zip(w_amp, vals)) + 1e-12

    total = sum(S.values())
    if total > 0:
        for k in S:
            S[k] /= total
    else:
        n = len(S)
        if n > 0:
            for k in S:
                S[k] = 1.0 / n
    return S


def absolute_weighted_sum_amplify_log_stable(evidence_list, weights, pes):
    weights = np.asarray(weights, dtype=float)

    # 鲁棒中心化：用中位数代替均值
    center = np.median(weights)
    if center <= 1e-12:
        center = 1e-12

    # log 放大 + 方差正则化
    w_amp = np.log1p(weights) / np.log1p(center)
    w_amp /= (1.0 + weights.var())

    # 归一化（保持均值=1）
    w_amp /= w_amp.mean() + 1e-12

    # 证据融合
    S = {}
    for subset in pes:
        vals = [e.get(subset, 0.0) for e in evidence_list]
        S[subset] = sum(w * v for w, v in zip(w_amp, vals))

    # softmax 归一化（保证跨折次分布一致）
    vals = np.array(list(S.values()))
    exp_vals = np.exp(vals - np.max(vals))  # 防溢出
    norm_vals = exp_vals / (exp_vals.sum() + 1e-12)

    return {k: v for k, v in zip(S.keys(), norm_vals)}

def fuse_evidence(evidence, R, hypotheses = {'A', 'B', 'C'}, alpha=0.1):
    """
    计算融合概率：
    - 使用软似然函数进行最终的概率估计
    - 输入: evidence (列表), 归一化后的证据列表；R (数组), 权重；alpha (浮点数), 乐观系数。
    - 输出: 融合后的概率分布。
    """
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
        L = similarity_based_soft_likelihood_youhua(omega, evidence, R, alpha)
        fused[omega] = L

    # print("fused_result_before_nor:", convert_numpy_types(fused))
    # 归一化
    total_L = sum(fused.values()) + 1e-10
    fused = {k: v / total_L for k, v in fused.items()}
    # print("fused_result_after_nor:", convert_numpy_types(fused))
    return fused

def calculate_OPT(pmf_data, hypotheses={'A', 'B', 'C'}):
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

def convert_numpy_types(obj):
    """
    递归转换数据结构中的 np.int64 和 np.float64 为 Python 原生类型
    支持处理: 列表/元组/集合/字典/嵌套结构
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (str, bytes, bytearray)):
        return obj
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return type(obj)(convert_numpy_types(x) for x in obj)
    else:
        return obj
def batch_OPT(data_list, hypotheses = {'A', 'B', 'C','D'}):
    opt_list = []
    for d in data_list:
        opt_one  = calculate_OPT(d, hypotheses)
        print("opt_one:", convert_numpy_types(opt_one))
        opt_list.append(opt_one)
    return opt_list

# -------------------------------
# 基础交集函数
# -------------------------------
def right_intersection(A, B):
    """
    右正交 (RI)，即 B 中去除不在 A 中的元素
    """
    return tuple(item for item in B if item in A)

def left_intersection(A, B):
    """
    左正交 (LI)，即 A 中去除不在 B 中的元素
    """
    return tuple(item for item in A if item in B)

# -------------------------------
# K 值计算
# -------------------------------
def calculate_KR(M1, M2):
    """
    计算右正交和的 K^R (K_R)
    输入: [(集合, 权重), ...]
    """
    K_R = 0
    for B, w1 in M1:
        for C, w2 in M2:
            if right_intersection(B, C) == ():
                K_R += w1 * w2
    return K_R

def calculate_KL(M1, M2):
    """
    计算左正交和的 K^L (K_L)
    """
    K_L = 0
    for B, w1 in M1:
        for C, w2 in M2:
            if left_intersection(B, C) == ():
                K_L += w1 * w2
    return K_L

# -------------------------------
# ROS 与 LOS
# -------------------------------
def ROS(M1, M2):
    """
    右正交和 (ROS)
    输入: [(集合, 权重), ...]
    输出: {集合: 权重, ...}，保留 0 值
    """
    K_R = calculate_KR(M1, M2)
    result = defaultdict(float)

    if K_R != 1:  # 防止除以 0
        # 遍历所有可能交集
        all_keys = set(A for A, _ in M1) | set(C for C, _ in M2)
        for A in all_keys:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if right_intersection(B, C) == A:
                        weight_sum += w1 * w2
            result[A] = (1 / (1 - K_R)) * weight_sum  # 即使 weight_sum=0 也保留

    return dict(result)

def LOS(M1, M2):
    """
    左正交和 (LOS)
    """
    K_L = calculate_KL(M1, M2)
    result = defaultdict(float)

    if K_L != 1:  # 防止除以 0
        all_keys = set(A for A, _ in M1) | set(C for C, _ in M2)
        for A in all_keys:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if left_intersection(B, C) == A:
                        weight_sum += w1 * w2
            result[A] = (1 / (1 - K_L)) * weight_sum

    return dict(result)

# -------------------------------
# 连续正交和
# -------------------------------
def dict_to_pmf(rps_dict):
    """
    将字典格式 {('A',): 0.99, ...} 转换为 [(集合, 权重), ...]
    """
    return [(k, v) for k, v in rps_dict.items()]

def continuous_right_orthogonal_sum(PMFs):
    """
    连续执行右正交和操作
    输入: [ {('A',):0.9,...}, {...}, ... ]
    输出: {集合: 权重, ...}
    """
    result = dict_to_pmf(PMFs[0])
    for i in range(1, len(PMFs)):
        result = ROS(result, dict_to_pmf(PMFs[i])).items()
    return dict(result)

def continuous_left_orthogonal_sum(PMFs):
    """
    连续执行左正交和操作
    """
    result = dict_to_pmf(PMFs[0])
    for i in range(1, len(PMFs)):
        result = LOS(result, dict_to_pmf(PMFs[i])).items()
    return dict(result)


def calculate_pic(probabilities: Union[Dict[str, float], List[float], np.ndarray],
                 elements: List[str] = None) -> float:
    """
    计算概率信息量 (Probabilistic Information Content - PIC)

    根据公式: PIC(P) = 1 + (1/log₂N) * Σ P(θ)log₂(P(θ))

    Args:
        probabilities: 概率分布，可以是:
            - 字典: {元素: 概率}
            - 列表: 概率值列表（需要提供elements参数）
            - numpy数组: 概率值数组（需要提供elements参数）
        elements: 可选，元素名称列表，当probabilities不是字典时需要提供

    Returns:
        float: PIC值，范围[0, 1]

    Raises:
        ValueError: 概率和为1或包含无效值
    """
    # 处理不同类型的输入
    if isinstance(probabilities, dict):
        probs = list(probabilities.values())
        N = len(probabilities)
    else:
        if elements is not None:
            N = len(elements)
        else:
            N = len(probabilities)
        probs = probabilities

    # 转换为numpy数组便于计算
    probs = np.array(probs)

    # 验证输入
    if not np.all(probs >= 0):
        raise ValueError("所有概率值必须非负")

    if not math.isclose(np.sum(probs), 1.0, rel_tol=1e-10):
        raise ValueError(f"概率和必须为1，当前和为{np.sum(probs):.6f}")

    if N <= 0:
        raise ValueError("元素数量必须大于0")

    # 计算PIC
    entropy_term = 0.0
    for p in probs:
        if p > 0:  # 避免log(0)
            entropy_term += p * math.log2(p)

    # PIC = 1 - 归一化香农熵
    pic_value = 1 + (1 / math.log2(N)) * entropy_term

    return pic_value

from typing import Dict, Tuple, List
from collections import defaultdict
import math
from itertools import permutations, combinations
def zhou_combination(M1_dict: Dict[Tuple, float], M2_dict: Dict[Tuple, float]) -> Dict[Tuple, float]:
    """
    通用的Zhou组合规则实现
    输入格式: {('A',): 0.7, ('B',): 0.15, ('C',): 0.15}
    输出格式: {排列元组: 质量值}
    """
    # 步骤1: 提取所有元素并生成幂集和幂排列集
    all_elements = _extract_all_elements(M1_dict, M2_dict)
    ps = _generate_power_set(all_elements)
    pes = _generate_power_permutation_set(all_elements)

    # 步骤2: 补全输入字典
    full_M1 = _complete_input_dict(M1_dict, all_elements)
    full_M2 = _complete_input_dict(M2_dict, all_elements)

    # 步骤3: 还原BPA分布
    bpa1 = _pmf_to_bpa(full_M1, ps)
    bpa2 = _pmf_to_bpa(full_M2, ps)

    # 步骤4: Dempster组合
    combined_bpa = _dempster_combination(bpa1, bpa2, ps)

    # 步骤5: 计算序一致度 π(B_i^j)
    pi_values = _calculate_pi_values(full_M1, full_M2, ps, all_elements)

    # 步骤6: 计算权重 w(B_i^j)
    w_values = _calculate_weight_values(pi_values, ps, all_elements)

    # 步骤7: 分配信念
    result_pmf = _distribute_belief(combined_bpa, w_values, ps, all_elements)

    # 步骤8: 归一化并返回结果
    return _normalize_result(result_pmf)

# ==================== 辅助函数 ====================

def _extract_all_elements(*dicts):
    """从所有字典中提取唯一元素"""
    all_elements = set()
    for d in dicts:
        for perm in d.keys():
            all_elements.update(perm)
    return sorted(all_elements)

def _generate_power_set(elements):
    """生成幂集 PS(Θ)"""
    ps = []
    for i in range(len(elements) + 1):
        for comb in combinations(elements, i):
            ps.append(tuple(sorted(comb)))
    return ps

def _generate_power_permutation_set(elements):
    """生成幂排列集 PES(Θ)"""
    pes = []
    for i in range(len(elements) + 1):
        for perm in permutations(elements, i):
            pes.append(perm)
    return pes

def _get_all_permutations(subset):
    """获取给定子集的所有排列"""
    if not subset:
        return [()]
    return list(permutations(subset))

def _kendall_tau_distance(perm1, perm2):
    """计算两个排列的肯德尔秩相关系数"""
    common_elements = set(perm1) & set(perm2)
    if len(common_elements) < 2:
        return 1.0 if len(common_elements) == 1 else 0.0

    concordant = 0
    discordant = 0
    common_list = list(common_elements)

    for i in range(len(common_list)):
        for j in range(i + 1, len(common_list)):
            a, b = common_list[i], common_list[j]

            # 获取在两个排列中的相对顺序
            try:
                a1, b1 = perm1.index(a), perm1.index(b)
                a2, b2 = perm2.index(a), perm2.index(b)

                if (a1 < b1 and a2 < b2) or (a1 > b1 and a2 > b2):
                    concordant += 1
                else:
                    discordant += 1
            except ValueError:
                continue

    total_pairs = len(common_list) * (len(common_list) - 1) // 2
    return (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

def _complete_input_dict(input_dict, elements):
    """补全输入字典，确保包含所有单元素排列"""
    completed = defaultdict(float)

    # 添加输入的所有项
    for perm, mass in input_dict.items():
        completed[perm] += mass

    # 确保所有单元素排列都存在（即使质量为0）
    for elem in elements:
        single_perm = (elem,)
        if single_perm not in completed:
            completed[single_perm] = 0.0

    return dict(completed)

def _pmf_to_bpa(pmf_dict, ps):
    """将PMF转换为BPA"""
    bpa = defaultdict(float)
    for subset in ps:
        bpa_value = 0.0
        for perm in _get_all_permutations(subset):
            bpa_value += pmf_dict.get(perm, 0.0)
        if bpa_value > 0:
            bpa[subset] = bpa_value
    return dict(bpa)

def _dempster_combination(bpa1, bpa2, ps):
    """Dempster组合规则"""
    combined = defaultdict(float)
    conflict = 0.0

    for s1, m1 in bpa1.items():
        for s2, m2 in bpa2.items():
            intersection = tuple(sorted(set(s1) & set(s2)))
            product = m1 * m2

            if intersection:
                combined[intersection] += product
            else:
                conflict += product

    # 归一化
    if conflict < 1.0:
        normalization = 1.0 - conflict
        for key in combined:
            combined[key] /= normalization

    return dict(combined)

def _calculate_pi_values(M1, M2, ps, elements):
    """计算序一致度 π(B_i^j)"""
    pi_values = {}

    for subset in ps:
        for perm in _get_all_permutations(subset):
            # 基本项: ℳ_1(B_i^j) + ℳ_2(B_i^j)
            pi = M1.get(perm, 0.0) + M2.get(perm, 0.0)

            # 超集项求和
            for superset in ps:
                if set(subset).issubset(set(superset)) and len(superset) > len(subset):
                    for super_perm in _get_all_permutations(superset):
                        tau = _kendall_tau_distance(super_perm, perm)
                        pi += tau * (M1.get(super_perm, 0.0) + M2.get(super_perm, 0.0))

            pi_values[perm] = pi

    return pi_values

def _calculate_weight_values(pi_values, ps, elements):
    """计算权重 w(B_i^j)"""
    w_values = {}

    for subset in ps:
        perms = _get_all_permutations(subset)
        total_pi = sum(pi_values.get(perm, 0.0) for perm in perms)

        for perm in perms:
            if total_pi > 0:
                w_values[perm] = pi_values.get(perm, 0.0) / total_pi
            else:
                # 平均分配权重
                w_values[perm] = 1.0 / len(perms) if perms else 0.0

    return w_values

def _distribute_belief(combined_bpa, w_values, ps, elements):
    """分配信念 ℳ(B_i^j) = m_ℳ(B_i) × w(B_i^j)"""
    result_pmf = {}

    for subset in ps:
        bpa_value = combined_bpa.get(subset, 0.0)
        for perm in _get_all_permutations(subset):
            result_pmf[perm] = bpa_value * w_values.get(perm, 0.0)

    return result_pmf

def _normalize_result(result_pmf):
    """归一化结果"""
    total = sum(result_pmf.values())
    if total > 0:
        result_pmf = {k: v / total for k, v in result_pmf.items()}

    # 过滤掉接近零的值
    return {k: v for k, v in result_pmf.items() if v > 1e-10}

def main_function(evidence, hypotheses_all, hypotheses = {'A', 'B', 'C','D'}):
    # 处理证据体格式
    # normalized_evidence = normalize_evidence(evidence)
    # OPT_evidence = batch_OPT(evidence, hypotheses)

    # 计算相似度矩阵
    S = compute_similarity_matrix(evidence)

    # 计算权重 R
    # print(S)
    # 计算综合置信度权重
    R_values = compute_weights_with_both_metrics(S, evidence)
    # print("Optimal R:", R_values)
# Crd； [0.23478793 0.09370717 0.23994348 0.21806852 0.21349289]
    # R_values = [0.2406,0.0899,0.2409,0.2251,0.2034]
    # 计算融合概率
    alpha = 0.1 # 乐观系数
    # fused_probabilities = fuse_evidence(evidence, R_values, hypotheses_all, alpha)
    fused_probabilities = weighted_consensus_rps(evidence,R_values,hypotheses_all)
    # fused_probabilities = absolute_weighted_sum_amplify_log(evidence,R_values,hypotheses_all)
    # M_w = defaultdict(float)
    # for subset in hypotheses_all:
    #     for i, ev in enumerate(evidence):
    #         M_w[subset] += R_values[i] * ev.get(subset, 0.0)
    # print("fused_probabilities:", convert_numpy_types(fused_probabilities))


    # rps_list_r = [copy.deepcopy(fused_probabilities) for _ in range(len(evidence))]

    # 使用连续左交和
    # ros_result = continuous_right_orthogonal_sum(rps_list_r)
    # los_result = continuous_right_orthogonal_sum(rps_list_r)
    ros_result = dict(fused_probabilities)
    N = len(evidence)
    for _ in range(N - 1):
        ros_result = zhou_combination(ros_result, fused_probabilities)
    # print("Optimal ros:", ros_result)
    # print("Optimal los:", los_result)

    fused_end = calculate_OPT(ros_result, hypotheses)
    # print("Fused end:", convert_numpy_types(fused_end))
    # pic_uniform = calculate_pic(fused_end)
    # print(f"均匀分布: PIC = {pic_uniform:.6f}")
    return fused_end

def extract_hypotheses(evidence_rps):
    """
    从RPS证据结构中抽取所有唯一的假设组合

    参数:
        evidence_rps: 包含多个RPS证据字典的列表

    返回:
        包含所有唯一假设的集合(set)，每个假设是一个元组
    """
    hypotheses = set()
    for evidence in evidence_rps:
        hypotheses.update(evidence.keys())
    return hypotheses

# 辅助函数定义
def get_ordered_permutations(n):
    """获取所有按顺序的排列组合"""
    all_combinations = []
    for r in range(1, n + 1):
        for comb in itertools.combinations(range(n), r):
            all_combinations.append(comb)
    return all_combinations

def calculate_weighted_pmf(weight_matrix, sorted_nmv):
    """计算加权概率质量函数 - 修复版本"""
    num_combinations, num_attributes = weight_matrix.shape
    num_classes = sorted_nmv.shape[0]

    weighted_pmf = np.zeros((num_combinations, num_attributes))

    for attr_idx in range(num_attributes):
        weights = weight_matrix[:, attr_idx]
        # 对于每个组合，计算加权概率
        for comb_idx, combination in enumerate(get_ordered_permutations(num_classes)):
            if len(combination) == 1:
                # 单元素组合
                class_idx = combination[0]
                weighted_pmf[comb_idx, attr_idx] = weights[comb_idx] * sorted_nmv[class_idx, attr_idx]
            else:
                # 多元素组合 - 简化处理
                weighted_pmf[comb_idx, attr_idx] = weights[comb_idx] * np.mean([sorted_nmv[i, attr_idx] for i in combination])

    return weighted_pmf

hypotheses = {'A', 'B', 'C'}

def evaluate_fusion(test_evidence, test_labels):
    """对生成的多个样本做融合并计算准确率"""
    correct = 0
    total = len(test_evidence)

    for evidence, true_lab in zip(test_evidence, test_labels):
        # print("true_lab",true_lab)
        hypotheses_all = extract_hypotheses(evidence)
        # print("evidence",evidence)
        fused_result = main_function(evidence, hypotheses_all, hypotheses)
        # print(fused_result)
        # 预测标签 = 概率最大的类
        pred = max(fused_result, key=fused_result.get)
        true_label_str = index_to_label[true_lab]

        if pred == true_label_str:
            correct += 1

    acc = correct / total
    return acc

def gen_rps_fun_for_sample(X_train, y_train, test_sample):
    """为单个测试样本生成RPS - 修复版本"""
    num_classes = len(np.unique(y_train))
    num_attributes = X_train.shape[1]

    # 计算每个类中每个属性的 mean value and standard deviation
    mean_std_by_class = []
    for class_label in np.unique(y_train):
        X_class = X_train[y_train == class_label]
        mean_std = [(np.mean(X_class[:, i]), np.std(X_class[:, i], ddof=1)) for i in range(X_class.shape[1])]
        mean_std_by_class.append(mean_std)

    mean_std_by_class = np.array(mean_std_by_class)

    # 为每个类和每个属性建立高斯分布函数
    gaussian_functions = np.empty((num_classes, num_attributes), dtype=object)
    for class_label in range(num_classes):
        for i in range(num_attributes):
            mean, std = mean_std_by_class[class_label, i]
            gaussian_functions[class_label, i] = norm(loc=mean, scale=std)

    # 计算该测试样本在每个类中每个属性的高斯分布结果
    gaussian_results = []
    for class_label in range(num_classes):
        class_results = []
        for i in range(num_attributes):
            pdf_value = gaussian_functions[class_label, i].pdf(test_sample[i])
            class_results.append(pdf_value)
        gaussian_results.append(class_results)

    gaussian_results = np.array(gaussian_results)
    column_sums = np.sum(gaussian_results, axis=0)
    normalized_results = gaussian_results / column_sums

    # 对归一化后的MV进行降序排序
    sorted_indices = np.argsort(-normalized_results, axis=0)
    sorted_nmv = np.take_along_axis(normalized_results, sorted_indices, axis=0)

    # 计算排序后的均值
    x_mean_ord = np.empty((num_classes, num_attributes))
    for attr_idx in range(num_attributes):
        for class_idx in range(num_classes):
            sorted_class_idx = sorted_indices[class_idx, attr_idx]
            x_mean_ord[class_idx, attr_idx] = mean_std_by_class[sorted_class_idx, attr_idx, 0]

    # 计算支持度
    supporting_degree = np.exp(-np.abs(test_sample - x_mean_ord))

    # 获取所有排列组合
    all_combinations = get_ordered_permutations(num_classes)
    num_combinations = len(all_combinations)

    # 初始化权重矩阵
    weight_matrix = np.zeros((num_combinations, num_attributes))

    # 对每个属性计算权重
    for attr_idx in range(num_attributes):
        s = supporting_degree[:, attr_idx]
        for comb_idx, combination in enumerate(all_combinations):
            q = len(combination)
            weight = 1.0
            for u in range(q):
                i_u = combination[u]
                numerator = s[i_u]
                denominator_sum = np.sum(s[list(combination[u:])])
                weight *= numerator / denominator_sum
            weight_matrix[comb_idx, attr_idx] = weight

    # 计算加权概率质量函数
    weighted_pmf = calculate_weighted_pmf(weight_matrix, sorted_nmv)

    # 构建RPS
    RPS_w = []
    for j in range(num_attributes):
        RPS_w_j = set()
        thetas = sorted_indices[:, j]
        weighted_pmf_j = weighted_pmf[:, j]

        for idx, combination in enumerate(all_combinations):
            A = thetas[list(combination)]
            M_A = weighted_pmf_j[idx]
            RPS_w_j.add((tuple(A), M_A))

        RPS_w.append(RPS_w_j)

    return RPS_w

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

def cross_validation_with_rps(n_splits=5, n_repeats=100):
    """100次五折交叉验证，按照Algorithm 1 RPSRCA流程"""

    # 加载Iris数据集
    iris = load_iris()
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # 初始化重复的五折交叉验证
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    all_accuracies = []  # 存储所有重复实验的准确率
    repeat_count = 1     # 重复计数器

    for train_index, test_index in rkf.split(X):
        print(f"正在处理第 {repeat_count} 次重复的第 {((repeat_count-1) % n_splits) + 1} 折...")

        # 划分训练测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 为当前折生成所有测试样本的RPS
        all_evidence = []
        all_labels = []

        for test_idx in range(len(X_test_scaled)):
            try:
                gen_rps = gen_rps_fun_for_sample(X_train_scaled, y_train, X_test_scaled[test_idx])
                convert_numpy_types(gen_rps)
                labeled_evidence = convert_to_labeled_rps(gen_rps)
                # print("labeled_evidence",labeled_evidence)
                all_evidence.append(labeled_evidence)
                all_labels.append(y_test[test_idx])
            except Exception as e:
                print(f"处理样本 {test_idx} 时出错: {e}")
                continue

        # 在当前折次中进行融合并计算准确率
        if len(all_evidence) > 0:
            acc = evaluate_fusion(all_evidence, all_labels)
            all_accuracies.append(acc)
            print(f"第 {repeat_count} 次重复的第 {((repeat_count-1) % n_splits) + 1} 折准确率: {acc:.4f}")
        else:
            print(f"第 {repeat_count} 次重复的第 {((repeat_count-1) % n_splits) + 1} 折无有效样本")
            all_accuracies.append(0)

        # 每完成一次完整的5折交叉验证，输出中间结果
        if repeat_count % n_splits == 0:
            current_repeat = repeat_count // n_splits
            current_accuracies = all_accuracies[-n_splits:]
            mean_acc = np.mean(current_accuracies)
            print(f"\n=== 第 {current_repeat} 次5折交叉验证完成 ===")
            print(f"本次平均准确率: {mean_acc:.4f}")
            print(f"各折准确率: {[f'{acc:.4f}' for acc in current_accuracies]}\n")

        repeat_count += 1

    # 计算最终统计结果
    total_repeats = n_repeats
    total_folds = n_splits * n_repeats

    # 计算每次5折交叉验证的平均值
    repeat_means = []
    for i in range(n_repeats):
        start_idx = i * n_splits
        end_idx = start_idx + n_splits
        repeat_mean = np.mean(all_accuracies[start_idx:end_idx])
        repeat_means.append(repeat_mean)

    # 输出最终结果
    print("=" * 50)
    print(f"100次五折交叉验证最终结果:")
    print(f"总折次数: {total_folds}")
    print(f"总体平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
    print(f"每次5折交叉验证的平均准确率: {np.mean(repeat_means):.4f} (±{np.std(repeat_means):.4f})")
    print(f"最高准确率: {np.max(all_accuracies):.4f}")
    print(f"最低准确率: {np.min(all_accuracies):.4f}")

    return {
        'all_accuracies': all_accuracies,
        'repeat_means': repeat_means,
        'overall_mean': np.mean(all_accuracies),
        'overall_std': np.std(all_accuracies),
        'repeat_mean': np.mean(repeat_means),
        'repeat_std': np.std(repeat_means)
    }

# 执行100次五折交叉验证
results = cross_validation_with_rps(n_splits=5, n_repeats=100)


# if __name__ == '__main__':
#     evidenceList = []
#     evidence5 = [
#         {  # 传感器1
#             ('A',): 0.31,
#             ('B',): 0.0,
#             ('C',): 0.29,
#             ('A', 'C',): 0.0,
#             ('C', 'A',): 0.0,
#             ('A', 'B', 'C'): 0.0167,
#             ('A', 'C', 'B'): 0.0167,
#             ('B', 'A', 'C'): 0.0167,
#             ('B', 'C', 'A'): 0.0167,
#             ('C', 'A', 'B'): 0.3167,
#             ('C', 'B', 'A'): 0.0167
#         },
#         {  # 传感器2
#             ('A',): 0.0,
#             ('B',): 0.8,
#             ('C',): 0.2,
#             ('A', 'C',): 0.0,
#             ('C', 'A',): 0.0,
#             ('A', 'B', 'C'): 0.0,
#             ('A', 'C', 'B'): 0.0,
#             ('B', 'A', 'C'): 0.0,
#             ('B', 'C', 'A'): 0.0,
#             ('C', 'A', 'B'): 0.0,
#             ('C', 'B', 'A'): 0.0
#         },
#         {  # 传感器3
#             ('A',): 0.27,
#             ('B',): 0.07,
#             ('C',): 0.21,
#             ('A', 'C',): 0.0,
#             ('C', 'A',): 0.0,
#             ('A', 'B', 'C'): 0.025,
#             ('A', 'C', 'B'): 0.025,
#             ('B', 'A', 'C'): 0.025,
#             ('B', 'C', 'A'): 0.025,
#             ('C', 'A', 'B'): 0.325,
#             ('C', 'B', 'A'): 0.025
#         },
#         {  # 传感器4
#             ('A',): 0.25,
#             ('B',): 0.05,
#             ('C',): 0.3,
#             ('A', 'C',): 0.09,
#             ('C', 'A',): 0.31,
#             ('A', 'B', 'C'): 0.0,
#             ('A', 'C', 'B'): 0.0,
#             ('B', 'A', 'C'): 0.0,
#             ('B', 'C', 'A'): 0.0,
#             ('C', 'A', 'B'): 0.0,
#             ('C', 'B', 'A'): 0.0
#         },
#         {  # 专家
#             ('A',): 0.25,
#             ('B',): 0.0,
#             ('C',): 0.2,
#             ('A', 'C'): 0.36,
#             ('C', 'A'): 0.0,
#             ('A', 'B', 'C'): 0.0233,
#             ('A', 'C', 'B'): 0.0733,
#             ('B', 'A', 'C'): 0.0233,
#             ('B', 'C', 'A'): 0.0233,
#             ('C', 'A', 'B'): 0.0233,
#             ('C', 'B', 'A'): 0.0233
#         }
#     ]
#     labeled_evidence1 = [{('B', 'C'): 0.03327384787501296, ('A',): 0.9019616778270755, ('A', 'C'): 0.39866767795888225,
#                           ('A', 'B'): 0.3718147939721801, ('B',): 0.08115425823940943, ('C',): 0.016884063933515225,
#                           ('A', 'B', 'C'): 0.1534570958354748},
#                          {('C', 'B', 'A'): 0.10182584737432683, ('B', 'A'): 0.16947693526879107,
#                           ('C', 'B'): 0.2455900301874588, ('C', 'A'): 0.2325030640067866, ('A',): 0.1918774432426149,
#                           ('B',): 0.35284268983452355, ('C',): 0.4552798669228617},
#                          {('A', 'B'): 0.41598591681815067, ('A', 'C'): 0.45577200539511675,
#                           ('A', 'B', 'C'): 0.17332676532019237, ('B',): 1.058743745570101e-08,
#                           ('C',): 9.43747425561326e-14, ('A',): 0.9999999894124683, ('B', 'C'): 3.5757071139477724e-09},
#                          {('B',): 2.7875087890433815e-08, ('A', 'C'): 0.4561745969379547,
#                           ('A', 'B', 'C'): 0.17796638594216732, ('A', 'B'): 0.40403893256141427,
#                           ('A',): 0.9999999716680565, ('B', 'C'): 1.008610274197881e-08,
#                           ('C',): 4.5685557258858824e-10}]
#     evidenceList.append(evidence5)
#     hypotheses = {'A', 'B', 'C', 'D'}
#     for evidence in evidenceList:
#         hypotheses_all = extract_hypotheses(evidence)
#         result = main_function(evidence, hypotheses_all, hypotheses)
#         print(result)
