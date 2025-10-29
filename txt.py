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
    print("Optimal R:", R_values)
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
    pic_uniform = calculate_pic(fused_end)
    print(f"均匀分布: PIC = {pic_uniform:.6f}")
    return fused_end

