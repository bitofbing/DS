import numpy as np
import math
from itertools import permutations, combinations
from typing import List, Dict, Tuple
from collections import defaultdict


class PMFFusion:
    def __init__(self, elements: List[str]):
        self.elements = elements
        self.n = len(elements)
        self.pes = self._generate_pes()
        self.ps = self._generate_ps()

    def _generate_pes(self) -> List[Tuple]:
        """生成幂排列集PES(Θ)"""
        pes = []
        for i in range(0, self.n + 1):
            if i == 0:
                pes.append(())
            else:
                for perm in permutations(self.elements, i):
                    pes.append(perm)
        return pes

    def _generate_ps(self) -> List[Tuple]:
        """生成幂集PS(Θ)"""
        ps = []
        for i in range(0, self.n + 1):
            if i == 0:
                ps.append(())
            else:
                for comb in combinations(self.elements, i):
                    ps.append(tuple(sorted(comb)))
        return ps

    def _get_all_permutations(self, subset: Tuple) -> List[Tuple]:
        """获取无序集合的所有排列"""
        if len(subset) == 0:
            return [()]
        return list(permutations(subset))

    def _kendall_tau_distance(self, perm1: Tuple, perm2: Tuple) -> float:
        """
        计算肯德尔秩距离 τ(B_k^t, B_i^j)
        根据公式：τ(B_k^t, B_i^j) = (S - D) / C_r^2
        """
        if len(perm1) != len(perm2):
            return 0.0

        common_elements = set(perm1) & set(perm2)
        r = len(common_elements)

        if r < 2:
            return 1.0 if r == 1 else 0.0

        S, D = 0, 0
        common_list = list(common_elements)

        # 计算相同和相反顺序的元素对数
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                elem_i, elem_j = common_list[i], common_list[j]

                # 获取元素位置
                pos1_i = perm1.index(elem_i)
                pos1_j = perm1.index(elem_j)
                pos2_i = perm2.index(elem_i)
                pos2_j = perm2.index(elem_j)

                # 判断顺序关系
                order1 = pos1_i < pos1_j
                order2 = pos2_i < pos2_j

                if order1 == order2:
                    S += 1
                else:
                    D += 1

        total_pairs = math.comb(r, 2)  # C_r^2
        return (S - D) / total_pairs if total_pairs > 0 else 0.0

    def pmf_to_bpa(self, pmf_dict: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        步骤(1): 还原PMF分布对应的BPA分布
        公式(2-18): m_ℳ(B_i) = Σ_{j=1}^{|B_i|!} ℳ(B_i^j)
        """
        bpa = defaultdict(float)
        for subset in self.ps:
            bpa_value = 0.0
            for perm in self._get_all_permutations(subset):
                bpa_value += pmf_dict.get(perm, 0.0)
            bpa[subset] = bpa_value
        return bpa

    def dempster_combination(self, bpa1: Dict[Tuple, float], bpa2: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        步骤(2): Dempster组合规则融合BPA
        公式(2-19): m_ℳ(B_i) = Σ_{B_i = B_j ∩ B_k} m_{ℳ_1}(B_j) m_{ℳ_2}(B_k)
        """
        combined_bpa = defaultdict(float)
        conflict = 0.0

        # 计算组合结果
        for s1, v1 in bpa1.items():
            for s2, v2 in bpa2.items():
                intersection = tuple(sorted(set(s1) & set(s2)))
                product = v1 * v2

                if intersection:  # 交集不为空
                    combined_bpa[intersection] += product
                else:  # 冲突处理
                    conflict += product

        # 归一化
        if conflict < 1.0:
            normalization = 1.0 - conflict
            for key in combined_bpa:
                combined_bpa[key] /= normalization

        return combined_bpa

    def zhou_combination(self, M1_dict: Dict[Tuple, float], M2_dict: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        完整的Zhou组合规则实现
        """
        # 步骤(1): 还原BPA分布
        bpa1 = self.pmf_to_bpa(M1_dict)
        bpa2 = self.pmf_to_bpa(M2_dict)

        # 步骤(2): Dempster组合
        combined_bpa = self.dempster_combination(bpa1, bpa2)

        # 步骤(3): 计算序一致度 π(B_i^j)
        pi_values = {}
        for subset in self.ps:
            for perm in self._get_all_permutations(subset):
                # 基本项: ℳ_1(B_i^j) + ℳ_2(B_i^j)
                pi = M1_dict.get(perm, 0.0) + M2_dict.get(perm, 0.0)

                # 超集项求和: Σ_{B_i ⊂ B_k} τ(B_k^t, B_i^j) * (ℳ_1(B_k^t) + ℳ_2(B_k^t))
                for superset in self.ps:
                    if set(subset).issubset(set(superset)) and len(superset) > len(subset):
                        for super_perm in self._get_all_permutations(superset):
                            tau = self._kendall_tau_distance(super_perm, perm)
                            pi += tau * (M1_dict.get(super_perm, 0.0) + M2_dict.get(super_perm, 0.0))

                pi_values[perm] = pi

        # 步骤(4): 计算权重 w(B_i^j)
        w_values = {}
        for subset in self.ps:
            perms = self._get_all_permutations(subset)
            total_pi = sum(pi_values.get(perm, 0.0) for perm in perms)

            for perm in perms:
                if total_pi > 0:
                    w_values[perm] = pi_values.get(perm, 0.0) / total_pi
                else:
                    # 平均分配权重
                    w_values[perm] = 1.0 / len(perms) if perms else 0.0

        # 步骤(5): 分配信念 ℳ(B_i^j) = m_ℳ(B_i) × w(B_i^j)
        result_pmf = {}
        for subset in self.ps:
            bpa_value = combined_bpa.get(subset, 0.0)
            for perm in self._get_all_permutations(subset):
                result_pmf[perm] = bpa_value * w_values.get(perm, 0.0)

        # 步骤(6): 归一化（移除空集）
        total = sum(result_pmf.values())
        if total > 0:
            for key in result_pmf:
                result_pmf[key] /= total

        return result_pmf

    # ========== 完整的融合算法（基于PMF不确定度和差异度） ==========
    def dict_to_vector(self, pmf_dict: Dict[Tuple, float]) -> np.ndarray:
        """PMF字典转换为向量"""
        vector = np.zeros(len(self.pes))
        for subset, value in pmf_dict.items():
            if subset in self.pes:
                idx = self.pes.index(subset)
                vector[idx] = value
        return vector

    def _P(self, n: int, i: int) -> int:
        """排列数 P(n,i) = n!/(n-i)!"""
        if i < 0 or i > n:
            return 0
        return math.factorial(n) // math.factorial(n - i)

    def _F(self, i: int) -> int:
        """F(i) = Σ_{k=0}^{i} P(i,k)"""
        total = 0
        for k in range(0, i + 1):
            total += self._P(i, k)
        return total

    def d_RPS(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """RPS距离"""
        distance = 0.0
        for idx in range(len(M1)):
            i = len(self.pes[idx])
            F_i = self._F(i)
            if F_i > 1:
                distance += (M1[idx] - M2[idx]) ** 2 / (F_i - 1)
        return math.sqrt(distance / 2)

    def D_RPS(self, M1: np.ndarray, M2: np.ndarray) -> float:
        """RPS散度"""
        term1, term2 = 0.0, 0.0
        for idx in range(len(M1)):
            if M1[idx] > 0 and M2[idx] > 0:
                term1 += M1[idx] * math.log(2 * M1[idx] / (M1[idx] + M2[idx]))
                term2 += M2[idx] * math.log(2 * M2[idx] / (M1[idx] + M2[idx]))
        return 0.5 * (term1 + term2)

    def H_RPS(self, M: np.ndarray) -> float:
        """RPS熵"""
        entropy = 0.0
        for idx in range(len(M)):
            if M[idx] > 0:
                i = len(self.pes[idx])
                F_i = self._F(i)
                if F_i > 1:
                    entropy += M[idx] * math.log(M[idx] / (F_i - 1))
        return -entropy

    def fusion(self, evidence_list: List[Dict[Tuple, float]],
               diff_method: str = 'distance') -> Dict[Tuple, float]:
        """
        完整的融合算法
        """
        N = len(evidence_list)

        # 计算相似度矩阵和权重
        SM = np.eye(N)
        for i in range(N):
            vec_i = self.dict_to_vector(evidence_list[i])
            for j in range(i + 1, N):
                vec_j = self.dict_to_vector(evidence_list[j])
                if diff_method == 'distance':
                    dif = self.d_RPS(vec_i, vec_j)
                else:
                    dif = self.D_RPS(vec_i, vec_j)
                SM[i, j] = SM[j, i] = 1 - dif

        # 计算权重
        # print("SM", SM)
        support = np.sum(SM, axis=1) - np.diag(SM)
        credibility = support / np.sum(support)
        # print("credibility",credibility)
        entropies = np.array([self.H_RPS(self.dict_to_vector(e)) for e in evidence_list])
        norm_entropies = entropies / np.sum(entropies)
        avg_cred = np.mean(credibility)
        delta_cred = credibility - avg_cred
        cred_H = credibility * np.power(norm_entropies, -delta_cred)
        weights = cred_H / np.sum(cred_H)

        # 加权平均生成新证据
        M_w = defaultdict(float)
        for subset in self.pes:
            for i, evidence in enumerate(evidence_list):
                M_w[subset] += weights[i] * evidence.get(subset, 0.0)

        # 多次Zhou组合
        M_f = dict(M_w)
        for _ in range(N - 1):
            M_f = self.zhou_combination(M_f, M_w)

        return M_f


def solve_meowa_weights(n, orness):
    """
    求解MEOWA权重向量 (公式2-27和2-28)

    参数:
    n: 权重向量的维度
    orness: Orness测度值 [0, 1]

    返回:
    w: MEOWA权重向量
    """
    # 特殊情况处理
    if n == 1:
        return np.array([1.0])

    if orness == 0.5:
        # 当Orness=0.5时，权重均匀分布
        return np.ones(n) / n

    # 构造方程并求解h (公式2-27)
    def equation(h):
        result = 0
        for i in range(1, n + 1):
            term = ((n - i) / (n - 1) - orness) * (h ** (n - i))
            result += term
        return result

    # 寻找方程的正解h*
    # 使用二分法求解
    if orness > 0.5:
        # Orness > 0.5，h* > 1
        h_low, h_high = 1.0, 100.0
    else:
        # Orness < 0.5，0 < h* < 1
        h_low, h_high = 0.001, 0.999

    tolerance = 1e-10
    max_iter = 1000

    for _ in range(max_iter):
        h_mid = (h_low + h_high) / 2
        f_mid = equation(h_mid)

        if abs(f_mid) < tolerance:
            h_star = h_mid
            break

        if equation(h_low) * f_mid < 0:
            h_high = h_mid
        else:
            h_low = h_mid
    else:
        h_star = (h_low + h_high) / 2

    # 计算权重向量 (公式2-28)
    denominator = 0
    weights = np.zeros(n)

    for i in range(1, n + 1):
        weights[i - 1] = h_star ** (n - i)
        denominator += weights[i - 1]

    weights = weights / denominator

    return weights


def meowa_probability_conversion(rps_data, orness=0.5):
    """
    MEOWA概率转换算法 (公式2-29)

    参数:
    rps_data: RPS数据，格式为 {排列元组: 质量值}
    orness: Orness测度值，默认为0.5

    返回:
    probability_dist: 概率分布字典 {元素: 概率}
    """
    # 提取所有元素
    all_elements = set()
    for permutation in rps_data.keys():
        all_elements.update(permutation)
    all_elements = sorted(all_elements)

    # 初始化概率分布
    probability_dist = {element: 0.0 for element in all_elements}

    # 遍历每个排列及其质量值
    for permutation, mass in rps_data.items():
        if mass == 0:
            continue

        n = len(permutation)  # 排列长度

        # 获取MEOWA权重向量
        weights = solve_meowa_weights(n, orness)

        # 对排列中的每个元素分配质量值
        for idx, element in enumerate(permutation):
            # r(θ_l) = idx + 1 (顺序索引，从1开始)
            rank = idx + 1
            weight_value = weights[rank - 1]  # 权重向量的第r(θ_l)个值

            # 累加概率值
            probability_dist[element] += weight_value * mass

    return probability_dist


def calculate_orness(weights):
    """
    计算权重向量的Orness测度 (公式2-25)

    参数:
    weights: 权重向量

    返回:
    orness: Orness测度值
    """
    n = len(weights)
    if n <= 1:
        return 0.5

    orness = 0
    for i in range(n):
        orness += (n - i - 1) * weights[i]

    orness /= (n - 1)
    return orness


def validate_probability_distribution(prob_dist):
    """
    验证概率分布是否有效
    """
    total = sum(prob_dist.values())
    is_valid = abs(total - 1.0) < 1e-10
    all_non_negative = all(p >= 0 for p in prob_dist.values())

    return is_valid and all_non_negative, total


# 使用示例
# 测试验证
def test_fusion():
    elements = ['A', 'B', 'C']
    fusion_model = PMFFusion(elements)

    # 输入证据（符合您要求的格式）
    evidence5 = [
        {  # 传感器1
            ('A',): 0.31,
            ('B',): 0.0,
            ('C',): 0.29,
            ('A', 'C',): 0.0,
            ('C', 'A',): 0.0,
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
            ('A', 'C',): 0.0,
            ('C', 'A',): 0.0,
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
            ('A', 'C',): 0.0,
            ('C', 'A',): 0.0,
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
            ('A', 'C',): 0.09,
            ('C', 'A',): 0.31,
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
    labeled_evidence28 = [{('B', 'A'): 0.22636190904787196, ('C', 'A'): 0.14237847434635276, ('B',): 0.5934774840152125,
                           ('B', 'C', 'A'): 0.11290247949701185, ('A',): 0.004080986332863588,
                           ('B', 'C'): 0.28484933439635246, ('C',): 0.4024415296519239},
                          {('B',): 0.5166425995769549, ('B', 'A'): 0.24453766094418075, ('C',): 0.4088987037022492,
                           ('A',): 0.07445869672079587, ('B', 'C'): 0.28942404770260866,
                           ('B', 'C', 'A'): 0.13677368794054443, ('C', 'A'): 0.17924689225930507},
                          {('A',): 7.714289473858699e-75, ('B', 'C', 'A'): 0.1314998780966697,
                           ('C', 'A'): 0.1403779411136986, ('B', 'A'): 0.27151763057275713,
                           ('B', 'C'): 0.2708512969783288, ('C',): 0.34759316743316016, ('B',): 0.6524068325668397},
                          {('B', 'A'): 0.03858602924098985, ('C', 'A'): 0.38660697619251155, ('C',): 0.9047300962694884,
                           ('B',): 0.09526990373051156, ('A',): 1.500708200726857e-47,
                           ('C', 'B', 'A'): 0.1424578573105348, ('C', 'B'): 0.2898060963036712}]
    labeled_evidence24 = [
        {('A', 'C'): 0.4084071081944036, ('C',): 0.008923636348204403, ('A', 'B', 'C'): 0.15954509459327368,
         ('A', 'B'): 0.3754309390974994, ('B',): 0.07012895413321292, ('A',): 0.9209474095185827,
         ('B', 'C'): 0.02758968297424276},
        {('C',): 0.28044185881331424, ('C', 'A'): 0.11118917967701636, ('B', 'C', 'A'): 0.13747597540056683,
         ('B', 'C'): 0.3079631828890033, ('B', 'A'): 0.29826871508354796, ('A',): 0.019391026245051488,
         ('B',): 0.7001671149416343},
        {('A',): 3.697790505596163e-62, ('B', 'A'): 0.37405491603976626, ('B', 'C'): 0.3122947472612872,
         ('B',): 0.8987850345657982, ('C', 'A'): 0.037905601057216146, ('B', 'C', 'A'): 0.13851662290885777,
         ('C',): 0.10121496543420191},
        {('C', 'A'): 0.2793928794110556, ('B', 'A'): 0.12851152507864289, ('C',): 0.6827017222726427,
         ('C', 'B', 'A'): 0.124580144586322, ('A',): 1.1778437827453888e-41, ('C', 'B'): 0.25698792294219497,
         ('B',): 0.31729827772735736}]
    labeled_evidence25 = [
        {('B', 'A'): 0.25409293676063066, ('B', 'A', 'C'): 0.0956611613703336, ('C',): 0.17632022271576558,
         ('B',): 0.6443519619416713, ('B', 'C'): 0.28641800683053287, ('A',): 0.17932781534256298,
         ('A', 'C'): 0.1047939011602609},
        {('C',): 0.28044185881331424, ('C', 'A'): 0.11118917967701636, ('B', 'C', 'A'): 0.13747597540056683,
         ('B', 'C'): 0.3079631828890033, ('B', 'A'): 0.29826871508354796, ('A',): 0.019391026245051488,
         ('B',): 0.7001671149416343},
        {('C', 'B', 'A'): 0.13022071218115394, ('C',): 0.6206358201355873, ('C', 'B'): 0.2575874622011634,
         ('A',): 5.406631909903558e-84, ('C', 'A'): 0.26087162938359987, ('B', 'A'): 0.15788317672226365,
         ('B',): 0.37936417986441284},
        {('C', 'B'): 0.35046275352317646, ('A',): 1.567304802558493e-60, ('C', 'B', 'A'): 0.17685431287976985,
         ('B',): 0.0036044298648960586, ('B', 'A'): 0.0014598591025912187, ('C', 'A'): 0.45288106591005217,
         ('C',): 0.996395570135104}]

    result = fusion_model.fusion(labeled_evidence25, diff_method='distance')
    prob_dist = meowa_probability_conversion(result, 0.7)
    print("融合结果:")
    print(prob_dist)
    # for subset, value in sorted(result.items(), key=lambda x: (-len(x[0]), x[0])):
    #     if value > 0.001:
    #         print(f"{subset}: {value:.4f}")
    return prob_dist

if __name__ == "__main__":
    test_fusion()