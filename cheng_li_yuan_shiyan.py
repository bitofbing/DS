from collections.abc import Iterable

import numpy as np
import math
from itertools import permutations, combinations
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import pickle
import os
from ucimlrepo import fetch_ucirepo
import pandas as pd


# 保留您原有的所有类和函数定义...
# [这里包含您提供的所有代码，从PMFFusion类到test_fusion函数]
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
        support = np.sum(SM, axis=1) - np.diag(SM)
        credibility = support / np.sum(support)
        entropies = np.array([self.H_RPS(self.dict_to_vector(e)) for e in evidence_list])
        norm_entropies = entropies / np.sum(entropies)
        avg_cred = np.mean(credibility)
        delta_cred = credibility - avg_cred
        cred_H = credibility * np.power(norm_entropies, -delta_cred)
        weights = cred_H / np.sum(cred_H)
        # print("weights",weights)

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

# ==================== 新增的交叉验证相关函数 ====================

def gen_rps_fun_for_sample(X_train, y_train, test_sample):
    """为单个测试样本生成RPS - 基于高斯分布的方法"""
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
    weighted_pmf = calculate_weighted_pmf(weight_matrix, sorted_nmv, num_classes)

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


def get_ordered_permutations(n):
    """获取所有按顺序的排列组合"""
    all_combinations = []
    for r in range(1, n + 1):
        for comb in combinations(range(n), r):
            all_combinations.append(comb)
    return all_combinations


def calculate_weighted_pmf(weight_matrix, sorted_nmv, num_classes):
    """计算加权概率质量函数"""
    num_combinations, num_attributes = weight_matrix.shape
    weighted_pmf = np.zeros((num_combinations, num_attributes))

    all_combinations = get_ordered_permutations(num_classes)

    for attr_idx in range(num_attributes):
        weights = weight_matrix[:, attr_idx]
        for comb_idx, combination in enumerate(all_combinations):
            if len(combination) == 1:
                # 单元素组合
                class_idx = combination[0]
                weighted_pmf[comb_idx, attr_idx] = weights[comb_idx] * sorted_nmv[class_idx, attr_idx]
            else:
                # 多元素组合 - 平均处理
                weighted_pmf[comb_idx, attr_idx] = weights[comb_idx] * np.mean(
                    [sorted_nmv[i, attr_idx] for i in combination])

    return weighted_pmf


# index_to_label = {0: 'A', 1: 'B', 2: 'C'}
index_to_label= {
0: "A",
1: "B",
2: "C",
3: "D",
4: "E",
5: "F",
6: "G",
7: "H",
8: "I",
9: "J"
}
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

def classify_sample(test_rps, elements, orness=0.7):
    """对单个测试样本进行分类"""
    # 使用PMF融合模型进行融合
    fusion_model = PMFFusion(elements)

    try:
        # 执行融合
        fused_result = fusion_model.fusion(test_rps, diff_method='distance')

        # MEOWA概率转换
        prob_dist = meowa_probability_conversion(fused_result, orness)

        # 预测类别
        if prob_dist:
            predicted_class = max(prob_dist.items(), key=lambda x: x[1])[0]
            return predicted_class, prob_dist
        else:
            # 默认预测第一个类别
            return elements[0] if elements else 'A', {elements[0] if elements else 'A': 1.0}
    except Exception as e:
        print(f"分类过程中出错: {e}")
        # 返回默认预测
        return elements[0] if elements else 'A', {elements[0] if elements else 'A': 1.0}

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

def cross_validation_with_rps(n_splits=5, n_repeats=100):
    """100次五折交叉验证主函数"""

    # 加载Iris数据集
    iris = load_iris()
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # 定义类别标签
    elements = ['A', 'B', 'C']  # 对应 setosa, versicolor, virginica
    class_labels = ['setosa', 'versicolor', 'virginica']

    # 初始化重复的五折交叉验证
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    all_accuracies = []
    repeat_count = 1

    for train_index, test_index in rkf.split(X):
        print(f"正在处理第 {repeat_count} 次重复的第 {((repeat_count - 1) % n_splits) + 1} 折...")

        # 划分训练测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 为当前折生成所有测试样本的RPS并进行分类
        correct_predictions = 0
        total_predictions = 0

        for test_idx in range(len(X_test_scaled)):
            try:
                # 1. 生成测试样本的RPS
                gen_rps = gen_rps_fun_for_sample(X_train_scaled, y_train, X_test_scaled[test_idx])

                # 2. 将RPS转换为标准格式
                convert_numpy_types(gen_rps)
                labeled_evidence = convert_to_labeled_rps(gen_rps)
                # print("labeled_evidence",labeled_evidence)
                # 4. 执行分类
                predicted_class, prob_dist = classify_sample(labeled_evidence, elements, orness=0.7)

                # 5. 映射预测结果到真实标签
                # 这里假设A->0, B->1, C->2
                class_mapping = {'A': 0, 'B': 1, 'C': 2}
                predicted_label = class_mapping.get(predicted_class, 0)
                # print("predicted_label", predicted_label)
                true_label = y_test[test_idx]
                # print("true_label", true_label)
                # 6. 检查分类是否正确
                if predicted_label == true_label:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                print(f"处理样本 {test_idx} 时出错: {e}")
                continue

        # 计算当前折的准确率
        if total_predictions > 0:
            acc = correct_predictions / total_predictions
            all_accuracies.append(acc)
            print(f"第 {repeat_count} 次重复的第 {((repeat_count - 1) % n_splits) + 1} 折准确率: {acc:.4f}")
        else:
            print(f"第 {repeat_count} 次重复的第 {((repeat_count - 1) % n_splits) + 1} 折无有效样本")
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
    return calculate_final_statistics(all_accuracies, n_splits, n_repeats)


# def calculate_final_statistics(all_accuracies, n_splits, n_repeats):
#     """计算最终统计结果"""
#     # 计算每次5折交叉验证的平均值
#     repeat_means = []
#     for i in range(n_repeats):
#         start_idx = i * n_splits
#         end_idx = start_idx + n_splits
#         repeat_mean = np.mean(all_accuracies[start_idx:end_idx])
#         repeat_means.append(repeat_mean)
#
#     # 输出最终结果
#     print("=" * 60)
#     print(f"{n_repeats}次{n_splits}折交叉验证最终结果:")
#     print(f"总折次数: {len(all_accuracies)}")
#     print(f"总体平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
#     print(f"每次{n_splits}折交叉验证的平均准确率: {np.mean(repeat_means):.4f} (±{np.std(repeat_means):.4f})")
#     print(f"最高准确率: {np.max(all_accuracies):.4f}")
#     print(f"最低准确率: {np.min(all_accuracies):.4f}")
#
#     return {
#         'all_accuracies': all_accuracies,
#         'repeat_means': repeat_means,
#         'overall_mean': np.mean(all_accuracies),
#         'overall_std': np.std(all_accuracies),
#         'repeat_mean': np.mean(repeat_means),
#         'repeat_std': np.std(repeat_means)
#     }

def load_sonar_dataset():
    """加载Sonar数据集"""
    sonar = fetch_ucirepo(id=151)
    X = sonar.data.features
    y = sonar.data.targets

    # 将目标变量转换为数值：Rock=0, Mine=1
    y = (y == 'M').astype(int).values.ravel()

    return {'data': X.values, 'target': y}

def generalized_cross_validation_with_rps(
        dataset_name: str = 'iris',
        n_splits: int = 5,
        n_repeats: int = 100,
        results_dir: str = 'cv_results\cheng_li_yuan_shiyan',
        save_final: bool = True
) -> Dict[str, Any]:
    """
    通用的五折交叉验证函数，支持多种数据集

    参数:
        dataset_name: 数据集名称 ('iris', 'wine', 'breast_cancer', 'digits', 'heart')
        n_splits: 折数 (默认为5)
        n_repeats: 重复次数 (默认为100)
        results_dir: 结果保存目录
        save_final: 是否保存最终结果

    返回:
        包含完整结果的字典
    """
    # 1. 加载数据集
    global elements
    global class_mapping
    data_loader = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits,
        'heart': load_heart_dataset,
        'sonar': load_sonar_dataset  # 新增Sonar数据集
    }

    if dataset_name not in data_loader:
        raise ValueError(f"不支持的数据集: {dataset_name}. 可选: {list(data_loader.keys())}")

    dataset = data_loader[dataset_name]()
    X = dataset['data']
    y = dataset['target']

    # 2. 根据数据集设置类别标签
    # 2. 根据数据集设置类别标签和映射关系
    if dataset_name == 'iris':
        # 3类: A, B, C
        class_labels = ['setosa', 'versicolor', 'virginica']
        elements = ['A', 'B', 'C']
        class_mapping = {'A': 0, 'B': 1, 'C': 2}
        original_to_letter = {0: 'A', 1: 'B', 2: 'C'}
    elif dataset_name == 'wine':
        # 3类: A, B, C
        class_labels = ['Class_1', 'Class_2', 'Class_3']
        elements = ['A', 'B', 'C']
        class_mapping = {'A': 0, 'B': 1, 'C': 2}
        original_to_letter = {0: 'A', 1: 'B', 2: 'C'}
    elif dataset_name == 'breast_cancer':
        # 2类: A, B
        class_labels = ['Benign', 'Malignant']
        elements = ['A', 'B']
        class_mapping = {'A': 0, 'B': 1}
        original_to_letter = {0: 'A', 1: 'B'}
    elif dataset_name == 'digits':
        # 10类: A-J
        class_labels = [str(i) for i in range(10)]
        elements = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        class_mapping = {letter: i for i, letter in enumerate(elements)}
        original_to_letter = {i: letter for i, letter in enumerate(elements)}
    elif dataset_name == 'heart':
        # 2类: A, B
        class_labels = ['No_Disease', 'Disease']
        elements = ['A', 'B']
        class_mapping = {'A': 0, 'B': 1}
        original_to_letter = {0: 'A', 1: 'B'}
    elif dataset_name == 'sonar':
        # 2类: A, B
        class_labels = ['Rock', 'Mine']
        elements = ['A', 'B']
        class_mapping = {'A': 0, 'B': 1}

    # 3. 创建结果目录
    os.makedirs(results_dir, exist_ok=True)

    # 4. 初始化交叉验证
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    file_results = {
        'dataset': dataset_name,
        'mean_accuracy': 0.0,  # 存储每次重复的统计摘要
        'std_accuracy':0.0
    }
    # 5. 存储所有结果
    all_results = {
        'dataset': dataset_name,
        'n_splits': n_splits,
        'n_repeats': n_repeats,
        'class_labels': class_labels,
        'class_mapping': class_mapping,
        'fold_results': [],  # 存储每一折的详细结果
        'repeat_summaries': [],  # 存储每次重复的统计摘要
        'all_accuracies': []  # 存储所有准确率
    }

    repeat_count = 1
    current_repeat_results = []

    for train_index, test_index in rkf.split(X):
        fold_result = {
            'repeat': (repeat_count - 1) // n_splits + 1,
            'fold': ((repeat_count - 1) % n_splits) + 1,
            'train_indices': train_index.tolist(),
            'test_indices': test_index.tolist(),
            'sample_results': [],
            'accuracy': 0.0
        }

        # 划分训练测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        correct_predictions = 0
        total_predictions = 0

        for test_idx in range(len(X_test_scaled)):
            # sample_result = {
            #     'true_label': int(y_test[test_idx]),
            #     'predicted_label': None,
            #     'prob_dist': None,
            #     'evidence': None,
            #     'correct': False
            # }

            try:
                # 1. 生成测试样本的RPS
                gen_rps = gen_rps_fun_for_sample(X_train_scaled, y_train, X_test_scaled[test_idx])

                # 2. 将RPS转换为标准格式
                convert_numpy_types(gen_rps)
                labeled_evidence = convert_to_labeled_rps(gen_rps)

                # 3. 执行分类
                predicted_class, prob_dist = classify_sample(labeled_evidence, elements)

                # 4. 映射预测结果
                predicted_label = class_mapping.get(predicted_class, 0)
                true_label = y_test[test_idx]

                # 5. 记录结果
                # sample_result.update({
                #     'predicted_label': int(predicted_label),
                #     'prob_dist': prob_dist,
                #     'evidence': labeled_evidence,
                #     'correct': predicted_label == true_label
                # })

                if predicted_label == true_label:
                    correct_predictions += 1
                total_predictions += 1

            except Exception as e:
                print(f"处理样本 {test_idx} 时出错: {e}")
                # sample_result['error'] = str(e)

            # fold_result['sample_results'].append(sample_result)

        # 计算当前折的准确率
        if total_predictions > 0:
            acc = correct_predictions / total_predictions
            fold_result['accuracy'] = acc
            all_results['all_accuracies'].append(acc)
        else:
            fold_result['accuracy'] = 0.0
            all_results['all_accuracies'].append(0.0)

        current_repeat_results.append(fold_result)
        all_results['fold_results'].append(fold_result)

        print(f"Repeat {(repeat_count - 1) // n_splits + 1}, Fold {((repeat_count - 1) % n_splits) + 1}: "
              f"Accuracy = {fold_result['accuracy']:.4f}")

        # 每完成一次完整的5折交叉验证，生成统计摘要但不保存文件
        if repeat_count % n_splits == 0:
            current_repeat = repeat_count // n_splits
            current_accuracies = [r['accuracy'] for r in current_repeat_results[-n_splits:]]
            mean_acc = np.mean(current_accuracies)

            repeat_summary = {
                'repeat': current_repeat,
                # 'fold_accuracies': current_accuracies,
                'mean_accuracy': mean_acc,
                # 'std_accuracy': np.std(current_accuracies),
                # 'min_accuracy': min(current_accuracies),
                # 'max_accuracy': max(current_accuracies)
            }

            # file_results['repeat_summaries'].append(repeat_summary)

            print(f"\n=== Repeat {current_repeat} 完成 ===")
            print(f"平均准确率: {mean_acc:.4f} ± {np.std(current_accuracies):.4f}")
            print(f"各折准确率: {[f'{acc:.4f}' for acc in current_accuracies]}")
            print(f"最小准确率: {min(current_accuracies):.4f}")
            print(f"最大准确率: {max(current_accuracies):.4f}\n")

            # 重置当前重复结果（仅清空临时列表）
            current_repeat_results = []

        repeat_count += 1

    # 计算最终统计结果
    final_stats = calculate_final_statistics(all_results['all_accuracies'], n_splits, n_repeats)
    all_results.update(final_stats)
    file_results['mean_accuracy'] = final_stats['mean_accuracy']
    file_results['std_accuracy'] = final_stats['std_accuracy']

    # 保存完整结果
    if save_final:
        final_path = os.path.join(results_dir, f"{dataset_name}_final_results.pkl")
        with open(final_path, 'wb') as f:
            pickle.dump(file_results, f)
        print(f"\n完整结果已保存到: {final_path}")

    return all_results


def load_heart_dataset():
    """
    加载Heart疾病数据集
    使用UCI机器学习仓库的API获取数据
    """
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # 处理目标变量：将>0的值视为有疾病(1)，0为无疾病(0)
    y = (y > 0).astype(int).values.ravel()

    # 处理缺失值（如果有）
    if isinstance(X, pd.DataFrame):
        X = X.fillna(X.mean()).values

    return {'data': X, 'target': y}


def calculate_final_statistics(all_accuracies: List[float], n_splits: int, n_repeats: int) -> Dict[str, Any]:
    """计算最终的统计结果"""
    accuracies = np.array(all_accuracies)
    repeat_accuracies = accuracies.reshape(n_repeats, n_splits).mean(axis=1)

    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_repeat_accuracy': np.mean(repeat_accuracies),
        'std_repeat_accuracy': np.std(repeat_accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'median_accuracy': np.median(accuracies),
        'repeat_accuracies': repeat_accuracies.tolist(),
        'all_accuracies': all_accuracies
    }
# ==================== 主程序 ====================

if __name__ == "__main__":
    # # 首先运行原始测试
    # print("=== 原始测试示例 ===")
    # test_result = test_fusion()
    # print("测试融合结果:")
    # for element, prob in sorted(test_result.items()):
    #     print(f"  {element}: {prob:.6f}")
    # print()

    # 执行100次五折交叉验证
    print("=== 100次五折交叉验证 ===")
    # results = generalized_cross_validation_with_rps('iris')
    # results = cross_validation_with_rps()
    # 保存结果（可选）
    # breast_cancer数据集最终结果:
    # 平均准确率: 0.9343 ± 0.0241
    # heart数据集最终结果:
    # 平均准确率: 0.8234 ± 0.0442
    databases = ['sonar']
    for database in databases:
        iris_results = generalized_cross_validation_with_rps(database)
        print("\nsonar数据集最终结果:")
        print(f"平均准确率: {iris_results['mean_accuracy']:.4f} ± {iris_results['std_accuracy']:.4f}")