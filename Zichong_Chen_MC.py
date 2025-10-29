from collections.abc import Iterable

from itertools import permutations, combinations
from scipy.stats import norm
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from typing import Dict, List, Tuple, Any
import pickle
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

# 定义事件空间为ABC
gamma = ['A', 'B', 'C']
epsilon = 0.00001
# 按照图片中的数据创建RPS列表
example_data = [
    {  # RPS₁
        ('A',): 0.25,
        ('A', 'B'): 0.25,
        ('B', 'A'): 0.25,
        ('A', 'B', 'C'): 0.25
    },
    {  # RPS₂
        ('A',): 0.12,
        ('A', 'B'): 0.32,
        ('B', 'A'): 0.20,
        ('A', 'B', 'C'): 0.36
    },
    {  # RPS₃
        ('A',): 0.15,
        ('A', 'B'): 0.31,
        ('B', 'A'): 0.21,
        ('A', 'B', 'C'): 0.33
    },
    {  # RPS₄
        ('A',): 0.20,
        ('A', 'B'): 0.27,
        ('B', 'A'): 0.13,
        ('A', 'B', 'C'): 0.40
    }
]

# 完整的LOS-SRP算法实现
import numpy as np
from collections import defaultdict
import math


def right_intersection(A, B):
    return tuple(item for item in B if item in A)


def left_intersection(A, B):
    return tuple(item for item in A if item in B)


def calculate_KR(M1, M2):
    K_R = 0
    for B, w1 in M1:
        for C, w2 in M2:
            if right_intersection(B, C) == ():
                K_R += w1 * w2
    return K_R


def calculate_KL(M1, M2):
    K_L = 0
    for B, w1 in M1:
        for C, w2 in M2:
            if left_intersection(B, C) == ():
                K_L += w1 * w2
    return K_L


def ROS(M1, M2):
    K_R = calculate_KR(M1, M2)
    result = defaultdict(float)

    if K_R != 1:
        all_keys = set(A for A, _ in M1) | set(C for C, _ in M2)
        for A in all_keys:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if right_intersection(B, C) == A:
                        weight_sum += w1 * w2
            result[A] = (1 / (1 - K_R)) * weight_sum

    return dict(result)


def LOS(M1, M2):
    K_L = calculate_KL(M1, M2)
    result = defaultdict(float)

    if K_L != 1:
        all_keys = set(A for A, _ in M1) | set(C for C, _ in M2)
        for A in all_keys:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if left_intersection(B, C) == A:
                        weight_sum += w1 * w2
            result[A] = (1 / (1 - K_L)) * weight_sum

    return dict(result)


def dict_to_pmf(rps_dict):
    return [(k, v) for k, v in rps_dict.items()]


def continuous_left_orthogonal_sum(PMFs):
    result = dict_to_pmf(PMFs[0])
    for i in range(1, len(PMFs)):
        result = LOS(result, dict_to_pmf(PMFs[i])).items()
    return dict(result)


def get_all_permutation_events(gamma):
    from itertools import chain, combinations
    all_subsets = list(chain.from_iterable(
        combinations(gamma, r) for r in range(1, len(gamma) + 1)
    ))
    return [tuple(subset) for subset in all_subsets]


def calculate_beta_from_RPS(RPS_list):
    """
    根据图片计算beta值：beta = 最大排列事件的基数
    图片中明确说明：β = max(|A_ij|)
    """
    all_events = set()
    for RPS in RPS_list:
        all_events.update(RPS.keys())

    if not all_events:
        return 1

    max_cardinality = max(len(event) for event in all_events)
    return max_cardinality


def RP_divergence(RPS1, RPS2, beta=None):
    """
    根据图片精确修正的RP散度计算 - 公式(14)

    图片中的公式：D_RP(RPS1||RPS2) = 1/(β-1) * log( Σ (M1^β / M2^(β-1)) )
    但图片示例中显示为 M2^(-2)，这是因为当β=3时，(β-1)=2，但写成了负号（可能是排版错误）
    """
    all_events = set(RPS1.keys()) | set(RPS2.keys())

    # 如果未指定beta，使用最大事件基数
    if beta is None:
        beta = calculate_beta_from_RPS([RPS1, RPS2])

    # 特殊情况：当beta=1时，RP散度退化为KL散度
    if beta == 0:
        kl_divergence = 0
        for event in all_events:
            M1 = RPS1.get(event, 0.0)
            M2 = RPS2.get(event, 0.0)

            if M1 > 0 and M2 > 0:
                kl_divergence += M1 * math.log(M1 / M2)
            # elif M1 > 0 and M2 == 0:
            #     return float('inf')

        return kl_divergence

    else:
        summation = 0
        valid_events = 0

        for event in all_events:
            M1 = RPS1.get(event, epsilon)
            M2 = RPS2.get(event, epsilon)
            if M1 == 0 or M1 < epsilon:
                M1 = epsilon
            if M2 == 0 or M2 < epsilon:
                M2 = epsilon
            # 关键修正：按照图片中的数学定义，应该是 M2^(β-1)
            # 但图片示例中显示为负指数，这不符合数学定义，我们采用正确的数学定义
            # if M2 > 0 and M1 > 0:
            # 正确的数学公式：M1^β / M2^(β-1)
            # print("M1", M1)
            # print("M2", M2)
            # print("beta", beta)
            term = (M1 ** beta) / (M2 ** (beta - 1))
            summation += term
            valid_events += 1
            # elif M1 > 0 and M2 == 0:
            #     return float('inf')

        # if valid_events == 0:
        #     return float('inf')
        #
        # if summation <= 0:
        #     return float('inf')

        return (1 / (beta - 1)) * math.log2(summation)


def SRP_divergence(RPS1, RPS2, beta=None):
    """
    对称RP散度计算 - 公式(15)
    D_SRP(RPS1||RPS2) = [D_RP(RPS1||RPS2) + D_RP(RPS2||RPS1)] / 2
    """
    drp_12 = RP_divergence(RPS1, RPS2, beta)
    drp_21 = RP_divergence(RPS2, RPS1, beta)

    # if math.isinf(drp_12) or math.isinf(drp_21):
    #     return float('inf')

    return (drp_12 + drp_21) / 2


def build_SRP_matrix(RPS_list, gamma,beta):
    """
    SRP矩阵构建 - 公式(16)
    构建n×n的对称散度矩阵
    """
    n = len(RPS_list)
    SRPM = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                SRPM[i, j] = 0.0  # 自散度为0
            else:
                divergence = SRP_divergence(RPS_list[i], RPS_list[j],beta)
                SRPM[i, j] = max(0.0, divergence)

    return SRPM

def calculate_PIQ(RPS, gamma):
    summation = 0
    for event, weight in RPS.items():
        if weight > 0:
            summation += weight ** len(event)
    return math.sqrt(summation)


def LOS_SRP_algorithm(RPS_list, gamma):
    n = len(RPS_list)

    # Step 1: 建立SRP散度矩阵
    beta = calculate_beta_from_RPS(RPS_list)
    SRPM = build_SRP_matrix(RPS_list, gamma, beta)
    # print("SRP散度矩阵:")
    # print(SRPM)

    # Step 2: 计算每个RPS的可信度
    credibilities = []
    for i in range(n):
        denominator = sum(SRPM[i, j] for j in range(n))
        credibility = 1 / denominator if denominator > 0 else 0
        credibilities.append(credibility)

    # print(f"可信度: {credibilities}")

    # Step 3: 计算每个RPS的PIQ
    piq_values = [calculate_PIQ(RPS, gamma) for RPS in RPS_list]
    # print(f"PIQ值: {piq_values}")

    # Step 4: 计算加权可信度
    weighted_credibilities = [piq * crd for piq, crd in zip(piq_values, credibilities)]
    # print(f"加权可信度: {weighted_credibilities}")

    # Step 5: 计算每个RPS的权重
    total_weight = sum(weighted_credibilities)
    weights = [w_crd / total_weight if total_weight > 0 else 1 / n for w_crd in weighted_credibilities]
    # print(f"权重: {weights}")

    # Step 6: 生成新的RPS
    all_events = set()
    for RPS in RPS_list:
        all_events.update(RPS.keys())

    new_RPS = defaultdict(float)
    for event in all_events:
        weighted_sum = 0
        for i, RPS in enumerate(RPS_list):
            weighted_sum += weights[i] * RPS.get(event, 0)
        new_RPS[event] = weighted_sum

    total = sum(new_RPS.values())
    for key in new_RPS:
        new_RPS[key] /= total

    # print("新生成的RPS:")
    # for event, weight in sorted(new_RPS.items(), key=lambda x: (-x[1], len(x[0]), x[0])):
    #     print(f"  {event}: {weight:.4f}")

    # Step 7: 使用LOS融合n-1次
    PMFs = [new_RPS] * n

    if n > 1:
        final_result = continuous_left_orthogonal_sum(PMFs)
        # print("最终融合结果:")
        # for event, weight in sorted(final_result.items(), key=lambda x: (-x[1], len(x[0]), x[0])):
        #     print(f"  {event}: {weight:.4f}")
        # total = sum(final_result.values())
        # for key in final_result:
        #     final_result[key] /= total
        return final_result
    else:
        return dict(new_RPS)

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

def classify_sample(test_rps, elements):

    try:
        prob_dist = LOS_SRP_algorithm(test_rps, elements)

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
        results_dir: str = 'cv_results\Zichong_Chen_MC',
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
        'std_accuracy': 0.0
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

if __name__ == '__main__':
    labeled_evidence = [{('A', 'C'): np.float64(0.38978586216939615), ('C',): np.float64(0.021788575925991656),
                      ('B', 'C'): np.float64(0.04188611236167707), ('A',): np.float64(0.8765864352776236),
                      ('A', 'B'): np.float64(0.3699599058057721), ('A', 'B', 'C'): np.float64(0.15345709583547473),
                      ('B',): np.float64(0.10162498879638476)}, {('C', 'B', 'A'): np.float64(0.07573248913424155),
                                                                 ('B', 'A'): np.float64(0.14159703991315375),
                                                                 ('C',): np.float64(0.4479464713406571),
                                                                 ('C', 'B'): np.float64(0.22335734199492727),
                                                                 ('C', 'A'): np.float64(0.22106502721405277),
                                                                 ('B',): np.float64(0.2870186497829432),
                                                                 ('A',): np.float64(0.26503487887639965)}, {
        ('B',): np.float64(3.535880287486165e-08), ('A', 'C'): np.float64(0.4534065330729032),
        ('A', 'B'): np.float64(0.4119038624867854), ('A',): np.float64(0.999999964640819),
        ('A', 'B', 'C'): np.float64(0.17100522542829372), ('C',): np.float64(3.781653142367114e-13),
        ('B', 'C'): np.float64(1.1941789381465987e-08)}, {('B',): np.float64(2.7875087890433815e-08),
                                                          ('A', 'C'): np.float64(0.4561745969379547),
                                                          ('A', 'B', 'C'): np.float64(0.17796638594216732),
                                                          ('A', 'B'): np.float64(0.40403893256141427),
                                                          ('A',): np.float64(0.9999999716680565),
                                                          ('B', 'C'): np.float64(1.008610274197881e-08),
                                                          ('C',): np.float64(4.5685557258858824e-10)}]
    labeled_evidence2 = [{('B',): np.float64(0.026325327131249986), ('C', 'A'): np.float64(0.42245678820493354),
                      ('C', 'B', 'A'): np.float64(0.15510190409452862), ('B', 'A'): np.float64(0.009956248012456663),
                      ('C',): np.float64(0.9736746728662745), ('A',): np.float64(2.4755868649488724e-12),
                      ('C', 'B'): np.float64(0.339396341526945)}, {('C',): np.float64(0.35391643655411364),
                                                                   ('C', 'A'): np.float64(0.15254459589496294),
                                                                   ('B', 'A'): np.float64(0.2652755067834485),
                                                                   ('B', 'C', 'A'): np.float64(0.13377500400103942),
                                                                   ('B',): np.float64(0.5919518151903219),
                                                                   ('B', 'C'): np.float64(0.28745121709183363),
                                                                   ('A',): np.float64(0.05413174825556438)}, {
        ('A',): np.float64(2.0665239473506185e-189), ('B',): np.float64(6.768703894696776e-06),
        ('C', 'B', 'A'): np.float64(0.17579735494262255), ('C',): np.float64(0.9999932312961053),
        ('B', 'A'): np.float64(2.8156854953062905e-06), ('C', 'A'): np.float64(0.4557689252347937),
        ('C', 'B'): np.float64(0.33772810993267804)}, {('B',): np.float64(2.9247936183578867e-05),
                                                       ('C', 'B', 'A'): np.float64(0.1795039853087847),
                                                       ('C', 'A'): np.float64(0.4561612674879903),
                                                       ('C',): np.float64(0.9999707520638164),
                                                       ('B', 'A'): np.float64(1.1817304920636372e-05),
                                                       ('C', 'B'): np.float64(0.35599755996770027),
                                                       ('A',): np.float64(1.5949052515304947e-81)}]

    # 图片中的RPS数据转换为ABCD格式
    example_data2 = [
        {  # RPS₁
            ('A',): 0.75,  # <(γ1), 0.75>
            ('B',): 0.12,  # <(γ2), 0.12>
            ('C', 'D'): 0.13  # <(γ3,γ4), 0.13>
        },
        {  # RPS₂
            ('A',): 0.0001,  # <(γ1), 0.00>
            ('B',): 0.20,  # <(γ2), 0.20>
            ('C', 'D'): 0.80  # <(γ3,γ4), 0.80>
        },
        {  # RPS₃
            ('A',): 0.90,  # <(γ1), 0.90>
            ('B',): 0.05,  # <(γ2), 0.05>
            ('C', 'D'): 0.05  # <(γ3,γ4), 0.05>
        }
    ]
    # # 运行算法
    # print("=" * 60)
    # print("LOS-SRP算法演示（使用图片中的数据，事件空间为ABC）")
    # print("=" * 60)

    # # gamma = ['A', 'B', 'C','D']
    gamma = ['A', 'B', 'C']
    # final_result = LOS_SRP_algorithm(labeled_evidence2, gamma)
    # print("最终融合结果:")
    # for event, weight in sorted(final_result.items(), key=lambda x: (-x[1], len(x[0]), x[0])):
    #     print(f"  {event}: {weight:.4f}")
    # print(f"\n验证: 最终RPS权重和 = {sum(final_result.values()):.6f}")
    # 执行100次五折交叉验证
    # print("=== 100次五折交叉验证 ===")
    # results = cross_validation_with_rps(n_splits=5, n_repeats=100)
    #
    # # 保存结果（可选）
    # print("\n交叉验证完成！")
    # print(f"最终平均准确率: {results['overall_mean']:.4f} ± {results['overall_std']:.4f}")
    # 示例1: 使用Iris数据集
    # breast_cancer数据集最终结果:
    # 平均准确率: 0.3726 ± 0.0409
    # heart数据集最终结果:
    # 平均准确率: 0.5413 ± 0.0609
    # sonar数据集最终结果:
    # 平均准确率: 0.4664 ± 0.0695
    databases = ['sonar']
    for database in databases:
        iris_results = generalized_cross_validation_with_rps(database)
        print("\nIris数据集最终结果:")
        print(f"平均准确率: {iris_results['mean_accuracy']:.4f} ± {iris_results['std_accuracy']:.4f}")