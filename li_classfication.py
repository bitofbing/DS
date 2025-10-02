import copy
import math
import itertools
from collections.abc import Iterable
import numpy as np
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def get_permutations_from_dict(rps_dict):
    """从RPS字典中获取所有排列"""
    return list(rps_dict.keys())

def get_all_elements_from_dict(rps_dict):
    """从RPS字典中获取所有元素"""
    all_elements = set()
    for perm in rps_dict.keys():
        all_elements.update(perm)
    return sorted(all_elements)

def ordered_degree(A_dict, B_dict):
    """计算两个排列事件的有序度 (Definition 2.7) - 修改为直接使用字典"""
    # 获取交集元素
    intersection_elements = set()
    intersection_elements.update(get_all_elements_from_dict(A_dict))
    intersection_elements.update(get_all_elements_from_dict(B_dict))

    total_rank_diff = 0
    union_count = 0

    for element in intersection_elements:
        # 找到包含该元素的排列
        rank_A = None
        rank_B = None

        for perm_A, mass_A in A_dict.items():
            if mass_A > 0 and element in perm_A:
                rank_A = perm_A.index(element) + 1
                break

        for perm_B, mass_B in B_dict.items():
            if mass_B > 0 and element in perm_B:
                rank_B = perm_B.index(element) + 1
                break

        if rank_A is not None and rank_B is not None:
            total_rank_diff += abs(rank_A - rank_B)
            union_count += 1

    if union_count == 0:
        return 0

    pseudo_deviation = total_rank_diff / union_count
    return math.exp(-pseudo_deviation)

def rps_distance(rps1_dict, rps2_dict):
    """计算两个RPS之间的距离 (Definition 2.8) - 修改为直接使用字典"""
    # 创建向量表示
    all_perms = set(rps1_dict.keys()) | set(rps2_dict.keys())
    all_perms = sorted(all_perms, key=lambda x: (len(x), x))

    vec1 = np.array([rps1_dict.get(perm, 0) for perm in all_perms])
    vec2 = np.array([rps2_dict.get(perm, 0) for perm in all_perms])

    # 构建RD矩阵
    n = len(all_perms)
    RD = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:  # 利用对称性
                perm_i = all_perms[i]
                perm_j = all_perms[j]

                # 计算RD(A,B)
                intersection_size = len(set(perm_i) & set(perm_j))
                union_size = len(set(perm_i) | set(perm_j))

                # 直接使用字典计算有序度
                temp_dict1 = {perm_i: 1.0}
                temp_dict2 = {perm_j: 1.0}
                od_value = ordered_degree(temp_dict1, temp_dict2)

                RD[i, j] = (intersection_size / union_size) * od_value
                RD[j, i] = RD[i, j]  # 对称矩阵

    # 计算距离
    diff = vec1 - vec2
    try:
        distance = np.sqrt(0.5 * diff @ RD @ diff.T)
        return distance
    except:
        # 如果计算失败，返回一个较大的距离
        return 1.0

def similarity_matrix(rps_dict_list):
    """构建RPS相似度矩阵 (Definition 4.1) - 修改为使用字典列表"""
    n = len(rps_dict_list)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                distance = rps_distance(rps_dict_list[i], rps_dict_list[j])
                sim_matrix[i, j] = 1 - distance

    return sim_matrix

def support_degree(sim_matrix):
    """计算每个RPS的支持度 (Definition 4.2)"""
    n = len(sim_matrix)
    support = np.zeros(n)

    for i in range(n):
        support[i] = np.sum(sim_matrix[i]) - 1  # 减去自身相似度

    return support

def credibility_degree(support):
    """计算每个RPS的可信度 (Definition 4.3)"""
    total_support = np.sum(support)
    if total_support == 0:
        return np.ones(len(support)) / len(support)
    return support / total_support

def F_function(i):
    """计算F(i)函数 (Formula 17)"""
    if i <= 0:
        return 0
    result = 0
    for k in range(i + 1):
        if i - k >= 0:
            result += math.factorial(i) / math.factorial(i - k)
    return result

def rps_entropy(rps_dict):
    """计算RPS的熵 (Definition 2.9) - 修改为使用字典"""
    entropy = 0
    all_perms = list(rps_dict.keys())

    for perm, mass in rps_dict.items():
        if mass > 0:
            i = len(perm)  # 排列长度
            F_i = F_function(i)
            if F_i > 1 and mass > 0:
                term = mass * math.log(mass / (F_i - 1))
                entropy -= term

    return entropy

def weighted_subset_rps(rps_dict_list, credibilities, n_min=2):
    """计算加权子集RPS (Definition 4.4) - 修改为使用字典列表"""
    all_perms = set()
    for rps_dict in rps_dict_list:
        all_perms.update(rps_dict.keys())
    all_perms = sorted(all_perms, key=lambda x: (len(x), x))

    # 生成所有可能的子集组合（大小≥n_min）
    n_rps = len(rps_dict_list)
    weighted_rps_results = {}

    for subset_size in range(n_min, n_rps + 1):
        for subset_indices in itertools.combinations(range(n_rps), subset_size):
            # 计算加权质量函数
            weighted_mass = defaultdict(float)
            total_cred = 0

            for idx in subset_indices:
                cred = credibilities[idx]
                total_cred += cred
                for perm, mass in rps_dict_list[idx].items():
                    weighted_mass[perm] += cred * mass

            # 归一化
            if total_cred > 0:
                for perm in weighted_mass:
                    weighted_mass[perm] /= total_cred

            # 转换为标准格式
            result_dict = {}
            for perm in all_perms:
                result_dict[perm] = weighted_mass.get(perm, 0.0)

            subset_key = f"WS{subset_size}_{len(weighted_rps_results) + 1}"
            weighted_rps_results[subset_key] = result_dict

    return weighted_rps_results

def process_example(example_data):
    """处理示例数据 - 修改为使用字典列表"""
    # 步骤1: 构建相似度矩阵
    sim_matrix = similarity_matrix(example_data)
    # print("相似度矩阵:")
    # print(sim_matrix)
    # print()

    # 步骤2: 计算支持度
    support = support_degree(sim_matrix)
    # # print("支持度:")
    # for i, sup in enumerate(support):
    #     print(f"RPS{i + 1}: {sup:.4f}")
    # print()

    # 步骤3: 计算可信度
    credibility = credibility_degree(support)
    # print("可信度:")
    # for i, cred in enumerate(credibility):
    #     print(f"RPS{i + 1}: {cred:.4f}")
    # print()

    # 步骤4: 计算加权子集RPS
    weighted_rps = weighted_subset_rps(example_data, credibility, n_min=2)
    # print("加权子集RPS数量:", len(weighted_rps))
    # print()

    # 步骤5: 计算每个加权子集的熵并找到最小熵
    min_entropy = float('inf')
    min_entropy_key = None
    min_entropy_rps = None

    # print("加权子集熵值:")
    for key, rps_data in weighted_rps.items():
        entropy_val = rps_entropy(rps_data)
        # print(f"{key}: {entropy_val:.4f}")

        if entropy_val < min_entropy:
            min_entropy = entropy_val
            min_entropy_key = key
            min_entropy_rps = rps_data

    # print(f"\n最小熵子集: {min_entropy_key}, 熵值: {min_entropy:.4f}")

    return min_entropy_rps


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


# ==================== 步骤3: 最终分类 ====================
def final_classification(rps_me, rps_std_list, class_labels=None):
    """
    执行最终分类
    :param rps_me: 待分类的RPS (RandomPermutationSet对象)
    :param rps_std_list: 标准RPS列表 (RandomPermutationSet对象列表)
    :param class_labels: 类别标签列表，如果为None则使用索引作为标签
    :return: 分类结果，包含距离值和最终类别
    """
    # 步骤3(i): 计算距离
    distances = []
    for i, rps_std in enumerate(rps_std_list):
        dist = rps_distance(rps_me, rps_std)
        distances.append(dist)
        # print(f"距离 RPS_me 和 RPS_std_{i + 1}: {dist:.6f}")

    # 步骤3(ii): 比较距离值，找到最小距离
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    rps_final = rps_std_list[min_index]
    pred = max(rps_final, key=rps_final.get)
    return pred
    # # 确定类别标签
    # if class_labels is None:
    #     class_label = f"Class_{min_index + 1}"
    # else:
    #     class_label = class_labels[min_index]
    #
    # print(f"\n最小距离: {min_distance:.6f}")
    # print(f"最终分类结果: {class_label}")
    #
    # return {
    #     'distances': distances,
    #     'min_distance': min_distance,
    #     'min_index': min_index,
    #     'class_label': class_label
    # }

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

# 原有的所有类定义和函数保持不变...
# [这里包含您提供的所有类定义和函数，从RandomPermutationSet到final_classification]

# ==================== 新增的交叉验证框架 ====================

def cross_validation_with_rps(n_splits=5, n_repeats=100):
    """100次五折交叉验证主函数"""

    # 加载Iris数据集
    iris = load_iris()
    # 加载 wine 数据集
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # 初始化重复的五折交叉验证
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    all_accuracies = []  # 存储所有重复实验的准确率
    repeat_count = 1  # 重复计数器

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

                # 3. 处理RPS证据（您的核心算法）
                rps_me = process_example(labeled_evidence)

                # 4. 创建标准RPS列表（基于训练集）
                # rps_std_list = create_standard_rps_from_training(X_train_scaled, y_train)
                rps_list_r = [copy.deepcopy(rps_me) for _ in range(len(gen_rps))]
                orthogonal_sum = continuous_right_orthogonal_sum(rps_list_r)

                # 5. 执行最终分类
                classification_result = final_classification(
                    orthogonal_sum,
                    rps_list_r
                )

                # 6. 检查分类是否正确
                true_label = y_test[test_idx]
                # predicted_label = classification_result['min_index']  # 假设min_index对应类别
                predicted_label = classification_result
                true_label_str  = index_to_label[true_label]
                if predicted_label[0] == true_label_str:
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

# 定义映射关系
index_to_label = {0: 'S', 1: 'E', 2: 'V'}

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


def calculate_final_statistics(all_accuracies, n_splits, n_repeats):
    """计算最终统计结果"""
    # 计算每次5折交叉验证的平均值
    repeat_means = []
    for i in range(n_repeats):
        start_idx = i * n_splits
        end_idx = start_idx + n_splits
        repeat_mean = np.mean(all_accuracies[start_idx:end_idx])
        repeat_means.append(repeat_mean)

    # 输出最终结果
    print("=" * 60)
    print(f"{n_repeats}次{n_splits}折交叉验证最终结果:")
    print(f"总折次数: {len(all_accuracies)}")
    print(f"总体平均准确率: {np.mean(all_accuracies):.4f} (±{np.std(all_accuracies):.4f})")
    print(f"每次{n_splits}折交叉验证的平均准确率: {np.mean(repeat_means):.4f} (±{np.std(repeat_means):.4f})")
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

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 执行100次五折交叉验证
    results = cross_validation_with_rps(n_splits=5, n_repeats=100)

    # 保存结果（可选）
    print("\n交叉验证完成！")
    print(f"最终平均准确率: {results['overall_mean']:.4f} ± {results['overall_std']:.4f}")

    # # 执行处理流程
    # results = process_example(ev_rps2)
    # # 创建标准RPS列表 (这里使用原始ev_rps2作为标准RPS)
    # rps_std_list = [RandomPermutationSet(rps_data) for rps_data in ev_rps2]
    # rps_list_r = [copy.deepcopy(results) for _ in range(len(ev_rps2))]
    # orthogonal_sum = continuous_right_orthogonal_sum(rps_list_r)
    # classification_result = final_classification(RandomPermutationSet(orthogonal_sum), rps_std_list)
    # print(classification_result)


# 处理您的示例数据
example5_4 = [
    {  # RPS₁
        ('A', 'B', 'C'): 0.9,
        ('A', 'C', 'B'): 0.0,
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    },
    {  # RPS₂
        ('A', 'B', 'C'): 0.0,
        ('A', 'C', 'B'): 0.9,
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    },
    {  # RPS₃
        ('A', 'B', 'C'): 0.5,
        ('A', 'C', 'B'): 0.4,
        ('B', 'A', 'C'): 0.05,
        ('D',): 0.05
    }
]

evidence_rps = [
    {  # 第一组RPS证据
        ('A',): 0.2,
        ('B',): 0.08,
        ('C',): 0.0,
        ('B', 'A'): 0.05,
        ('A', 'B'): 0.12,
        ('A', 'C'): 0.03,
        ('C', 'A'): 0.0,
        ('A', 'B', 'C'): 0.12,
        ('B', 'A', 'C'): 0.1,
        ('B', 'C', 'A'): 0.05,
        ('A', 'C', 'B'): 0.25,
        ('C', 'A', 'B'): 0.0,
    },
    {  # 第二组RPS证据
        ('A',): 0.07,
        ('B',): 0.13,
        ('C',): 0.02,
        ('B', 'A'): 0.2,
        ('A', 'B'): 0.07,
        ('A', 'C'): 0.1,
        ('C', 'A'): 0.0,
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
        ('C', 'A'): 0.0,
        ('C', 'A', 'B'): 0.05,
        ('B', 'C', 'A'): 0.0,
        ('B', 'A', 'C'): 0.1,
        ('A', 'C', 'B'): 0.3,
        ('A', 'B', 'C'): 0.12
    }
]

ev_rps = [
    {('S', 'V', 'E'): 0.006954555165045033, ('E', 'V', 'S'): 0.025813146220725232, ('E', 'V'): 0.21158161078461896,
     ('V', 'E'): 0.12630526307958156, ('E', 'S', 'V'): 0.018255297906819686, ('V', 'E', 'S'): 0.018497732862412533,
     ('S', 'E', 'V'): 0.011649997381214387, ('E',): 0.5731331342160642, ('V', 'S', 'E'): 0.007809262383518508},
    {('S', 'V', 'E'): 0.012854692513762032, ('E', 'S', 'V'): 0.021139232112424976, ('V', 'E'): 0.1440308601589531,
     ('V', 'E', 'S'): 0.028299447822906273, ('E', 'V'): 0.17645358835615865, ('E',): 0.5541393697255563,
     ('S', 'E', 'V'): 0.015748407103624188, ('V', 'S', 'E'): 0.015191930076137743,
     ('E', 'V', 'S'): 0.03214247213047656},
    {('E', 'V', 'S'): 2.162089895557878e-114, ('E', 'S', 'V'): 8.69849694732277e-116,
     ('S', 'E', 'V'): 5.408959847014939e-116, ('E', 'V'): 0.07626595079639409, ('V',): 0.810388772553717,
     ('V', 'E'): 0.11334527664988885, ('V', 'S', 'E'): 1.8858155485975245e-115, ('S', 'V', 'E'): 8.038712477141707e-116,
     ('V', 'E', 'S'): 3.153958728468296e-114},
    {('V', 'S', 'E'): 4.380326273109402e-46, ('E', 'V'): 0.08573845743356745, ('V', 'E'): 0.08236412407318644,
     ('V', 'E', 'S'): 1.2558151371858575e-45, ('V',): 0.831897418493246, ('S', 'V', 'E'): 3.0133973711356856e-46,
     ('S', 'E', 'V'): 3.1368516953566586e-46, ('E', 'S', 'V'): 4.6968282252751586e-46,
     ('E', 'V', 'S'): 1.2935593182643496e-45}
]

# 运行算法
ev_rps2 = [
    {('S',): 0.8597874519936183, ('V', 'S', 'E'): 0.004826656846152488, ('E', 'V', 'S'): 0.00232304354480607,
     ('E', 'S', 'V'): 0.009386821806016948, ('E', 'S'): 0.02787470476262233, ('S', 'E'): 0.06602657421343949,
     ('S', 'V', 'E'): 0.010250597315175868, ('S', 'E', 'V'): 0.01748646035770921,
     ('V', 'E', 'S'): 0.002037689160459349},
    {('V', 'E'): 0.1657907051681264, ('V', 'S', 'E'): 0.030296772131616363, ('S', 'E', 'V'): 0.020686104084820585,
     ('S', 'V', 'E'): 0.025342744539935916, ('V', 'E', 'S'): 0.035176977695437046, ('E', 'V', 'S'): 0.03138155796666076,
     ('E', 'V'): 0.1353272443716286, ('E', 'S', 'V'): 0.02206161988693075, ('V',): 0.5339362741548436},
    {('E', 'S', 'V'): 4.422459965473793e-17, ('S',): 0.9999999999439214, ('S', 'E', 'V'): 5.922085196355288e-16,
     ('E', 'V', 'S'): 7.114282447383555e-19, ('V', 'E', 'S'): 6.820892889872117e-19,
     ('S', 'V', 'E'): 1.5933039689132633e-16, ('V', 'S', 'E'): 1.1407698214153826e-17,
     ('S', 'E'): 5.291409850222717e-11, ('E', 'S'): 3.163840693120981e-12},
    {('E', 'V', 'S'): 1.7593224080148952e-15, ('S',): 0.9999999945065885, ('S', 'V', 'E'): 8.677908717381138e-15,
     ('V', 'S', 'E'): 3.755678798402833e-15, ('V', 'S'): 9.328831146080873e-10, ('S', 'V'): 4.56048691226305e-09,
     ('V', 'E', 'S'): 1.5111628146043908e-15, ('E', 'S', 'V'): 8.600613185686967e-15,
     ('S', 'E', 'V'): 1.706954224293838e-14}
]

