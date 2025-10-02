import copy
import math
import itertools
from collections.abc import Iterable
import numpy as np
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_iris
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
                term = mass * math.log2(mass / (F_i - 1))
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
    print("相似度矩阵:")
    print(sim_matrix)
    print()

    # 步骤2: 计算支持度
    support = support_degree(sim_matrix)
    print("支持度:")
    for i, sup in enumerate(support):
        print(f"RPS{i + 1}: {sup:.4f}")
    print()

    # 步骤3: 计算可信度
    credibility = credibility_degree(support)
    print("可信度:")
    for i, cred in enumerate(credibility):
        print(f"RPS{i + 1}: {cred:.4f}")
    print()

    # 步骤4: 计算加权子集RPS
    weighted_rps = weighted_subset_rps(example_data, credibility, n_min=2)
    print("加权子集RPS数量:", len(weighted_rps))
    print()

    # 步骤5: 计算每个加权子集的熵并找到最小熵
    min_entropy = float('inf')
    min_entropy_key = None
    min_entropy_rps = None

    print("加权子集熵值:")
    for key, rps_data in weighted_rps.items():
        entropy_val = rps_entropy(rps_data)
        print(f"{key}: {entropy_val:.4f}")

        if entropy_val < min_entropy:
            min_entropy = entropy_val
            min_entropy_key = key
            min_entropy_rps = rps_data

    print(f"\n最小熵子集: {min_entropy_key}, 熵值: {min_entropy:.4f}")

    return min_entropy_rps

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

def calculate_KR(M1, M2):
    """
    计算右正交和的 K^R (K_R)
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

def ROS(M1, M2):
    """
    右正交和 (ROS)
    """
    K_R = calculate_KR(M1, M2)
    result = set()

    if K_R != 1:  # 防止 K_R 为 1 时出现除以 0 的情况
        for A, w1 in M1:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if right_intersection(B, C) == A:
                        weight_sum += w1 * w2
            if weight_sum > 0:
                result.add((A, (1 / (1 - K_R)) * weight_sum))

    return result

def LOS(M1, M2):
    """
    左正交和 (LOS)
    """
    K_L = calculate_KL(M1, M2)
    result = set()

    if K_L != 1:  # 防止 K_L 为 1 时出现除以 0 的情况
        for A, w1 in M1:
            weight_sum = 0
            for B, w1 in M1:
                for C, w2 in M2:
                    if left_intersection(B, C) == A:
                        weight_sum += w1 * w2
            if weight_sum > 0:
                result.add((A, (1 / (1 - K_L)) * weight_sum))

    return result

def dict_to_pmf(rps_dict):
    """
    将字典格式的RPS证据转换为PMF元组列表格式
    """
    return [(permutation, weight) for permutation, weight in rps_dict.items()]

def set_to_dict(result_set):
    """
    将结果集合转换为字典格式
    """
    return {perm: weight for perm, weight in result_set}

def continuous_right_orthogonal_sum(rps_list):
    """
    连续执行右正交和操作
    :param rps_list: 输入的RPS证据列表(字典格式)
    :param normalized: 是否返回归一化结果，默认为True
    :return: 字典格式的右正交和结果
    """
    PMFs = [dict_to_pmf(rps) for rps in rps_list]
    result = PMFs[0]
    for i in range(1, len(PMFs)):
        result = ROS(result, PMFs[i])

    result_dict = set_to_dict(result)
    return result_dict

def continuous_left_orthogonal_sum(rps_list):
    """
    连续执行左正交和操作
    :param rps_list: 输入的RPS证据列表(字典格式)
    :param normalized: 是否返回归一化结果，默认为True
    :return: 字典格式的左正交和结果
    """
    PMFs = [dict_to_pmf(rps) for rps in rps_list]
    result = PMFs[0]
    for i in range(1, len(PMFs)):
        result = LOS(result, PMFs[i])

    result_dict = set_to_dict(result)
    return result_dict

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

ev_rps = [{('S', 'V', 'E'): 0.006954555165045033, ('E', 'V', 'S'): 0.025813146220725232, ('E', 'V'): 0.21158161078461896, ('V', 'E'): 0.12630526307958156, ('E', 'S', 'V'): 0.018255297906819686, ('V', 'E', 'S'): 0.018497732862412533, ('S', 'E', 'V'): 0.011649997381214387, ('E',): 0.5731331342160642, ('V', 'S', 'E'): 0.007809262383518508}, {('S', 'V', 'E'): 0.012854692513762032, ('E', 'S', 'V'): 0.021139232112424976, ('V', 'E'): 0.1440308601589531, ('V', 'E', 'S'): 0.028299447822906273, ('E', 'V'): 0.17645358835615865, ('E',): 0.5541393697255563, ('S', 'E', 'V'): 0.015748407103624188, ('V', 'S', 'E'): 0.015191930076137743, ('E', 'V', 'S'): 0.03214247213047656}, {('E', 'V', 'S'): 2.162089895557878e-114, ('E', 'S', 'V'): 8.69849694732277e-116, ('S', 'E', 'V'): 5.408959847014939e-116, ('E', 'V'): 0.07626595079639409, ('V',): 0.810388772553717, ('V', 'E'): 0.11334527664988885, ('V', 'S', 'E'): 1.8858155485975245e-115, ('S', 'V', 'E'): 8.038712477141707e-116, ('V', 'E', 'S'): 3.153958728468296e-114}, {('V', 'S', 'E'): 4.380326273109402e-46, ('E', 'V'): 0.08573845743356745, ('V', 'E'): 0.08236412407318644, ('V', 'E', 'S'): 1.2558151371858575e-45, ('V',): 0.831897418493246, ('S', 'V', 'E'): 3.0133973711356856e-46, ('S', 'E', 'V'): 3.1368516953566586e-46, ('E', 'S', 'V'): 4.6968282252751586e-46, ('E', 'V', 'S'): 1.2935593182643496e-45}]# 运行算法
ev_rps2 = [{('S',): 0.8597874519936183, ('V', 'S', 'E'): 0.004826656846152488, ('E', 'V', 'S'): 0.00232304354480607, ('E', 'S', 'V'): 0.009386821806016948, ('E', 'S'): 0.02787470476262233, ('S', 'E'): 0.06602657421343949, ('S', 'V', 'E'): 0.010250597315175868, ('S', 'E', 'V'): 0.01748646035770921, ('V', 'E', 'S'): 0.002037689160459349}, {('V', 'E'): 0.1657907051681264, ('V', 'S', 'E'): 0.030296772131616363, ('S', 'E', 'V'): 0.020686104084820585, ('S', 'V', 'E'): 0.025342744539935916, ('V', 'E', 'S'): 0.035176977695437046, ('E', 'V', 'S'): 0.03138155796666076, ('E', 'V'): 0.1353272443716286, ('E', 'S', 'V'): 0.02206161988693075, ('V',): 0.5339362741548436}, {('E', 'S', 'V'): 4.422459965473793e-17, ('S',): 0.9999999999439214, ('S', 'E', 'V'): 5.922085196355288e-16, ('E', 'V', 'S'): 7.114282447383555e-19, ('V', 'E', 'S'): 6.820892889872117e-19, ('S', 'V', 'E'): 1.5933039689132633e-16, ('V', 'S', 'E'): 1.1407698214153826e-17, ('S', 'E'): 5.291409850222717e-11, ('E', 'S'): 3.163840693120981e-12}, {('E', 'V', 'S'): 1.7593224080148952e-15, ('S',): 0.9999999945065885, ('S', 'V', 'E'): 8.677908717381138e-15, ('V', 'S', 'E'): 3.755678798402833e-15, ('V', 'S'): 9.328831146080873e-10, ('S', 'V'): 4.56048691226305e-09, ('V', 'E', 'S'): 1.5111628146043908e-15, ('E', 'S', 'V'): 8.600613185686967e-15, ('S', 'E', 'V'): 1.706954224293838e-14}]
results = process_example(evidence_rps)
rps_list_r = [copy.deepcopy(results) for _ in range(len(evidence_rps))]
orthogonal_sum = continuous_right_orthogonal_sum(rps_list_r)
print(orthogonal_sum)