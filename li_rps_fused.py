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

# def weighted_subset_rps(rps_dict_list, credibilities, n_min=2):
#     """计算加权子集RPS (Definition 4.4) - 修改为使用字典列表"""
#     all_perms = set()
#     for rps_dict in rps_dict_list:
#         all_perms.update(rps_dict.keys())
#     all_perms = sorted(all_perms, key=lambda x: (len(x), x))
#
#     # 生成所有可能的子集组合（大小≥n_min）
#     n_rps = len(rps_dict_list)
#     weighted_rps_results = {}
#
#     for subset_size in range(n_min, n_rps + 1):
#         for subset_indices in itertools.combinations(range(n_rps), subset_size):
#             # 计算加权质量函数
#             weighted_mass = defaultdict(float)
#             total_cred = 0
#
#             for idx in subset_indices:
#                 cred = credibilities[idx]
#                 total_cred += cred
#                 for perm, mass in rps_dict_list[idx].items():
#                     weighted_mass[perm] += cred * mass
#
#             # 归一化
#             if total_cred > 0:
#                 for perm in weighted_mass:
#                     weighted_mass[perm] /= total_cred
#
#             # 转换为标准格式
#             result_dict = {}
#             for perm in all_perms:
#                 result_dict[perm] = weighted_mass.get(perm, 0.0)
#
#             subset_key = f"WS{subset_size}_{len(weighted_rps_results) + 1}"
#             weighted_rps_results[subset_key] = result_dict
#
#     return weighted_rps_results
def weighted_subset_rps_optimized(rps_dict_list, credibilities, n_min=2, subset_size=None):
    """优化版本：根据Definition 4.4计算加权子集RPS

    参数:
        subset_size: 指定子集大小（None=使用n_min，避免组合爆炸）
    """
    # 预计算所有排列
    all_perms = sorted(set().union(*[r.keys() for r in rps_dict_list]),
                       key=lambda x: (len(x), x))

    n_rps = len(rps_dict_list)
    weighted_rps_results = {}

    # 关键优化：只计算特定大小的子集，避免组合爆炸
    if subset_size is None:
        subset_size = n_min  # 默认使用最小子集大小

    # 只计算指定大小的子集（而不是所有大小）
    valid_sizes = [subset_size] if subset_size >= n_min else [n_min]

    counter = 1
    for size in valid_sizes:
        # 限制最大子集数量，防止组合爆炸
        max_combinations = min(100, math.comb(n_rps, size))  # 安全限制

        for i, subset_indices in enumerate(itertools.combinations(range(n_rps), size)):
            if i >= max_combinations:  # 防止组合爆炸
                break

            # 计算加权质量函数
            weighted_mass = defaultdict(float)
            total_cred = sum(credibilities[idx] for idx in subset_indices)

            if total_cred > 0:
                for idx in subset_indices:
                    cred = credibilities[idx]
                    for perm, mass in rps_dict_list[idx].items():
                        weighted_mass[perm] += cred * mass / total_cred

            # 转换为标准格式
            result_dict = {perm: weighted_mass.get(perm, 0.0) for perm in all_perms}
            weighted_rps_results[f"WS{size}_{counter}"] = result_dict
            counter += 1

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
    weighted_rps = weighted_subset_rps_optimized(example_data, credibility, n_min=2)
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
evidenceList = []
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
        ('C', 'A', 'B'): 0.0,
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

APP = [{('A',): 0.9995900419317356, ('A', 'B'): 0.40774757175979454, ('B',): 0.0004099580682644938}, {('A',): 0.5971853753427087, ('A', 'B'): 0.3393351986359887, ('B',): 0.40281462465729134}, {('A',): 0.9997950023050056, ('A', 'B'): 0.4098747030083355, ('B',): 0.0002049976949943664}, {('A',): 0.9999982298336008, ('A', 'B'): 0.40412826322968365, ('B',): 1.7701663992529018e-06}, {('A',): 0.6920612262248713, ('A', 'B'): 0.3424497300095851, ('B',): 0.3079387737751286}, {('A',): 0.8963005932616178, ('A', 'B'): 0.38622581342131784, ('B',): 0.10369940673838214}, {('A',): 0.9906736422033019, ('A', 'B'): 0.4022039137389094, ('B',): 0.009326357796698066}, {('A',): 0.999999998041602, ('A', 'B'): 0.41673214061608416, ('B',): 1.9583979895726484e-09}, {('A',): 0.6623748005135889, ('A', 'B'): 0.3363555782543691, ('B',): 0.3376251994864111}, {('A',): 0.4834634400801462, ('B',): 0.5165365599198538, ('B', 'A'): 0.24629880744490706}, {('A',): 0.9993326416690039, ('A', 'B'): 0.3767939739381413, ('B',): 0.0006673583309960059}, {('A',): 0.5202197354048541, ('A', 'B'): 0.25083448914158174, ('B',): 0.47978026459514594}, {('A',): 0.986663734205784, ('A', 'B'): 0.37468475072917673, ('B',): 0.013336265794215912}, {('A',): 0.9999999999999492, ('A', 'B'): 0.37230904174591584, ('B',): 5.088926275296195e-14}, {('A',): 0.5109322270220101, ('A', 'B'): 0.2650183851417403, ('B',): 0.4890677729779898}, {('A',): 0.590066617983992, ('A', 'B'): 0.3143265816965441, ('B',): 0.40993338201600804}, {('A',): 0.6290144857774191, ('A', 'B'): 0.27829881598379547, ('B',): 0.37098551422258086}, {('A',): 0.7600103061440523, ('A', 'B'): 0.34358718040661934, ('B',): 0.23998969385594776}, {('A',): 0.4186131782013987, ('B',): 0.5813868217986012, ('B', 'A'): 0.24877825565655684}, {('A',): 0.6152443698793529, ('A', 'B'): 0.2607879736452785, ('B',): 0.38475563012064706}, {('A',): 0.9999913309581715, ('A', 'B'): 0.41505442520567226, ('B',): 8.669041828507591e-06}, {('A',): 0.4660775639937172, ('B',): 0.5339224360062828, ('B', 'A'): 0.2811698191816898}, {('A',): 0.9999665697451503, ('A', 'B'): 0.41628353003043217, ('B',): 3.3430254849671534e-05}, {('A',): 0.9999999996839196, ('A', 'B'): 0.40849288719829213, ('B',): 3.1608031701430925e-10}, {('A',): 0.6056987826268303, ('A', 'B'): 0.3541499497695474, ('B',): 0.3943012173731696}, {('A',): 0.9357954945055119, ('A', 'B'): 0.3859218603928547, ('B',): 0.06420450549448804}, {('A',): 0.834782992273792, ('A', 'B'): 0.39584200903561007, ('B',): 0.16521700772620798}, {('A',): 0.9999567286046183, ('A', 'B'): 0.41826767284730804, ('B',): 4.3271395381658525e-05}, {('A',): 0.8638709618531356, ('A', 'B'): 0.35663192720763737, ('B',): 0.1361290381468644}, {('A',): 0.4452749522183817, ('B',): 0.5547250477816184, ('B', 'A'): 0.21892837117098254}]

results = process_example(APP)
rps_list_r = [copy.deepcopy(results) for _ in range(len(APP))]
orthogonal_sum = continuous_right_orthogonal_sum(rps_list_r)
print(orthogonal_sum)