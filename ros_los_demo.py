# from collections import defaultdict
#
# def right_intersection(A, B):
#     """
#     右正交 (RI)，即 B 中去除不在 A 中的元素
#     """
#     return tuple(item for item in B if item in A)
#
# def left_intersection(A, B):
#     """
#     左正交 (LI)，即 A 中去除不在 B 中的元素
#     """
#     return tuple(item for item in A if item in B)
#
# def normalize(result_dict):
#     total = sum(result_dict.values())
#     if total > 0:
#         return {k: round(v/total, 6) for k, v in result_dict.items()}
#     return result_dict
#
# def calculate_KR(M1, M2):
#     """
#     计算右正交和的 K^R (K_R)
#     """
#     K_R = 0
#     for B, wB in M1:
#         for C, wC in M2:
#             if right_intersection(B, C) == ():
#                 product = wB * wC
#                 K_R += product
#     #             if product > 0:
#     #                 print(f"  乘积 wB * wC = {wB:.4f} * {wC:.4f} = {product:.6f}")
#     # print(f"右正交和 K^R = {K_R:.6f}")
#     return K_R
#
# def calculate_KL(M1, M2):
#     """
#     计算左正交和的 K^L (K_L)
#     """
#     K_L = 0
#     for B, wB in M1:
#         for C, wC in M2:
#             if left_intersection(B, C) == ():
#                 product = wB * wC
#                 K_L += product
#     #             if product > 0:
#     #                 print(f"  乘积 wB * wC = {wB:.4f} * {wC:.4f} = {product:.6f}")
#     # print(f"左正交和 K^L = {K_L:.6f}")
#     return K_L
#
# def ROS(M1, M2):
#     """
#     右正交和 (ROS)
#     """
#     K_R = calculate_KR(M1, M2)
#     result = defaultdict(float)
#
#     if K_R != 1:  # 防止 K_R 为 1 时出现除以 0 的情况
#         for A, wA in M1:
#             weight_sum = 0
#             for B, wB in M1:
#                 for C, wC in M2:
#                     if right_intersection(B, C) == A:
#                         weight_sum += wB * wC
#             if weight_sum > 0:
#                 result[A] += (1 / (1 - K_R)) * weight_sum
#
#     return list(result.items())
#
# def LOS(M1, M2):
#     """
#     左正交和 (LOS)
#     """
#     K_L = calculate_KL(M1, M2)
#     result = defaultdict(float)
#
#     if K_L != 1:  # 防止 K_L 为 1 时出现除以 0 的情况
#         for A, wA in M1:
#             weight_sum = 0
#             for B, wB in M1:
#                 for C, wC in M2:
#                     if left_intersection(B, C) == A:
#                         weight_sum += wB * wC
#             if weight_sum > 0:
#                 result[A] += (1 / (1 - K_L)) * weight_sum
#
#     return list(result.items())
#
# def dict_to_pmf(rps_dict):
#     """
#     将字典格式的RPS证据转换为PMF元组列表格式
#     """
#     return [(permutation, weight) for permutation, weight in rps_dict.items()]
#
# def continuous_right_orthogonal_sum(rps_list):
#     """
#     连续右正交和操作，输出字典格式
#     """
#     PMFs = [dict_to_pmf(rps) for rps in rps_list]
#     result = PMFs[0]
#     for i in range(1, len(PMFs)):
#         result = ROS(result, PMFs[i])
#     # 转成字典
#     return normalize({k: round(v, 6) for k, v in result})
#
# def continuous_left_orthogonal_sum(rps_list):
#     """
#     连续左正交和操作，输出字典格式
#     """
#     PMFs = [dict_to_pmf(rps) for rps in rps_list]
#     result = PMFs[0]
#     for i in range(1, len(PMFs)):
#         result = LOS(result, PMFs[i])
#     # 转成字典
#     return normalize({k: round(v, 6) for k, v in result})
from collections import defaultdict

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

rps_lists2 = []
example_1 = [
    {  # RPS₁ - 强烈支持元素A
        ('A',): 0.99,
        ('B', 'C'): 0.005,
        ('C', 'B'): 0.005,
        ('D',): 0.00
    },
    {  # RPS₂ - 强烈支持元素D（与RPS₁完全相反）
        ('A',): 0.00,
        ('B', 'C'): 0.005,
        ('C', 'B'): 0.005,
        ('D',): 0.99
    }
]
# rps_lists2.append(example_1)

# 三组复杂 RPS 证据，组合顺序全部列出
# 强调顺序敏感性的 RPS 示例
# E1 = {
#     ('C',): 0,
#     ('A','B'): 0.1,  # A在B前占比高
#     ('B','A'): 0.1,
#     ('D',):0.8
# }
#
# E2 = {
#     ('C',): 0.9,
#     ('A','B'): 0.05,  # A在B前占比高
#     ('B','A'): 0.05,
#     ('D',):0
# }
#
# E3 = {
#     ('C',): 0.8,
#     ('A','B'): 0.02,  # A在B前占比高
#     ('B','A'): 0.02,
#     ('D',):0.16
# }
# rps_lists = [
#     [E1,E2,E3],
#     [E1,E3,E2],
#     [E2,E1,E3],
#     [E2,E3,E1],
#     [E3,E1,E2],
#     [E3,E2,E1]
# ]

example_conflict = [
    {  # RPS₂ - 强烈支持(C,B)和(B,A,C)排序（与RPS₁完全相反）
        ('A',): 0.0,  # 单元素A的概率为0
        ('B', 'C'): 0.005,  # B>C排序的低概率（共识区域）
        ('C', 'B'): 0.495,  # C>B排序的高概率
        ('A', 'B', 'C'): 0.0,  # A>B>C排序的概率为0
        ('A', 'C', 'B'): 0.005,  # A>C>B排序的低概率（共识区域）
        ('B', 'A', 'C'): 0.495  # B>A>C排序的高概率
    },
    {  # RPS₁ - 强烈支持A和(A,B,C)排序
        ('A',): 0.495,        # 单元素A的高概率
        ('B', 'C'): 0.005,    # B>C排序的低概率
        ('C', 'B'): 0.0,      # C>B排序的概率为0
        ('A', 'B', 'C'): 0.495,  # A>B>C排序的高概率
        ('A', 'C', 'B'): 0.005,  # A>C>B排序的低概率
        ('B', 'A', 'C'): 0.0   # B>A>C排序的概率为0
    },
]
rps_lists2.append(example_conflict)

example5_3_1=   {  # RPS₁
    ('A',): 0.9,
    ('B', 'C'): 0.015,
    ('C', 'B'): 0.0,  # 显式保留零值
    ('D',): 0.085
}
example5_3_2= {  # RPS₂
    ('A',): 0.0,
    ('B', 'C'): 0.015,
    ('C', 'B'): 0.9,  # 注意与RPS₁的顺序相反
    ('D',): 0.085
}
example5_3_3= {  # RPS₃
    ('A',): 0.45,
    ('B', 'C'): 0.015,
    ('C', 'B'): 0.45,  # 顺序敏感值
    ('D',): 0.085
}

example5_5_1 = {  # RPS₁
    ('X',): 0.3,
    ('X', 'Y'): 0.3,
    ('Y', 'X'): 0.3,
    ('Z',): 0.1
}

example5_5_2 = {  # RPS₂
    ('X',): 0.33,
    ('X', 'Y'): 0.33,
    ('Y', 'X'): 0.33,
    ('Z',): 0.01
}

example5_5_3 = {  # RPS₃
    ('X',): 0.3,
    ('X', 'Y'): 0.5,
    ('Y', 'X'): 0.1,
    ('Z',): 0.1
}
# 将 RPS 替换到 E1, E2, E3 的位置
E1 = example5_3_1
E2 = example5_3_2
E3 = example5_3_3

rps_lists3 = [
    [E1,E2,E3],
    [E1,E3,E2],
    [E2,E1,E3],
    [E2,E3,E1],
    [E3,E1,E2],
    [E3,E2,E1]
]
# 使用示例
for idx, e in enumerate(rps_lists3, 1):
    # 连续右正交和
    ros_result = continuous_right_orthogonal_sum(e)
    print(f"RPS组 {idx} 的连续右正交和结果 (E3^R):")
    for perm, weight in ros_result.items():
        print(f"  {perm}: {weight}")

    # 连续左正交和
    # los_result = continuous_left_orthogonal_sum(e)
    # print(f"\nRPS组 {idx} 的连续左正交和结果 (E3^L):")
    # for perm, weight in los_result.items():
    #     print(f"  {perm}: {weight}")

    print("\n" + "-"*50 + "\n")

