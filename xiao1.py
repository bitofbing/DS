import numpy as np
from math import log


def belief_jensen_shannon_divergence(m1, m2):
    """
    计算两个基本置信分配(BBA)之间的信念詹森-香农散度(BJS)
    对应Definition 3.1和公式(13)-(14)

    参数:
        m1, m2: 两个BBA字典，键为命题元组，值为置信度

    返回:
        BJS散度值
    """
    # 获取所有命题的并集
    propositions = set(m1.keys()).union(set(m2.keys()))

    # 计算中间量m_avg = (m1 + m2)/2
    m_avg = {}
    for prop in propositions:
        m_avg[prop] = (m1.get(prop, 0) + m2.get(prop, 0)) / 2

    # 计算S(m1, m_avg)和S(m2, m_avg)
    S1, S2 = 0, 0
    for prop in propositions:
        p1 = m1.get(prop, 0)
        p2 = m2.get(prop, 0)
        p_avg = m_avg.get(prop, 0)

        if p1 > 0 and p_avg > 0:
            S1 += p1 * log(p1 / p_avg)
        if p2 > 0 and p_avg > 0:
            S2 += p2 * log(p2 / p_avg)

    # 计算BJS散度
    BJS = 0.5 * (S1 + S2)
    return BJS


def calculate_credibility_degree(evidences):
    """
    计算证据的可信度(4.1节)

    参数:
        evidences: 证据列表，每个证据是一个BBA字典

    返回:
        credibility_degrees: 每个证据的可信度列表
    """
    k = len(evidences)
    if k == 0:
        return []

    # Step 1-1: 构建BJS散度矩阵
    DMM = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                DMM[i][j] = belief_jensen_shannon_divergence(evidences[i], evidences[j])

    # Step 1-2: 计算每个证据的平均距离
    avg_distances = []
    for i in range(k):
        sum_dist = sum(DMM[i])  # 对角线为0，所以直接求和
        avg_dist = sum_dist / (k - 1)
        avg_distances.append(avg_dist)

    # Step 1-3: 计算支持度
    supports = []
    for dist in avg_distances:
        if dist == 0:
            supports.append(float('inf'))  # 理论上不应该发生
        else:
            supports.append(1 / dist)

    # Step 1-4: 计算可信度
    sum_support = sum(supports)
    credibility_degrees = [sup / sum_support for sup in supports]

    return credibility_degrees


def measure_information_volume(bba):
    """
    计算单个证据的信息量(公式19)
    对应4.2节Step 2-2

    参数:
        bba: 基本置信分配字典，键为命题元组

    返回:
        信息量值
    """
    entropy = 0
    for prop, mass in bba.items():
        if mass > 0:
            cardinality = len(prop)  # 元组的长度即为命题的基数
            denominator = (2 ** cardinality) - 1
            entropy += mass * log(mass / denominator) if denominator != 0 else 0
    return np.exp(-entropy)


def normalize_information_volumes(evidences):
    """
    计算并归一化所有证据的信息量(4.2节)

    参数:
        evidences: 证据列表

    返回:
        normalized_IVs: 归一化后的信息量列表
    """
    # Step 2-1和Step 2-2: 计算每个证据的信息量
    IVs = [measure_information_volume(ev) for ev in evidences]

    # Step 2-3: 归一化信息量
    sum_IV = sum(IVs)
    normalized_IVs = [iv / sum_IV for iv in IVs]

    return normalized_IVs


def generate_weighted_average_evidence(evidences):
    """
    生成并融合加权平均证据(4.3节)

    参数:
        evidences: 证据列表

    返回:
        WAE: 加权平均证据(BBA字典)
    """
    k = len(evidences)
    if k == 0:
        return {}

    # Step 1: 计算可信度
    credibility_degrees = calculate_credibility_degree(evidences)

    # Step 2: 计算归一化信息量
    normalized_IVs = normalize_information_volumes(evidences)

    # Step 3-1: 调整可信度
    ACrd = [credibility_degrees[i] * normalized_IVs[i] for i in range(k)]

    # Step 3-2: 归一化调整后的可信度
    sum_ACrd = sum(ACrd)
    final_weights = [acrd / sum_ACrd for acrd in ACrd]

    # Step 3-3: 计算加权平均证据
    WAE = {}
    # 获取所有可能的命题
    all_propositions = set()
    for ev in evidences:
        all_propositions.update(ev.keys())

    for prop in all_propositions:
        weighted_mass = 0
        for i in range(k):
            weighted_mass += final_weights[i] * evidences[i].get(prop, 0)
        WAE[prop] = weighted_mass

    return WAE


def dempster_combination(m1, m2):
    """
    Dempster合成规则(对应Definition 2.4和公式7-8)

    参数:
        m1, m2: 两个BBA字典，键为命题元组

    返回:
        融合后的BBA字典
    """
    # 获取所有命题
    propositions = set(m1.keys()).union(set(m2.keys()))

    # 计算冲突系数K
    K = 0
    for b in m1:
        for c in m2:
            if not set(b).intersection(set(c)):  # 元组交集为空
                K += m1[b] * m2[c]

    # 合成规则
    m = {}
    for a in propositions:
        mass = 0
        for b in m1:
            for c in m2:
                if set(b).intersection(set(c)) == set(a):  # 元组交集等于a
                    mass += m1[b] * m2[c]
        if a:  # 非空集
            m[a] = mass / (1 - K) if (1 - K) != 0 else 0

    return m


def fuse_evidences(evidences):
    """
    完整证据融合流程(对应4.3节Step 3-4)

    参数:
        evidences: 证据列表，每个证据是键为命题元组的BBA字典

    返回:
        融合后的最终BBA
    """
    if not evidences:
        return {}

    # 获取加权平均证据
    WAE = generate_weighted_average_evidence(evidences)

    # 多次应用Dempster规则融合
    result = WAE
    for _ in range(len(evidences) - 1):
        result = dempster_combination(result, WAE)

    return result


# 使用示例
if __name__ == "__main__":
    evidence = [
        {('A',): 0.0437, ('B',): 0.3346, ('C',): 0.2916, ('A', 'B'): 0.0437, ('A', 'C'): 0.0239, ('B', 'C'): 0.2385,
         ('A', 'B', 'C'): 0.0239},  # 证据体 1
        {('A',): 0.0865, ('B',): 0.2879, ('C',): 0.1839, ('A', 'B'): 0.0863, ('A', 'C'): 0.0865, ('B', 'C'): 0.1825,
         ('A', 'B', 'C'): 0.0863},  # 证据体 2
        {('A',): 1.4e-09, ('B',): 0.6570, ('C',): 0.1726, ('A', 'B'): 1.3e-09, ('A', 'C'): 1.4e-11, ('B', 'C'): 0.1704,
         ('A', 'B', 'C'): 1.4e-11},  # 证据体 3
        {('A',): 8.20e-06, ('B',): 0.6616, ('C',): 0.1692, ('A', 'B'): 8.20e-06, ('A', 'C'): 3.80e-06,
         ('B', 'C'): 0.1692, ('A', 'B', 'C'): 3.80e-06}  # 证据体 4
    ]
    evidence1 = [{('S',): np.float64(0.4236476980970981), ('E',): np.float64(0.19275700671919366),
      ('V',): np.float64(0.04765507822569373), ('E', 'S'): np.float64(0.19296539273886568),
      ('S', 'V'): np.float64(0.04765987299672752), ('E', 'V'): np.float64(0.04765507822569373),
      ('E', 'S', 'V'): np.float64(0.04765987299672752)},
     {('S',): np.float64(0.245953491595259), ('E',): np.float64(0.1118661936397038),
      ('V',): np.float64(0.15319538560231452), ('E', 'S'): np.float64(0.11190804722504827),
      ('S', 'V'): np.float64(0.15330264107292238), ('E', 'V'): np.float64(0.1118661936397038),
      ('E', 'S', 'V'): np.float64(0.11190804722504827)},
     {('S',): np.float64(0.9999343205062626), ('E',): np.float64(3.2826154380314705e-05),
      ('V',): np.float64(5.493238048271591e-09), ('E', 'S'): np.float64(3.2828958416371255e-05),
      ('S', 'V'): np.float64(6.69723234363771e-09), ('E', 'V'): np.float64(5.493238048271591e-09),
      ('E', 'S', 'V'): np.float64(6.69723234363771e-09)},
     {('S',): np.float64(0.9997504490121758), ('E',): np.float64(0.00011576838900277515),
      ('V',): np.float64(4.503664796054944e-06), ('E', 'S'): np.float64(0.00011576849029658349),
      ('S', 'V'): np.float64(4.5033930648984895e-06), ('E', 'V'): np.float64(4.503664796054944e-06),
      ('E', 'S', 'V'): np.float64(4.5033858679814855e-06)}]
    # # 计算可信度 (4.1节)
    # credibility = calculate_credibility_degree(evidence)
    # print("Credibility degrees:", credibility)
    #
    # # 计算信息量 (4.2节)
    # info_volumes = normalize_information_volumes(evidence)
    # print("Normalized information volumes:", info_volumes)

    # 生成加权平均证据并融合 (4.3节)
    final_result = fuse_evidences(evidence1)
    print(final_result)
    # 打印融合结果，按置信度降序排列
    print("\nFinal fusion result (sorted by mass):")
    for prop, mass in sorted(final_result.items(), key=lambda x: x[1], reverse=True):
        print(f"{prop}: {mass:.6f}")