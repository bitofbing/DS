import numpy as np
from itertools import combinations


def jousselme_distance(m1, m2):
    """
    计算两个BBA之间的Jousselme距离
    """
    focal_elements = set(m1.keys()).union(set(m2.keys()))
    distance = 0
    for A in focal_elements:
        for B in focal_elements:
            intersection = len(set(A).intersection(set(B))) / len(set(A).union(set(B))) if len(
                set(A).union(set(B))) != 0 else 0
            distance += (m1.get(A, 0) - m2.get(A, 0)) * (m1.get(B, 0) - m2.get(B, 0)) * intersection
    return np.sqrt(0.5 * distance)


def measure_evidential_credibility(evidence):
    """
    测量证据可信度 (4.1节)
    """
    k = len(evidence)

    # Step 1-1: 构建相似性矩阵
    SMM = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                SMM[i, j] = 1  # 与自身的相似度为1
            else:
                d_ij = jousselme_distance(evidence[i], evidence[j])
                SMM[i, j] = 1 - d_ij  # Eq.(10)

    # Step 1-2: 计算支持度
    SD = np.zeros(k)
    for i in range(k):
        SD[i] = np.sum(SMM[i, :]) - 1  # 减去与自身的相似度 (Eq.12)

    # Step 1-3: 计算局部可信度
    CD = SD / np.sum(SD)  # Eq.(13)

    # Step 1-4: 计算全局可信度
    CDg = np.mean(CD)  # Eq.(14)

    return CD, CDg


def adjust_evidential_credibility(CD, CDg, omega=2.25, xi=0.88, eta=0.88):
    """
    调整证据可信度 (4.2节)
    """
    k = len(CD)

    # Step 2-1: 调整可信度
    ACD = np.zeros(k)
    for i in range(k):
        diff = CD[i] - CDg
        if diff >= 0:
            ACD[i] = diff ** xi  # Eq.(17) 第一种情况
        else:
            ACD[i] = -omega * ((-diff) ** eta)  # Eq.(17) 第二种情况

    # Step 2-2: 归一化
    min_ACD = np.min(ACD)
    max_ACD = np.max(ACD)
    ACD_bar = (ACD - min_ACD) / (max_ACD - min_ACD)  # Eq.(18)

    # Step 2-3: 指数函数处理
    exp_ACD = np.exp(ACD_bar)
    ACD_tilde = exp_ACD / np.sum(exp_ACD)  # Eq.(19)

    return ACD_tilde


def generate_weighted_mean_evidence(evidence, weights):
    """
    生成加权平均证据 (4.3节 Step 3-1)
    """
    # 获取所有证据的焦元并集
    focal_elements = set()
    for m in evidence:
        focal_elements.update(m.keys())

    # 创建加权平均证据
    WME = {}
    for fe in focal_elements:
        WME[fe] = 0.0
        for i in range(len(evidence)):
            WME[fe] += weights[i] * evidence[i].get(fe, 0.0)

    return WME


def dempster_combine(m1, m2):
    """
    DS组合规则 (DCR) (Definition 2.3)
    """
    # 获取所有焦元
    focal_elements = set(m1.keys()).union(set(m2.keys()))

    # 计算冲突系数K (Eq.6)
    K = 0
    for A in m1:
        for B in m2:
            if len(set(A).intersection(set(B))) == 0:
                K += m1[A] * m2[B]

    # 组合证据
    m_combined = {}
    for C in focal_elements:
        if C == ():  # 空集
            m_combined[C] = 0.0
        else:
            sum_val = 0.0
            for A in m1:
                for B in m2:
                    if set(A).intersection(set(B)) == set(C):
                        sum_val += m1[A] * m2[B]
            m_combined[C] = sum_val / (1 - K) if (1 - K) != 0 else 0

    return m_combined


def evidence_fusion(evidence):
    """
    完整的证据融合算法
    """
    # 1. 测量证据可信度
    CD, CDg = measure_evidential_credibility(evidence)

    # 2. 调整证据可信度
    weights = adjust_evidential_credibility(CD, CDg)

    # 3. 生成加权平均证据
    WME = generate_weighted_mean_evidence(evidence, weights)

    # 4. 组合证据 (k-1次)
    combined = WME
    for _ in range(len(evidence) - 1):
        combined = dempster_combine(combined, WME)

    return combined


# 示例使用
if __name__ == "__main__":
    # 用户提供的证据
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
    # 执行证据融合
    fused_result = evidence_fusion(evidence1)
    print(fused_result)
    # 打印融合结果
    print("融合结果:")
    for key, value in sorted(fused_result.items(), key=lambda x: -x[1]):
        print(f"{key}: {value}")