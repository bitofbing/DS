import numpy as np
def compute_deng_entropy(mass_function):
    """改进的邓熵计算，考虑复合假设"""
    masses = np.array(list(mass_function.values()))
    masses = masses / np.sum(masses)

    # 计算基本概率分配的邓熵
    deng_entropy = -np.sum(masses * np.log2(masses + 1e-10))

    # 考虑假设空间复杂度（假设数量）
    hypothesis_complexity = 1 - 1 / len(mass_function)
    return deng_entropy * hypothesis_complexity

def deng_entropy(mass_func):
    """
    计算邓熵（Deng Entropy）
    参数：
    - mass_func: 证据体字典 {frozenset: probability}
    - normalized: 是否归一化到[0,1]范围
    返回：
    - 邓熵值（默认归一化）
    """
    entropy = 0.0
    max_entropy = np.log2(sum(2 ** len(A) - 1 for A in mass_func.keys()))

    for A, m_A in mass_func.items():
        if m_A > 0:
            card = len(A)
            entropy -= m_A * np.log2(m_A / (2 ** card - 1))
        # 考虑假设空间复杂度（假设数量）
    hypothesis_complexity = 1 - 1 / len(mass_func)
    return entropy * hypothesis_complexity

# 案例1：简单假设
mass  =[]
mass1 = {'A': 0.6, 'B': 0.4}
mass.append(mass1)
# 案例2：复合假设
mass2 = {'A': 0.6, 'A|B': 0.4}
mass.append(mass2)
# 案例3：更多假设
mass3 = {'A': 0.4, 'B': 0.3, 'C': 0.3}
mass.append(mass3)
for m in mass:
    entropy = deng_entropy(m)
    # entropy = compute_deng_entropy(m)
    print(entropy)
