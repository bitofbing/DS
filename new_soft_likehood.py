import numpy as np
from scipy.optimize import minimize


def bhattacharyya_distance(p, q):
    """
    计算 Bhattacharyya 距离
    - p 和 q 是长度相同的概率分布
    """
    p = np.array(p)
    q = np.array(q)
    return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)  # 避免 log(0)


def compute_similarity_matrix(evidence):
    """
    计算相似度矩阵 S，基于 Bhattacharyya 距离
    - 统一假设空间，确保所有证据源的概率分布长度一致
    """
    n = len(evidence)

    # 找到所有证据源支持的假设的并集
    hypotheses = set()
    for e in evidence:
        hypotheses.update(e.keys())
    hypotheses = sorted(hypotheses)  # 排序以保证一致性

    # 将每个证据源的概率分布扩展到统一的假设空间
    expanded_evidence = []
    for e in evidence:
        expanded_e = []
        for h in hypotheses:
            expanded_e.append(e.get(h, 0))  # 如果假设不在证据源中，概率设为 0
        expanded_evidence.append(expanded_e)

    # 计算相似度矩阵
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            p = expanded_evidence[i]
            q = expanded_evidence[j]
            distance = bhattacharyya_distance(p, q)
            similarity = np.exp(-distance)  # 转化为相似度
            S[i, j] = similarity
            S[j, i] = similarity  # 矩阵对称

    return S

def soft_likelihood(omega, m_vals, R, alpha):
    """
    软似然函数：
    - omega: 假设
    - m_vals: 证据源对假设的支持度
    - R: 证据源的可信度权重
    - alpha: 控制参数
    """
    epsilon = 1e-10  # 小的正数
    R_o = R / (np.sum(R) + epsilon)  # 归一化可信度权重

    # 计算权重
    w = []
    if alpha == 0:
        # 当 alpha=0 时，退化为 Dempster 组合规则的权重
        w = R_o
    else:
        # 否则，使用软似然函数的权重公式
        for i in range(len(m_vals)):
            if i == 0:
                w.append(R_o[i] ** ((1 - alpha) / alpha))
            else:
                w.append((np.sum(R_o[:i + 1]) ** ((1 - alpha) / alpha) - (np.sum(R_o[:i]) ** ((1 - alpha) / alpha))))

    # 计算软似然值
    L = 0
    for i in range(len(m_vals)):
        prod = 1
        for k in range(i + 1):
            prod *= m_vals[k]  # m_vals[k] 已经是浮点数，不需要下标操作
        L += w[i] * prod

    return L

def objective_function(R, S):
    """
    目标函数：
    - 最大化权重与相似度的相关性
    - 约束条件：权重 R 需要归一化（sum(R) = 1）
    """
    R = np.array(R)
    epsilon = 1e-10

    # 计算权重与相似度的相关性
    correlation = np.sum(S * (R[:, None] * R[None, :]))

    # 正则化项：鼓励权重分布合理
    reg_term = -np.sum(np.log(R + epsilon))  # 避免权重为 0

    # 目标函数：最大化相关性 + 正则化项
    return -correlation + 0.1 * reg_term  # 负号用于最小化

# 示例证据体
evidence = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 1
    {('B',): 0.5, ('C',): 0.5},  # 证据体 2
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 3
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 4
    {('A', 'C'): 0.8, ('B',): 0.2}  # 证据体 5
]

# 假设列表
hypotheses = ['A', 'B', 'C']

# 计算相似度矩阵
S = compute_similarity_matrix(evidence)
print("Similarity Matrix S:\n", S)

# 初始猜测值（权重 R1, R2, R3, R4, R5）
initial_guess = [0.2, 0.2, 0.2, 0.2, 0.2]  # 初始权重均匀分布

# 设置 alpha 值
alpha = 0.1

# 优化权重
result = minimize(
    objective_function,
    initial_guess,
    args=(S,),
    bounds=[(0, 1)] * len(initial_guess),  # 权重在 [0, 1] 范围内
    constraints={'type': 'eq', 'fun': lambda R: np.sum(R) - 1},  # 约束条件：sum(R) = 1
    method='SLSQP',  # 使用 SLSQP 优化算法
    options={'maxiter': 1000}
)

# 输出最优权重
optimal_R = result.x
print("Optimal Weights R:", optimal_R)

# 计算软似然值
L = {}
for omega in hypotheses:
    omega_tuple = (omega,)
    m_vals = [sensor.get(omega_tuple, 0) for sensor in evidence]  # 获取证据源对假设的支持度
    L[omega] = soft_likelihood(omega_tuple, m_vals, optimal_R, alpha)

# 归一化软似然值
total_L = sum(L.values()) + 1e-10
Prob = {omega: L[omega] / total_L for omega in hypotheses}

print("Fused Probabilities:")
for omega, prob in Prob.items():
    print(f"{omega}: {prob}")