import numpy as np
from scipy.optimize import minimize


# 定义 Dempster 组合规则
def dempster_combine(m1, m2):
    # 计算冲突系数 K
    K = 0
    for A in m1:
        for B in m2:
            if set(A).isdisjoint(set(B)):
                K += m1[A] * m2[B]

    # 归一化因子
    if K == 1:
        return None  # 完全冲突，无法融合

    # 计算融合后的 BPA
    m_combined = {}
    for A in m1:
        for B in m2:
            if not set(A).isdisjoint(set(B)):
                key = tuple(sorted(set(A).union(set(B))))
                m_combined[key] = m_combined.get(key, 0) + m1[A] * m2[B]

    # 归一化
    for key in m_combined:
        m_combined[key] /= (1 - K)

    return m_combined


# 定义冲突矩阵计算函数
def compute_conflict_matrix(evidence):
    """
    计算冲突度矩阵：
    - 低冲突（高置信）类别应该在最终融合中权重大
    - 高冲突类别的 R 需要降低
    """
    n = len(evidence)
    C = np.zeros((n, n))  # 冲突矩阵

    for i in range(n):
        for j in range(i + 1, n):
            # 计算两者的冲突度
            ci = evidence[i]
            cj = evidence[j]

            # 冲突度定义：对所有可能性求和
            conflict = sum(ci[h] * cj[h] for h in ci if h in cj)

            C[i, j] = conflict
            C[j, i] = conflict

    return C

def compute_adjusted_conflict_matrix(evidence):
    """
    计算调整后的冲突矩阵：
    - 标准冲突度 C
    - 置信度 S
    - 计算 C_adj = C / (S + ϵ)
    """
    n = len(evidence)
    C = np.zeros((n, n))  # 冲突矩阵
    S = np.zeros(n)  # 置信度矩阵

    for i in range(n):
        S[i] = sum(evidence[i].values())  # 计算每个证据源的置信度
        for j in range(i + 1, n):
            # 计算标准冲突度
            ci = evidence[i]
            cj = evidence[j]
            conflict = sum(ci[h] * cj[h] for h in ci if h in cj)
            C[i, j] = conflict
            C[j, i] = conflict

    a = 1e-10
    C_adj = C / (S[:, None] + a)  # 用置信度 S 进行归一化调整

    return C_adj, S


# 定义软似然函数
def soft_likelihood(omega, m_vals, R, alpha):
    # 归一化可靠性度，避免除以 0
    epsilon = 1e-10  # 小的正数
    R_o = R / (np.sum(R) + epsilon)

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


# 定义目标函数
def objective_function(R, conflict_degrees):
    # 计算冲突系数 K
    K = np.sum(conflict_degrees * R)  # 加权冲突度

    # 惩罚项：确保 R_i 的分布合理，避免某些 R_i 接近 0
    penalty_R_distribution = np.sum(np.abs(R - np.mean(R)))  # 鼓励 R_i 分布均匀

    # 目标函数：最小化冲突 + 惩罚项
    return K + 1 * penalty_R_distribution  # 惩罚项权重可调整

def new_objective_function(R, C_adj, S):
    """
    目标函数：
    - 冲突惩罚（基于调整后冲突度 C_adj）
    - 置信度奖励（增强高可信类别）
    """
    R = np.array(R)
    epsilon = 1e-10

    # 低冲突度优先（调整后的冲突）
    conflict_penalty = np.sum(C_adj * (R[:, None] * R[None, :]))

    # 置信度奖励项（鼓励置信度高的类别）
    confidence_reward = -np.sum(S * R)  # 负号使得优化倾向于较高 S

    # 让 R 不要过度均匀
    reg_term = -np.sum(np.log(R + epsilon))

    return conflict_penalty + 0.5 * reg_term + 0.2 * confidence_reward

# 示例证据体
evidence = [
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 1
    {('B',): 0.5, ('C',): 0.5},  # 证据体 2
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 3
    {('A',): 0.7, ('B',): 0.15, ('C',): 0.15},  # 证据体 4
    {('A', 'C'): 0.8, ('B',): 0.2}  # 证据体 5
]
# evidence = [
#     {('A',): 0.6, ('B',): 0.3, ('A','B'): 0.1},  # 证据体 1
#     {('A',): 0.5, ('B',): 0.4, ('A','B'): 0.1}  # 证据体 2
# ]

# 假设列表（字符串形式）
hypotheses = ['A', 'B', 'C']
# hypotheses = ['A', 'B']

# 计算冲突矩阵和冲突度
C_adj, S = compute_adjusted_conflict_matrix(evidence)
# conflict_degrees = np.sum(C, axis=1)

# 固定 alpha 值
alpha = 0.1

# 初始猜测值（仅优化 R1, R2, R3, R4, R5）
initial_guess = [0.8, 0.7, 0.6, 0.5, 0.4]  # R1, R2, R3, R4, R5
# initial_guess = [0.8, 0.7]  # R1, R2

# 优化
result = minimize(
    new_objective_function,
    initial_guess,
    args=(C_adj, S),
    bounds=[(0, 1)] * len(initial_guess),
    method='trust-constr',
    options={'maxiter': 1000}
)

optimal_R = result.x
print("Optimal R:", optimal_R)

# 应用最优参数计算融合结果
L = {}
for omega in hypotheses:
    # 将假设转换为元组形式
    omega_tuple = (omega,)
    m_vals = [sensor.get(omega_tuple, 0) for sensor in evidence]  # 使用 get 方法避免 KeyError
    L[omega] = soft_likelihood(omega_tuple, m_vals, optimal_R, alpha)

# 归一化融合结果，避免除以 0
epsilon = 1e-10  # 小的正数
total_L = sum(L.values()) + epsilon
Prob = {omega: L[omega] / total_L for omega in hypotheses}

print("Fused probabilities:")
for omega, prob in Prob.items():
    print(f"{omega}: {prob}")

# 验证 alpha=0 时是否与 Dempster 组合规则一致
if alpha == 0:
    # 使用 Dempster 组合规则计算融合结果
    m_combined = evidence[0]
    for i in range(1, len(evidence)):
        m_combined = dempster_combine(m_combined, evidence[i])
    print("Dempster-Shafer fusion result:", m_combined)
