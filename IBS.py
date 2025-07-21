import numpy as np
from scipy.optimize import minimize

# Step 1: 定义 Table 3 中的观测数据，表示为信度区间
observations = [
    {"H1": [0.30, 0.40], "H2": [0.10, 0.25], "H3": [0.25, 0.35], "H_all": [0.10, 0.20]},
    {"H1": [0.35, 0.45], "H2": [0.10, 0.20], "H3": [0.20, 0.30], "H_all": [0.05, 0.15]},
    {"H1": [0.10, 0.25], "H2": [0.30, 0.45], "H3": [0.35, 0.50], "H_all": [0.10, 0.25]},
    {"H1": [0.30, 0.45], "H2": [0.30, 0.50], "H3": [0.15, 0.40], "H_all": [0.00, 0.20]}
]

# Step 2: 定义似然距离函数
def likelihood_distance(theta, observations):
    L_min, L_max = 1, 1  # 初始为 1，用于乘积
    for obs in observations:
        # 计算每个观测的似然下界和上界
        L_min_obs = (
            obs["H1"][0] * theta[0] +
            obs["H2"][0] * theta[1] +
            obs["H3"][0] * theta[2] +
            obs["H_all"][0]
        )
        L_max_obs = (
            obs["H1"][1] * theta[0] +
            obs["H2"][1] * theta[1] +
            obs["H3"][1] * theta[2] +
            obs["H_all"][1]
        )
        L_min *= max(L_min_obs, 0)  # 确保下界不小于 0
        L_max *= min(L_max_obs, 1)  # 确保上界不大于 1
    return -0.5 * (L_min + L_max)  # 负号用于最小化

# Step 3: 定义不确定性度量函数
# 当 α=1 时，不确定性度量为区间宽度的平均值
def uncertainty_measure(theta):
    # 计算区间宽度之和作为不确定性度量
    return (theta[1] - theta[0]) + (theta[2] - theta[1]) + (theta[0] - theta[2])

# Step 4: 目标函数
# 结合似然距离和不确定性度量
def objective(theta, observations, alpha=1):
    return likelihood_distance(theta, observations) - alpha * uncertainty_measure(theta)

# 定义总和约束，确保不确定性度量和三个概率的总和为 1
def constraint_sum_to_one(theta):
    return theta[0] + theta[1] + theta[2] + uncertainty_measure(theta) - 1

constraints = {
    "type": "eq",
    "fun": constraint_sum_to_one
}

# 初始猜测参数
initial_guess = np.array([0.33, 0.33, 0.34])

# 使用局部优化方法找到最佳参数，加入总和约束
result = minimize(
    objective, initial_guess, args=(observations, 1), bounds=[(0, 1), (0, 1), (0, 1)],
    constraints=constraints
)

# 输出优化结果
p_H1, p_H2, p_H3 = result.x
uncertainty = uncertainty_measure(result.x)
print("Estimated Probability Intervals for α = 1:")
print(f"PI(H1) = [{p_H1:.4f}, {p_H1:.4f}]")
print(f"PI(H2) = [{p_H2:.4f}, {p_H2:.4f}]")
print(f"PI(H3) = [{p_H3:.4f}, {p_H3:.4f}]")
print(f"Uncertainty Measure I_α(Θ) = {uncertainty:.4f}")
