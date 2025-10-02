import numpy as np


def analyze_similarity_matrix(sim_matrix):
    """
    分析相似度矩阵的一致性差值
    """
    n = sim_matrix.shape[0]

    # 移除对角线元素（自相似度）
    off_diag_sim = sim_matrix - np.eye(n)

    print("=== 相似度矩阵分析 ===")
    print("原始矩阵:")
    print(sim_matrix)
    print()

    # 1. 计算每行平均相似度（每个证据与其他证据的平均相似度）
    row_means = []
    for i in range(n):
        # 排除对角线元素
        other_sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i + 1:]])
        mean_sim = np.mean(other_sims)
        row_means.append(mean_sim)
        print(f"证据{i + 1}平均相似度: {mean_sim:.6f}")

    row_means = np.array(row_means)
    print(f"\n各行平均相似度: {row_means}")

    # 2. 方法一：标准差法（推荐）
    std_diff = np.std(row_means)
    print(f"\n方法一(标准差法)一致性差值: {std_diff:.6f}")

    # 3. 方法二：变异系数法
    mean_of_means = np.mean(row_means)
    if mean_of_means > 0:
        cv_diff = std_diff / mean_of_means
        print(f"方法二(变异系数法)一致性差值: {cv_diff:.6f}")
    else:
        print("方法二(变异系数法): 无法计算（均值为0）")

    # 4. 方法三：最大最小差值法
    range_diff = np.max(row_means) - np.min(row_means)
    print(f"方法三(范围法)一致性差值: {range_diff:.6f}")

    # 5. 方法四：基于整体偏离度
    overall_mean = np.mean(off_diag_sim[off_diag_sim > 0])  # 所有非对角线元素的均值
    individual_deviations = np.abs(row_means - overall_mean)
    deviation_diff = np.mean(individual_deviations)
    print(f"方法四(偏离度法)一致性差值: {deviation_diff:.6f}")

    # 6. 识别异常证据（一致性明显不同的证据）
    print(f"\n=== 异常证据识别 ===")
    threshold = np.mean(row_means) - 2 * np.std(row_means)  # 2σ原则
    for i, mean_val in enumerate(row_means):
        if mean_val < threshold:
            print(f"证据{i + 1}可能是异常证据（平均相似度: {mean_val:.6f}）")

    return {
        'std_diff': std_diff,
        'cv_diff': cv_diff if mean_of_means > 0 else 0,
        'range_diff': range_diff,
        'deviation_diff': deviation_diff,
        'individual_scores': row_means
    }


# 您的相似度矩阵
sim_matrix = np.array([
    [1., 0.21382774, 0.89679645, 0.66397344, 0.65322943],
    [0.21382774, 1., 0.28139861, 0.25745693, 0.21629666],
    [0.89679645, 0.28139861, 1., 0.64917656, 0.65376647],
    [0.66397344, 0.25745693, 0.64917656, 1., 0.6843329],
    [0.65322943, 0.21629666, 0.65376647, 0.6843329, 1.]
])

# 执行分析
results = analyze_similarity_matrix(sim_matrix)

print(f"\n=== 推荐使用 ===")
print("1. 标准差法: 最直观，反映绝对差异")
print("2. 变异系数法: 反映相对差异，消除量纲影响")
print(f"您的数据标准差法结果: {results['std_diff']:.6f}")