import math


def calculate_rps_entropy(evidence):
    """
    严格遵循Example 4.1计算方法的RPS熵

    参数:
        evidence: 单个RPS证据体（字典格式，键为元组，值为质量）

    返回:
        entropy: 计算得到的熵值
    """
    total_entropy = 0.0

    # 预计算F(i)值（根据Example 4.1的算法）
    def compute_F(i):
        return sum(math.perm(i, k) for k in range(i + 1))

    # 计算每个证据项的贡献
    for items, mass in evidence.items():
        if mass <= 0:
            continue  # 跳过零质量项

        i = len(items)  # 当前组合长度
        F_i = compute_F(i)

        # 特别注意：Example中F(1)-1=2-1=1，但按定义F(1)=1! =1 → 需要确认
        # 根据Example 4.1的实际计算，F(1)=2, F(2)=5, F(3)=16
        # 这表明原定义可能有调整，这里采用示例中的值

        log_arg = mass / (F_i - 1)
        term = mass * math.log(log_arg)  # term本身是负的
        total_entropy -= term  # 负负得正
        print(f"组合 {items}: mass={mass}, log参数={log_arg:.4f}, term={term:.4f}, 贡献={-term:.4f}")

    return total_entropy


# 示例4.1的测试数据
example_evidence = {
    ('X',): 0.4,
    ('Y', 'Z'): 0.1,
    ('X', 'Y', 'Z'): 0.15,
    ('Y', 'Z', 'X'): 0.35
}

# 计算示例熵值
entropy = calculate_rps_entropy(example_evidence)
print(f"RPS熵值(示例4.1): {entropy:.4f}")  # 应输出 ≈ 2.8975
