import math
from typing import Dict, Tuple


def compute_rps_entropy_calibrated(evidence: Dict[Tuple, float]) -> float:
    """
    根据例子4.10校准的RPS熵计算

    参数:
        evidence: 单个RPS证据体（字典格式，键为元组，值为质量）

    返回:
        entropy: 计算得到的RPS熵值
    """
    total_entropy = 0.0

    # 根据例子4.10的定义计算F(i)
    def compute_F(i: int) -> int:
        """
        计算F(i) = Σ_{k=0}^{i} P(i,k)
        其中P(i,k) = i!/(i-k)!
        """
        total = 0
        for k in range(0, i + 1):
            if k == 0:
                total += 1  # P(i,0) = 1 (空排列)
            else:
                total += math.perm(i, k)
        return total

    # 计算每个证据项的贡献
    for items, mass in evidence.items():
        if mass <= 0:
            continue  # 跳过零质量项

        i = len(items)  # 当前排列的长度
        F_i = compute_F(i)

        # 验证F(i)值计算（与例子4.10一致）
        if i == 1:
            assert F_i == 2  # F(1)=2
        elif i == 2:
            assert F_i == 5  # F(2)=5
        elif i == 3:
            assert F_i == 16  # F(3)=16

        # 计算分母 F(i)-1
        denominator = F_i - 1
        if denominator <= 0:
            continue  # 避免除以零

        # 计算对数项的参数 M(A_ij)/(F(i)-1)
        log_arg = mass / denominator

        # 计算熵贡献项（使用自然对数）
        if log_arg > 0:
            term = mass * math.log2(log_arg)
            total_entropy -= term  # 负负得正

            # 调试输出
            print(f"组合 {items}: mass={mass}, F({i})={F_i}, 分母={denominator}, "
                  f"log参数={log_arg:.6f}, 贡献={-term:.6f}")

    return total_entropy


# 测试例子4.10
def test_example_4_10():
    """
    使用例子4.10的数据测试校准后的RPS熵计算
    """
    # 例子4.10的数据
    example_evidence = {  # 传感器1
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
    }

    print("=== 测试例子4.10 ===")
    entropy = compute_rps_entropy_calibrated(example_evidence)
    print(f"计算得到的RPS熵: {entropy:.4f}")
    print(f"期望结果: 3.9551")
    print(f"误差: {abs(entropy - 3.9551):.6f}")

    return entropy


# 校准您原来的代码
def compute_rps_entropy_fixed(evidence: Dict[Tuple, float]) -> float:
    """
    修正后的RPS熵计算（保持您原来的函数签名）
    """
    total_entropy = compute_rps_entropy_calibrated(evidence)

    # 可选：如果需要归一化，可以添加以下代码
    # 但根据例子4.10，原始RPS熵不应该被归一化
    max_possible_entropy = math.log(len(evidence) * 10)
    normalize_entropy = min(total_entropy / max_possible_entropy, 1.0)

    return normalize_entropy


if __name__ == "__main__":
    # 测试校准
    test_example_4_10()