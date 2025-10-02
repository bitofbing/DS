import math
from typing import Dict, List, Tuple, Union
import numpy as np
def calculate_pic(probabilities: Union[Dict[str, float], List[float], np.ndarray],
                 elements: List[str] = None) -> float:
    """
    计算概率信息量 (Probabilistic Information Content - PIC)

    根据公式: PIC(P) = 1 + (1/log₂N) * Σ P(θ)log₂(P(θ))

    Args:
        probabilities: 概率分布，可以是:
            - 字典: {元素: 概率}
            - 列表: 概率值列表（需要提供elements参数）
            - numpy数组: 概率值数组（需要提供elements参数）
        elements: 可选，元素名称列表，当probabilities不是字典时需要提供

    Returns:
        float: PIC值，范围[0, 1]

    Raises:
        ValueError: 概率和为1或包含无效值
    """
    # 处理不同类型的输入
    if isinstance(probabilities, dict):
        probs = list(probabilities.values())
        N = len(probabilities)
    else:
        if elements is not None:
            N = len(elements)
        else:
            N = len(probabilities)
        probs = probabilities

    # 转换为numpy数组便于计算
    probs = np.array(probs)

    # 验证输入
    if not np.all(probs >= 0):
        raise ValueError("所有概率值必须非负")

    if not math.isclose(np.sum(probs), 1.0, rel_tol=1e-10):
        raise ValueError(f"概率和必须为1，当前和为{np.sum(probs):.6f}")

    if N <= 0:
        raise ValueError("元素数量必须大于0")

    # 计算PIC
    entropy_term = 0.0
    for p in probs:
        if p > 0:  # 避免log(0)
            entropy_term += p * math.log2(p)

    # PIC = 1 - 归一化香农熵
    pic_value = 1 + (1 / math.log2(N)) * entropy_term

    return pic_value
# 0.4887, p(θ2) = 0.0178, p(θ3) = 0.4935
fused_end ={'A': 0.4887,'B': 0.0178,'C':0.4935}
pic_uniform = calculate_pic(fused_end)
print(f"均匀分布: PIC = {pic_uniform:.6f}")