import numpy as np
import math
from collections import defaultdict


# 定义一个基本信念赋值 (BBA) 类
class BPA:
    def __init__(self, frame_of_discernment):

        # 设置一个属性值
        self.frame = frame_of_discernment

        # 把传入的参数设置默认值0
        self.m = {frozenset([hypothesis]): 0 for hypothesis in self.frame}
        self.m[frozenset([])] = 0  # 空集

    # 分配信念质量
    def assign_mass(self, hypothesis, mass):
        # 直接检查 hypothesis 是否为空集或在识别框架内
        # all(h in self.frame for h in hypothesis)：确保传入的 hypothesis 中的每个元素都在识别框架（self.frame）中
        assert hypothesis == frozenset() or all(h in self.frame for h in hypothesis), "假设不在识别框架中"

        # 如果 hypothesis 是字符串，转换为单元素的 frozenset
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])

        # 否则，确保 hypothesis 是 frozenset 类型，如果不是则将其转换为 frozenset
        elif not isinstance(hypothesis, frozenset):
            hypothesis = frozenset(hypothesis)

        # 赋予信念质量
        self.m[hypothesis] = mass

    # 遍历幂集，如果幂集有信任质量就取，没有就赋值为0
    # 通过数组得到向量形式
    def to_vector(self):
         # 将 BBA 转换为向量形式
        subsets = [frozenset(subset) for subset in self.powerset(self.frame)]
        return np.array([self.m.get(subset, 0) for subset in subsets])

# 求出A B C能组合的所有幂集
    @staticmethod
    def powerset(s):
         # 生成识别框架[A B C]的幂集
        x = len(s)
        return [[s[j] for j in range(x) if (i & (1 << j))] for i in range(1 << x)]

def jousselme_distance(bba1, bba2):
    # 将 BBA 转换为向量形式
    # 也就是公式中的m1 m2
    v1 = bba1.to_vector()
    v2 = bba2.to_vector()

    # 计算相似性矩阵 D
    frame = bba1.frame

    # 组成幂集
    subsets = [frozenset(subset) for subset in bba1.powerset(frame)]

    # 创建一个幂集个数N*N的矩阵
    D = np.zeros((len(subsets), len(subsets)))

    for i, Ai in enumerate(subsets):
        for j, Aj in enumerate(subsets):
            # 两个幂集的交集长度除于两个幂集的并集的长度
            # len(Ai.union(Aj)) 如果并集为空说明是空集相并，直接为1
            D[i, j] = len(Ai.intersection(Aj)) / len(Ai.union(Aj)) if len(Ai.union(Aj)) > 0 else 1

    # 计算 Jousselme 距离
    # 向量相减，得到的是列向量
    diff = v1 - v2

    # np.dot用于矩阵相乘
    distance = np.sqrt(0.5 * np.dot(np.dot(diff.T, D), diff))
    return distance

    # 示例
# frame = ['A', 'B', 'C']
frame = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

# 定义两个BBA
bba1 = BPA(frame)
bba1.assign_mass(['2', '3', '4'], 0.05)
bba1.assign_mass(['7'], 0.05)
bba1.assign_mass(['1'], 0.8)
bba1.assign_mass(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'], 0.8)

bba2 = BPA(frame)
bba2.assign_mass(['1', '2', '3', '4', '5'], 1)

# 计算两个 BBA 的 Jousselme 距离
distance = jousselme_distance(bba1, bba2)
print(f"BBA1 和 BBA2 之间的 Jousselme 距离: {distance:.4f}")