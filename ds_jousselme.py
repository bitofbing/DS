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

        # 针对6A中的例子，不加上空集
        # self.m[frozenset([])] = 0  # 空集

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

    # 通过6A论文中修改，不是求幂集，而是求焦元的集合
    def getUnion(self, other_bba):
        # 假设合并后会包含 A B C
        union_bpa = BPA(self.frame)

        # 把焦元合并
        unionKey = []
        for h1 in self.m:
            unionKey.append(h1)
        for h2 in other_bba.m:
            if h2 not in unionKey:
                unionKey.append(h2)
        array = np.array(sorted(unionKey))
        for key in array:
            union_bpa.assign_mass(key, 0)
        return union_bpa

    # 遍历幂集，如果幂集有信任质量就取，没有就赋值为0
    # 通过数组得到向量形式
    def to_vector(self, union_bpa):
        # 将 BBA 转换为向量形式
        return np.array([self.m.get(key, 0) for key in union_bpa.m])


def jousselme_distance(bba1, bba2):
    # 将 BBA 转换为向量形式
    # 也就是公式中的m1 m2

    get_union = bba1.getUnion(bba2)
    v1 = bba1.to_vector(get_union)
    v2 = bba2.to_vector(get_union)

    # 创建一个幂集个数N*N的矩阵
    D = np.zeros((len(get_union.m), len(get_union.m)))

    for i, Ai in enumerate(get_union.m):
        for j, Aj in enumerate(get_union.m):
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


frame = ['A', 'B', 'C']
# frame = ['A', 'B']

# 定义两个BBA
bba1 = BPA(frame)
bba1.assign_mass(['A'], 0.3)
bba1.assign_mass(['A', 'B'], 0.4)
bba1.assign_mass(['A', 'B', 'C'], 0.3)

bba2 = BPA(frame)
bba2.assign_mass(['B'], 0.2)
bba2.assign_mass(['C'], 0.3)
bba2.assign_mass(['A', 'B', 'C'], 0.5)
bba1.getUnion(bba2)

# 计算两个 BBA 的 Jousselme 距离
distance = jousselme_distance(bba1, bba2)
print(f"BBA1 和 BBA2 之间的 Jousselme 距离: {distance:.4f}")
