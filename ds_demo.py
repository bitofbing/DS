import numpy as np
import math
from collections import defaultdict


# 定义一个基本信念赋值 (BBA) 类
class BBA:
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

    # 证据组合（Dempster's Rule of Combination）
    def combine(self, other_bba):

        # 新建一个，用于存放计算好的值
        combined_bba = BBA(self.frame)
        for h1 in self.m:
            for h2 in other_bba.m:
                # 求交集，用h1与h2交集
                # 虽然有空集相交，但是空集默认值为0，没有参与累加
                intersection = h1.intersection(h2)

                # 把两个BBA的交集累加
                if len(intersection) <= 1:
                    combined_bba.m[intersection] += self.m[h1] * other_bba.m[h2]

        # 归一化，处理冲突，K就是所有交集为空的总和
        total_conflict = combined_bba.m[frozenset([])]
        if total_conflict > 0:
            for hypothesis in combined_bba.m:

                # 只要不是空集就处理一下
                if hypothesis != frozenset([]):
                    combined_bba.m[hypothesis] /= (1 - total_conflict)
        return combined_bba

    # 计算信任度（belief）
    def belief(self, hypothesis):

        # 参数预处理，保证是字符串
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])
        else:
            hypothesis = frozenset(hypothesis)

            # h 是键值，比如集合frozenset(['A']) mass是键值对应的信任质量 比如0.7
            # h.issubset(hypothesis)判断h是否是hypothesis的子集
            # 简单来说只要h是hypothesis的子集就相加
        return sum(mass for h, mass in self.m.items() if h.issubset(hypothesis))

    # 计算似然度（plausibility）
    def plausibility(self, hypothesis):
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])
        else:
            hypothesis = frozenset(hypothesis)

            # 筛选出所有与输入假设 hypothesis 有交集的证据集合。
            # 简单来说只要有交集就相加
        return sum(mass for h, mass in self.m.items() if h.intersection(hypothesis))

    # 将信念质量转换为 Pignistic 概率
    def to_pignistic(self):

        # 字典键值
        pignistic = defaultdict(float)
        for hypothesis, mass in self.m.items():

            # 将每个假设集的质量均匀分配给其元素
            if len(hypothesis) > 0:

                # 遍历键值集合元素，把信任质量平均分
                for element in hypothesis:
                    pignistic[element] += mass / len(hypothesis)
        return pignistic

    # 打印概率距离
    def print_pignistic(self):
        pignistic = self.to_pignistic()
        for element, probability in pignistic.items():
            print(f"BetP({element}) = {probability:.4f}")

def pignistic_distance(bba1, bba2):
    # 将 BBA 转换为 Pignistic 概率
    pignistic1 = bba1.to_pignistic()
    pignistic2 = bba2.to_pignistic()

    # 获取识别框架的全集
    all_elements = bba1.frame.union(bba2.frame)

    # 计算 Pignistic 概率之间的欧氏距离
    distance = 0.0
    for element in all_elements:
        prob1 = pignistic1.get(element, 0.0)
        prob2 = pignistic2.get(element, 0.0)
        distance += (prob1 - prob2) ** 2

    return math.sqrt(distance)

# 示例：假设框架 {A, B, C}
# 广义中也包含没有识别的命题，用空集来体现
# frame_of_discernment = ['A', 'B','C','D']
#
# # 创建两个BBA实例
# bba1 = BBA(frame_of_discernment)
# bba2 = BBA(frame_of_discernment)

# 为BBA分配置信质量
frame_of_discernment = ['A', 'B']

# 创建两个BBA实例
bba1 = BBA(frame_of_discernment)
bba2 = BBA(frame_of_discernment)

# 为BBA分配置信质量
bba1.assign_mass('A', 0.6)
bba1.assign_mass('B', 0.3)
bba1.assign_mass(frozenset(['A', 'B']), 0.1)

bba2.assign_mass('A', 0.5)
bba2.assign_mass('B', 0.4)
bba2.assign_mass(frozenset(['A', 'B']), 0.1)
# bba1.assign_mass('A', 0.5)
# bba1.assign_mass('B', 0.2)
# bba1.assign_mass('C', 0.15)
# bba1.assign_mass('D', 0.1)
# bba1.assign_mass(frozenset(['A', 'B', 'C', 'D']), 0.05)
#
# bba2.assign_mass('A', 0.2)
# bba2.assign_mass('B', 0.7)
# bba2.assign_mass(frozenset(['A', 'B']), 0.1)

# 组合两个BBA
combined_bba = bba1.combine(bba2)

# 打印组合后的BBA
# print("组合后的信念赋值：")
# for hypothesis, mass in combined_bba.m.items():
#     print(f"Hypothesis {set(hypothesis)}: Mass = {mass}")

# 计算信任度和似然度
# belief_A = combined_bba.belief('A')
# plausibility_A = combined_bba.plausibility('A')
#
# print(f"\nBelief(A): {belief_A}")
# print(f"Plausibility(A): {plausibility_A}")

# 打印 Pignistic 概率
print("BBA1 的 Pignistic 概率：")
bba1.print_pignistic()

# 计算两个 BBA 的 Pignistic 概率距离
# distance = pignistic_distance(bba1, bba2)
# print(f"\nBBA1 和 BBA2 之间的 Pignistic 概率距离: {distance:.4f}")