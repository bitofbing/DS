# 广义证据理论
import numpy as np


# 定义一个基本信念赋值 (BBA) 类
class GBPA:
    # 初始化
    def __init__(self, frame_of_discernment):

        # 识别框架，可以是不完全的
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
        # 这里传入的空集同样可以
        self.m[hypothesis] = mass

    # 广义组合规则（Generalized Combination Rule）
    def GCR(self, other_bba):

        # 新建一个，用于存放计算好的值
        combined_bba = GBPA(self.frame)
        for h1 in self.m:
            for h2 in other_bba.m:
                # 求交集，用h1与h2交集
                # 空集参与求交集
                intersection = h1.intersection(h2)

                # 把两个GPBA的交集累加
                # 且空集也要参与每个求交集
                if len(intersection) <= 1:
                    combined_bba.m[intersection] += self.m[h1] * other_bba.m[h2]

        # 求m(空集)
        m_empty1 = self.m[frozenset([])]
        m_empty2 = other_bba.m[frozenset([])]
        m_empty = m_empty1 * m_empty2

        # 归一化，处理冲突，K就是所有交集为空的总和
        K = combined_bba.m[frozenset([])]
        print()
        # K=1时完全冲突了
        if K > 0 and K < 1:
            for hypothesis in combined_bba.m:

                # 只要不是空集就处理一下
                if hypothesis != frozenset([]):
                    combined_bba.m[hypothesis] *= (1 - m_empty)
                    combined_bba.m[hypothesis] /= (1 - K)
            combined_bba.m[frozenset([])] = m_empty
        elif K == 1:
            combined_bba.m[frozenset([])] = 1
        return combined_bba

    # 计算信任度（Gbelief）
    def Gbelief(self, hypothesis):
        # h 是键值，比如集合frozenset(['A']) mass是键值对应的信任质量 比如0.7
        # h.issubset(hypothesis)判断h是否是hypothesis的子集
        # 简单来说只要h是hypothesis的子集就相加
        # 如果是空集就直接返回
        if hypothesis == frozenset([]):
            return self.m[hypothesis]
        else:
            sumVal = 0
            for h, mass in self.m.items():
                if h.issubset(hypothesis) and h != frozenset([]):
                    sumVal += mass
            return sumVal

    # 打印Gbel
    def print_GBelief(self):
        for hypothesis, mass in self.m.items():
            gbelief = self.Gbelief(hypothesis)
            print(f"GBel {set(hypothesis)}: = {gbelief:.1f}")

    # 计算似然度（plausibility）
    def Gpl(self, hypothesis):
        if hypothesis == frozenset([]):
            return self.m[hypothesis]
        else:
            # 筛选出所有与输入假设 hypothesis 有交集的证据集合。
            # 简单来说只要有交集就相加
            return sum(mass for h, mass in self.m.items() if h.intersection(hypothesis))

    # 打印Gbel
    def print_GPl(self):
        for hypothesis, mass in self.m.items():
            Gpl = self.Gpl(hypothesis)
            print(f"GPl {set(hypothesis)}: = {Gpl:.1f}")

    # 广义证据距离#########
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


def gbpa_distance(bba1, bba2):
    # 将 GPBA转换为向量形式
    # 也就是公式中的m1 m2
    v1 = bba1.to_vector()
    v2 = bba2.to_vector()

    # 计算两个向量的内积
    # 计算相似性矩阵 D
    frame = bba1.frame
    subsets = [frozenset(subset) for subset in bba1.powerset(frame)]

    # 创建一个幂集个数N*N的矩阵
    D = np.zeros((len(subsets), len(subsets)))

    for i, Ai in enumerate(subsets):
        for j, Aj in enumerate(subsets):

            # 两个幂集的交集长度除于两个幂集的并集的长度
            # len(Ai.union(Aj)) 如果并集为空说明是空集相并，直接为1
            D[i, j] = len(Ai.intersection(Aj)) / len(Ai.union(Aj)) if len(Ai.union(Aj)) > 0 else 1

    dotVal = 0
    for i, m1 in enumerate(subsets):
        for j, m2 in enumerate(subsets):
            # 计算两个不同向量的内积
            dotVal += v1[i] * v2[j] * D[i, j]

    # np.dot用于矩阵相乘
    distance = np.sqrt(0.5 * (np.dot(v1, v1) + np.dot(v2, v2) - 2 * dotVal))
    return distance


# 辨识度框架可能不完整，这里是广义前提下
# frame_of_discernment = ['A', 'B', 'C']
frame_of_discernment = ['A', 'B']
# frame_of_discernment = ['B']

# 创建两个GBPA实例
gbpa1 = GBPA(frame_of_discernment)
gbpa2 = GBPA(frame_of_discernment)
# gbpa3 = GBPA(frame_of_discernment)

# 为GBPA分配置信质量
gbpa1.assign_mass('A', 1)
# gbpa1.assign_mass('B', 0.2)
# gbpa1.assign_mass(frozenset([]), 0.9)

gbpa2.assign_mass('B', 0.1)
# gbpa2.assign_mass(frozenset(['B', 'C']), 0.1)
gbpa2.assign_mass(frozenset([]), 0.9)

# gbpa3.assign_mass('A', 0.6)
# gbpa3.assign_mass('B', 0.2)
# gbpa3.assign_mass(frozenset(['B', 'C']), 0.1)
# gbpa3.assign_mass(frozenset([]), 0.1)

# 组合两个GBPA
# combined_bba = gbpa1.GCR(gbpa2)

# 打印组合后的GBPA
# print("组合后的信念赋值：")
# for hypothesis, mass in combined_bba.m.items():
#     print(f"Hypothesis {set(hypothesis)}: Mass = {mass:.3f}")

# 计算命题的信任度
# combined_bba.Gbelief()

# 打印GBel
# print("打印GBel：")
# gbpa3.print_GBelief()

# 打印GPl
# print("打印GPl：")
# gbpa3.print_GPl()
# if __name__ == '__main__':
# h = frozenset(['B'])
# a = frozenset(['B'])
# print(h.issubset(a))

# 计算两个 BBA 的 Jousselme 距离
distance = gbpa_distance(gbpa1, gbpa2)
print(f"gbpa1 和 gbpa2 之间的 gbpa_distance 距离: {distance:.4f}")
