# 1.得到证据距离
# 2.循环遍历两两比较得到相似矩阵
# 3.通过相似矩阵每一行求出mi的sup(mi)，求和得到Crdi
# 4.用每个mi对应的信任度乘于焦元，然后存放到新的集合里，对应的其他mi取出来累加
import numpy as np


# 定义一个基本信念赋值 (BBA) 类
class BPA:
    def __init__(self, frame_of_discernment):

        # 设置一个属性值
        self.frame = frame_of_discernment

        # 把传入的参数设置默认值0
        self.m = {frozenset([hypothesis]): 0 for hypothesis in self.frame}
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

    # 证据组合（Dempster's Rule of Combination）
    def combine(self, other_bba):

        # 新建一个，用于存放计算好的值
        combined_bba = BPA(self.frame)
        combined_bba.assign_mass(frozenset([]), 0)
        for h1 in self.m:
            for h2 in other_bba.m:
                # 求交集，用h1与h2交集
                # 虽然有空集相交，但是空集默认值为0，没有参与累加
                intersection = h1.intersection(h2)

                # 把两个BBA的交集累加
                if len(intersection) > 1:
                    combined_bba.assign_mass(intersection, 0)
                combined_bba.m[intersection] += self.m[h1] * other_bba.m[h2]

        # 归一化，处理冲突，K就是所有交集为空的总和
        total_conflict = combined_bba.m[frozenset([])]
        if total_conflict > 0:
            for hypothesis in combined_bba.m:

                # 只要不是空集就处理一下
                if hypothesis != frozenset([]):
                    combined_bba.m[hypothesis] /= (1 - total_conflict)
            del combined_bba.m[frozenset([])]
        return combined_bba

    # 通过数组得到向量形式
    def to_vector(self, union_bpa):
        # 将 BBA 转换为向量形式
        return np.array([self.m.get(key, 0) for key in union_bpa.m])


# 取出所有的焦元，放到一个综合框架中
def getUnion(n, frame_of_discernment, args):
    # 设置一个新的假设集
    union_bpa = BPA(frame_of_discernment)
    # 把焦元合并
    unionKey = []
    for i in range(0, n):
        i_arg = args[i]
        for h1 in i_arg.m:

            # 如果键值没有在数组中就添加
            if h1 not in unionKey:
                unionKey.append(h1)
    array = np.array(sorted(unionKey))
    for key in array:
        union_bpa.assign_mass(key, 0)
    return union_bpa


def jousselme_distance(bba1, bba2, union_bpa):
    # 将 BBA 转换为向量形式
    # 也就是公式中的m1 m2

    v1 = bba1.to_vector(union_bpa)
    v2 = bba2.to_vector(union_bpa)

    # 创建一个幂集个数N*N的矩阵
    D = np.zeros((len(union_bpa.m), len(union_bpa.m)))

    for i, Ai in enumerate(union_bpa.m):
        for j, Aj in enumerate(union_bpa.m):
            # 两个幂集的交集长度除于两个幂集的并集的长度
            # len(Ai.union(Aj)) 如果并集为空说明是空集相并，直接为1
            D[i, j] = len(Ai.intersection(Aj)) / len(Ai.union(Aj)) if len(Ai.union(Aj)) > 0 else 1

    # 计算 Jousselme 距离
    # 向量相减，得到的是列向量
    diff = v1 - v2

    # np.dot用于矩阵相乘
    distance = np.sqrt(0.5 * np.dot(np.dot(diff.T, D), diff))
    return distance


# 获得两个BOE之间的相似度
def getSimilar(BPAI, BPAJ, union_bpa):
    return 1 - jousselme_distance(BPAI, BPAJ, union_bpa)


# 获取similarity measure matrix (SMM)
def getSMM(n, args, union_bpa):
    SMM = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            # 赋值，构建SMM矩阵
            SMM[i, j] = getSimilar(args[i], args[j], union_bpa)
    return SMM


# 通过SMM计算支持度Crdi
def Crid(n, args, union_bpa):
    smm = getSMM(n, args, union_bpa)
    totalSup = 0.00
    supDict = {}

    # 遍历矩阵smm得到Sup(mi)
    for i in range(0, n):
        sup = 0.00
        for j in range(0, n):
            if i != j:
                sup += smm[i, j]
        totalSup += sup
        supDict[str(i + 1)] = sup

    # 处理数据得到Crdi
    for item in supDict.keys():
        crdi = supDict[item] / totalSup
        supDict[item] = crdi
    return supDict


# 计算加权平均质量
def MAE(frame_of_discernment, n, *args):
    # 组合后的焦元集合
    union_bpa = getUnion(n, frame_of_discernment, args)
    crid = Crid(n, args, union_bpa)
    for i in range(0, n):

        # 按顺序遍历假设集和置信度
        i_arg = args[i]
        i_crid = crid[str(i + 1)]
        for h1 in i_arg.m:
            union_bpa.m[h1] += i_arg.m[h1] * i_crid
    return union_bpa


# 迭代求weightAvg
def weightAvg(avg1, avg2, n):
    # 进行第一次组合
    avg__combine = avg1.combine(avg2)
    # 从1开始循环到n-1，避免多次组合同一对象
    for i in range(1, n - 1):
        avg__combine = avg__combine.combine(avg2)
    return avg__combine


if __name__ == '__main__':
    # 为BBA分配置信质量
    frame_of_discernment = ['A', 'B', 'C']
    n = 3
    # 创建两个BBA实例
    bba1 = BPA(frame_of_discernment)
    bba2 = BPA(frame_of_discernment)
    bba3 = BPA(frame_of_discernment)
    bba4 = BPA(frame_of_discernment)
    bba5 = BPA(frame_of_discernment)

    bba1.assign_mass('A', 0.5)
    bba1.assign_mass('B', 0.2)
    bba1.assign_mass('C', 0.3)

    bba2.assign_mass('A', 0)
    bba2.assign_mass('B', 0.9)
    bba2.assign_mass('C', 0.1)

    bba3.assign_mass('A', 0.55)
    bba3.assign_mass('B', 0.1)
    bba3.assign_mass(frozenset(['A', 'C']), 0.35)

    mae_bpa = MAE(frame_of_discernment, n, bba1, bba2, bba3)
    weight_avg = weightAvg(mae_bpa, mae_bpa, n)
    print("组合后的信念赋值：")
    for hypothesis, mass in weight_avg.m.items():
        print(f"Hypothesis {set(hypothesis)}: Mass = {mass}")
