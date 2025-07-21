# （5）Murphy C K——Combining belief functions when evidence conflicts
# 求出两个假设的综合各项的平均值，循环n-1次，有两个假设就循环一次，在循环中执行DS-combine
# 如果是更多次，就利用上次组合得到的继续和平均值组合
import math

import numpy as np


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
        return combined_bba


# 取出所有的焦元，放到一个综合框架中
def getUnion(n, frame_of_discernment, args):
    # 设置一个新的假设集
    union_bpa = BBA(frame_of_discernment)
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


# 求两个假设的所有焦元的平均值,并合并到一个新的BPA中
# 假设输入的假设已经按照字典顺序了排列了
def average(n, frame_of_discernment, *args):
    union_bba = getUnion(n, frame_of_discernment, args)
    for i in range(0, n):
        i_arg = args[i]
        for h1 in i_arg.m:
            union_bba.m[h1] += i_arg.m[h1]
    for h2 in union_bba.m:
        union_bba.m[h2] /= n
    return union_bba


# 迭代求weightAvg
def weightAvg(avg1, avg2, n):
    # 进行第一次组合
    avg__combine = avg1.combine(avg2)
    # 从1开始循环到n-1，避免多次组合同一对象
    for i in range(1, n - 1):
        avg__combine = avg__combine.combine(avg2)
    return avg__combine


# 为BBA分配置信质量
frame_of_discernment = ['A', 'B', 'C']

# 创建两个BBA实例
bba1 = BBA(frame_of_discernment)
bba2 = BBA(frame_of_discernment)

bba1.assign_mass('A', 0.5)
bba1.assign_mass(frozenset(['B', 'C']), 0.5)

bba2.assign_mass('C', 0.5)
bba2.assign_mass(frozenset(['A', 'B']), 0.5)
# bba1.assign_mass('A', 0.5)
# bba1.assign_mass('B', 0.2)
# bba1.assign_mass('C', 0.3)
#
# bba2.assign_mass('A', 0)
# bba2.assign_mass('B', 0.9)
# bba2.assign_mass('C', 0.1)
# 求平均值
n = 2
average_bba = average(n, frame_of_discernment, bba1, bba2)
avg = weightAvg(average_bba, average_bba, n)
print("组合后的信念赋值：")
for hypothesis, mass in avg.m.items():
    print(f"Hypothesis {set(hypothesis)}: Mass = {mass}")

if __name__ == '__main__':
    value = 0.5 * (1 + 2 * math.pow(1 / 63, 2) - (3 / 63))
    print(np.sqrt(value))
