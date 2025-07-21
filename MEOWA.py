import math

import numpy as np
from scipy.optimize import minimize


# 定义一个基本信念赋值 (BBA) 类
class RPS:
    def __init__(self):
        # 设置一个属性值
        # self.frame = frame_of_discernment

        # 把传入的参数设置默认值0
        self.m = {tuple([]): 0}

    # 分配信念质量
    def assign_mass(self, hypothesis, mass):
        # 直接检查 hypothesis 是否为空集或在识别框架内

        # 如果 hypothesis 是字符串，转换为单元素的 frozenset
        if isinstance(hypothesis, str):
            hypothesis = tuple([hypothesis])

        # 否则，确保 hypothesis 是 frozenset 类型，如果不是则将其转换为 frozenset
        elif not isinstance(hypothesis, tuple):
            hypothesis = hypothesis

        # 赋予信念质量
        self.m[hypothesis] = mass

    # 证据组合LOS以左边的为主
    def LOS_ROS(self, other_bba):

        # 新建一个，用于存放计算好的值
        combined_bba = RPS()
        for h1 in self.m:
            for h2 in other_bba.m:
                # 求交集，以h1的为主，保留h1中的元素顺序
                intersection = tuple([x for x in h1 if x in h2])

                # 把两个RPS的交集累加
                get = combined_bba.m.get(intersection)
                if get is None:
                    combined_bba.m.setdefault(intersection, self.m[h1] * other_bba.m[h2])
                else:
                    combined_bba.m[intersection] += self.m[h1] * other_bba.m[h2]

        # 归一化，处理冲突，K就是所有交集为空的总和
        total_conflict = combined_bba.m[tuple([])]
        if total_conflict > 0:
            for hypothesis in combined_bba.m:

                # 只要不是空集就处理一下
                if hypothesis != []:
                    combined_bba.m[hypothesis] /= (1 - total_conflict)
        return combined_bba


# 定义熵函数（避免 log(0) 的情况）
def entropy(w):
    # 添加一个小常数避免 log(0)
    w = np.clip(w, 1e-10, 1.0)  # 将权重限制在 [1e-10, 1.0] 范围内
    return -np.sum(w * np.log(w))


# 定义目标函数（最大化熵 = 最小化负熵）
def objective(w):
    return -entropy(w)


# 定义约束条件
def constraint_sum(w):
    return np.sum(w) - 1  # 权重和为1


def constraint_orness(w, n, alpha):
    return (1 / (n - 1)) * np.sum((n - np.arange(1, n + 1)) * w) - alpha  # 顺序度量等于alpha


# 生成 MEOWA 权重
def generate_meowa_weights(n, alpha):
    # 初始权重猜测（均匀分布）
    w0 = np.ones(n) / n

    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},  # 权重和为1
        {'type': 'eq', 'fun': lambda w: constraint_orness(w, n, alpha)}  # 顺序度量等于alpha
    ]

    # 权重非负的边界条件
    bounds = [(0, 1) for _ in range(n)]

    # 求解优化问题
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("优化失败: " + result.message)


def MEP(frame_of_discernment, perm, weights_2, weights_3):
    # 遍历元素框架A B C
    MEP_perm = {}
    for e in frame_of_discernment:
        # 遍历融合后的信念
        mass_sum = 0.00
        for hypothesis, mass in perm.m.items():
            # 判断框架元素
            if e in hypothesis:
                # 判断e在hypothesis中的位置
                index = hypothesis.index(e)
                len1 = len(hypothesis)
                if len1 == 2:
                    mass_sum += weights_2[index] * mass
                elif len1 == 3:
                    mass_sum += weights_3[index] * mass
                else: mass_sum += mass
        MEP_perm[e] = mass_sum
    return MEP_perm


F3 = tuple(['A', 'B'])
F3_1 = tuple(['A', 'B'])
F3_2 = tuple(['B', 'A'])
F7 = tuple(['A', 'B', 'C'])
F7_1 = tuple(['A', 'B', 'C'])
F7_6 = tuple(['C', 'B', 'A'])

# 为BBA分配置信质量
frame_of_discernment = ['A', 'B', 'C']

# 创建两个BBA实例
rps1 = RPS()
rps1.assign_mass(F3_1, 0.5)
rps1.assign_mass(F7_6, 0.5)

rps2 = RPS()
rps2.assign_mass(F3_2, 0.5)
rps2.assign_mass(F7_1, 0.5)

# 左交集
los = rps1.LOS_ROS(rps2)
print("组合后的左信念赋值：")
for hypothesis, mass in los.m.items():
    print(f"Hypothesis {hypothesis}: Mass = {mass}")
# 右交集
ros = rps2.LOS_ROS(rps1)
print("组合后的右信念赋值：")
for hypothesis, mass in ros.m.items():
    print(f"Hypothesis {hypothesis}: Mass = {mass}")

# 生成含有两个元素的MEOWA算子
n = 2
weights_2 = generate_meowa_weights(n, 0.7)
print("生成的 MEOWA 权重:", weights_2)

# 验证权重和与顺序度量
print("权重和:", np.sum(weights_2))
print("顺序度量:", (1 / (n - 1)) * np.sum((n - np.arange(1, n + 1)) * weights_2))

# 生成含有三个元素的MEOWA算子
# 虽然组合不同，但是可以乘对应位置的权重
n = 3
weights_3 = generate_meowa_weights(n, 0.7)
print("生成的 MEOWA 权重:", weights_3)

# 验证权重和与顺序度量
print("权重和:", np.sum(weights_3))
print("顺序度量:", (1 / (n - 1)) * np.sum((n - np.arange(1, n + 1)) * weights_3))

mep_1 = MEP(frame_of_discernment, los, weights_2, weights_3)
# 遍历键值对
for key, value in mep_1.items():
    print(f"mep_1{key}: {value}")

mep_2 = MEP(frame_of_discernment, ros, weights_2, weights_3)
# 遍历键值对
for key, value in mep_2.items():
    print(f"mep_2{key}: {value}")