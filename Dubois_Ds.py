# Dubois 如果有交集就直接赋值给交集，如果没有交集就赋值给并集
# 如果证据源没有假设全集Θ，就不在相加
# 与Yager不同的是，冲突质量不在分配给全集，而是分配给交集

# 定义一个基本信念赋值 (BBA) 类
class DBPA:
    def __init__(self, frame_of_discernment):

        # 设置一个属性值
        self.frame = frame_of_discernment

        # 把传入的参数设置默认值0
        self.m = {frozenset([hypothesis]): 0 for hypothesis in self.frame}
        # self.m[frozenset([])] = 0  # 空集
        self.m[frozenset(['Θ'])] = 0  # 空集

    # 分配信念质量
    def assign_mass(self, hypothesis, mass):
        # 直接检查 hypothesis 是否为空集或在识别框架内
        # all(h in self.frame for h in hypothesis)：确保传入的 hypothesis 中的每个元素都在识别框架（self.frame）中
        # assert hypothesis == frozenset() or all(h in self.frame for h in hypothesis), "假设不在识别框架中"

        # 如果 hypothesis 是字符串，转换为单元素的 frozenset
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])

        # 否则，确保 hypothesis 是 frozenset 类型，如果不是则将其转换为 frozenset
        elif not isinstance(hypothesis, frozenset):
            hypothesis = frozenset(hypothesis)

        # 赋予信念质量
        self.m[hypothesis] = mass

    def dager_combine(self, other_ybpa):

        # 新建一个，用于存放计算好的值
        combine_dbpa = DBPA(self.frame)
        for h1 in self.m:
            for h2 in other_ybpa.m:

                # 求交集，并进行值得累加
                # 不对全集框架Θ进行计算
                if h1 != frozenset(['Θ']) and h2 != frozenset(['Θ']):
                    intersection = h1.intersection(h2)

                    # 如果交集为空，就求合集
                    if len(intersection) == 0:
                        union = frozenset(sorted(h1.union(h2)))
                        get = combine_dbpa.m.get(union)
                        if get is None:
                            combine_dbpa.m.setdefault(union, self.m[h1] * other_ybpa.m[h2])
                        else:
                            combine_dbpa.m[union] += self.m[h1] * other_ybpa.m[h2]
                    else:
                        combine_dbpa.m[intersection] += self.m[h1] * other_ybpa.m[h2]

        # 只相加m1 m2对全集得分配
        mass_Θ = self.m[frozenset(['Θ'])] + other_ybpa.m[frozenset(['Θ'])]
        combine_dbpa.m[frozenset(['Θ'])] = mass_Θ

        return combine_dbpa


# 示例：假设框架 {A, B}
frame_of_discernment = ['A', 'B', 'C']

# 创建两个BBA实例 ,
bba1 = DBPA(frame_of_discernment)
bba2 = DBPA(frame_of_discernment)

# 为BBA分配置信质量
bba1.assign_mass('A', 0.6)
bba1.assign_mass('B', 0.3)
bba1.assign_mass('Θ', 0.1)

bba2.assign_mass('A', 0.5)
bba2.assign_mass('B', 0.2)
bba2.assign_mass('C', 0.3)

# 组合两个BBA
combined_bba = bba1.dager_combine(bba2)

# 打印组合后的BBA
print("组合后的信念赋值：")
for hypothesis, mass in combined_bba.m.items():
    print(f"Hypothesis {set(hypothesis)}: Mass = {mass}")
