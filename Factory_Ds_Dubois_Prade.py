# 计算在Dubois组合规则下的权重因子
# 权重因子的分子为交集为空，但是并集为全集的子集，分母为冲突质量的总和

# 定义一个基本信念赋值 (BBA) 类
class DBPA:
    def __init__(self, frame_of_discernment):

        # 设置一个属性值
        self.frame = frame_of_discernment

        # 把传入的参数设置默认值0
        self.m = {frozenset([hypothesis]): 0 for hypothesis in self.frame}
        self.m[frozenset([])] = 0  # 空集
        self.m[frozenset(['Θ'])] = 0  # 空集

    # 分配信念质量
    def assign_mass(self, hypothesis, mass):
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])

        # 否则，确保 hypothesis 是 frozenset 类型，如果不是则将其转换为 frozenset
        elif not isinstance(hypothesis, frozenset):
            hypothesis = frozenset(hypothesis)

        # 赋予信念质量
        self.m[hypothesis] = mass

    # 计算权重因子
    # 简单来说就是把交集为空的求和，再把所有交集为空的求总和
    def dubois_factory_weight(self, other_dbpa):
        # 新建一个，用于存放计算好的值
        combine_dbpa = DBPA(self.frame)
        for h1 in self.m:
            for h2 in other_dbpa.m:

                # 求交集，并进行值得累加
                # 不对全集框架Θ进行计算
                if h1 != frozenset(['Θ']) and h2 != frozenset(['Θ']):
                    intersection = h1.intersection(h2)

                    # 如果交集为空，就求合集
                    if len(intersection) == 0:
                        combine_dbpa.m[intersection] += self.m[h1] * other_dbpa.m[h2]
                        union = frozenset(sorted(h1.union(h2)))
                        get = combine_dbpa.m.get(union)
                        if get is None:
                            combine_dbpa.m.setdefault(union, self.m[h1] * other_dbpa.m[h2])
                        else:
                            combine_dbpa.m[union] += self.m[h1] * other_dbpa.m[h2]
                    else:
                        combine_dbpa.m[intersection] += self.m[h1] * other_dbpa.m[h2]

        # 只相加m1 m2对全集得分配
        mass_Θ = self.m[frozenset(['Θ'])] + other_dbpa.m[frozenset(['Θ'])]
        combine_dbpa.m[frozenset(['Θ'])] = mass_Θ

        # m(∅)=K
        total_conflict = combine_dbpa.m[frozenset([])]

        # m(Θ)是所有交集为空的累加
        # 处理合并后的，计算权重因子
        factory_combined_bba = DBPA(self.frame)
        for h1 in combine_dbpa.m:
            get = factory_combined_bba.m.get(h1)
            factory_weight = combine_dbpa.m[h1] / total_conflict
            if get is None:
                factory_combined_bba.m.setdefault(h1, factory_weight)
            else:
                factory_combined_bba.m[h1] = factory_weight

        # 取出
        # for hypothesis in combine_dbpa.m:
        combine = {'factor': factory_combined_bba, 'combined_bba': combine_dbpa}
        return combine

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
        massVal = 0
        for h, mass in self.m.items():
            if h.issubset(hypothesis) and h != frozenset([]):
                massVal += mass
        return massVal

    # 计算似然度（plausibility）
    def plausibility(self, hypothesis):
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])
        else:
            hypothesis = frozenset(hypothesis)

            # 筛选出所有与输入假设 hypothesis 有交集的证据集合。
            # 简单来说只要有交集就相加
        return sum(mass for h, mass in self.m.items() if h.intersection(hypothesis))


frame_of_discernment = ['A', 'B', 'C']

# 创建两个BBA实例
bba1 = DBPA(frame_of_discernment)
bba2 = DBPA(frame_of_discernment)

e= 0.01

# 常量值
k =0.1

# 为BBA分配置信质量
bba1.assign_mass('A', e)
bba1.assign_mass('B', k)
bba1.assign_mass('C', 1 -k -e)

bba2.assign_mass('A', 1 -k -e)
bba2.assign_mass('B', k)
bba2.assign_mass('C', e)

# 打印组合后的权重因子
combine = bba1.dubois_factory_weight(bba2)
print("组合后的权重因子：")
for hypothesis, mass in combine['factor'].m.items():
    if len(hypothesis) > 1:
        print(f"w({set(hypothesis)},m1,m2): Mass = {mass:.3f}")

# 计算组合的bel和pl
bba_combine = combine['combined_bba']
# 计算信任度和似然度
belief_A = bba_combine.belief('A')
belief_B = bba_combine.belief('B')
belief_C = bba_combine.belief('C')
plausibility_A = bba_combine.plausibility('A')
plausibility_B = bba_combine.plausibility('B')
plausibility_C = bba_combine.plausibility('C')

print(f"\nBelief(A): {belief_A:.3f}")
print(f"Plausibility(A): {plausibility_A:.3f}")
print(f"\nBelief(B): {belief_B:.3f}")
print(f"Plausibility(B): {plausibility_B:.3f}")
print(f"\nBelief(C): {belief_C:.3f}")
print(f"Plausibility(C): {plausibility_C:.3f}")