# 在经典的Dempster下的通用权重计算
# Dempster 组合规则的经典公式相似，表明该规则在通用框架中可以被重新定义。

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

        # 如果 hypothesis 是字符串，转换为单元素的 frozenset
        if isinstance(hypothesis, str):
            hypothesis = frozenset([hypothesis])

        # 否则，确保 hypothesis 是 frozenset 类型，如果不是则将其转换为 frozenset
        elif not isinstance(hypothesis, frozenset):
            hypothesis = frozenset(hypothesis)

        # 赋予信念质量
        self.m[hypothesis] = mass

    def factory_weight(self, other_bba):

        # 用于存放求交集后的中间值
        combined_bba = BBA(self.frame)
        for h1 in self.m:
            for h2 in other_bba.m:
                # 求交集，用h1与h2交集
                # 虽然有空集相交，但是空集默认值为0，没有参与累加
                intersection = h1.intersection(h2)

                # 这一步得到公式里的m∩(A) m∩(空集)
                if len(intersection) <= 1:
                    combined_bba.m[intersection] += self.m[h1] * other_bba.m[h2]

        # 用于保存权重值
        factory_combined_bba = BBA(self.frame)
        for h1 in combined_bba.m:
            factory_combined_bba.m[h1] = combined_bba.m[h1]

        # m(∅)=K
        total_conflict = factory_combined_bba.m[frozenset([])]
        if total_conflict > 0:
            for hypothesis in factory_combined_bba.m:

                # combined_bba存储的就是重新规定的权重因子
                if hypothesis != frozenset([]):
                    factory_combined_bba.m[hypothesis] /= (1 - total_conflict)

        # combined_bba保存最终组合的值
        for hypothesis in combined_bba.m:
            if hypothesis != frozenset([]):
                combined_bba.m[hypothesis] += factory_combined_bba.m[hypothesis] * total_conflict
        combine = {'factor': factory_combined_bba, 'combined_bba': combined_bba}
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
bba1 = BBA(frame_of_discernment)
bba2 = BBA(frame_of_discernment)

# 为BBA分配置信质量
bba1.assign_mass('A', 0.1)
bba1.assign_mass('B', 0.1)
bba1.assign_mass('C', 0.8)

bba2.assign_mass('A', 0.8)
bba2.assign_mass('B', 0.1)
bba2.assign_mass('C', 0.1)

# 打印组合后的权重因子
combine = bba1.factory_weight(bba2)
print("组合后的权重因子：")
for hypothesis, mass in combine['factor'].m.items():
    if hypothesis != frozenset([]):
        print(f"w({set(hypothesis)},m1,m2): Mass = {mass:.2f}")

# 计算组合的bel和pl
bba_combine = combine['combined_bba']
# 计算信任度和似然度
belief_A = bba_combine.belief('A')
belief_B = bba_combine.belief('B')
plausibility_A = bba_combine.plausibility('A')
plausibility_B = bba_combine.plausibility('B')

print(f"\nBelief(A): {belief_A:.2f}")
print(f"Plausibility(A): {plausibility_A:.2f}")
print(f"\nBelief(B): {belief_B:.2f}")
print(f"Plausibility(B): {plausibility_B:.2f}")
