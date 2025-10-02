import itertools
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict


class NewOrthogonalSumRPS:
    """
    Implementation of the New Orthogonal Sum in RPS based on Definitions 3.1-3.6.
    """

    def __init__(self, fod: List[Any], fixed_order: List[Any]):
        self.theta = fod
        self.n = len(fod)
        self.fixed_order = fixed_order
        self.symbolic_weights = [f'ω{i + 1}' for i in range(self.n)]
        self.pes = self._generate_pes()

    def _generate_pes(self) -> List[Tuple[Any, ...]]:
        pes = [()]
        for r in range(1, self.n + 1):
            for subset in itertools.combinations(self.theta, r):
                for perm in itertools.permutations(subset):
                    pes.append(perm)
        return pes

    def get_order_code(self, event: Tuple[Any, ...]) -> Tuple[str, ...]:
        order_code = ['0'] * self.n
        if not event:
            return tuple(order_code)
        for i, element_in_s in enumerate(self.fixed_order):
            if element_in_s in event:
                rank_in_A = event.index(element_in_s) + 1
                order_code[i] = self.symbolic_weights[rank_in_A - 1]
        return tuple(order_code)

    def order_code_intersection(self, v_a: Tuple[str, ...], v_b: Tuple[str, ...]) -> Tuple[str, ...]:
        def symbolic_min(a: str, b: str) -> str:
            if a == '0' or b == '0':
                return '0'
            if a == b:
                return a
            a_idx = self.symbolic_weights.index(a)
            b_idx = self.symbolic_weights.index(b)
            return self.symbolic_weights[max(a_idx, b_idx)]

        return tuple(symbolic_min(a, b) for a, b in zip(v_a, v_b))

    def decompose_order_code(self, v: Tuple[str, ...]) -> List[Tuple[str, ...]]:
        non_zero_elements = []
        for i, symbol in enumerate(v):
            if symbol != '0':
                non_zero_elements.append((i, symbol))

        if not non_zero_elements:
            return [v]

        decomposed_codes = []
        for keep_mask in range(1, 1 << len(non_zero_elements)):
            new_code = list(v)
            valid = True
            kept_symbols = set()
            for j, (pos, symbol) in enumerate(non_zero_elements):
                if keep_mask & (1 << j):
                    if symbol in kept_symbols:
                        valid = False
                        break
                    kept_symbols.add(symbol)
                else:
                    new_code[pos] = '0'
            if valid:
                decomposed_codes.append(tuple(new_code))
        return decomposed_codes

    def map_to_pes(self, v: Tuple[str, ...]) -> List[Tuple[Any, ...]]:
        required_elements = []
        element_ranks = {}
        for i, symbol in enumerate(v):
            if symbol != '0':
                element = self.fixed_order[i]
                required_elements.append(element)
                rank = self.symbolic_weights.index(symbol) + 1
                element_ranks[element] = rank

        if not required_elements:
            return [()]

        candidate_events = list(itertools.permutations(required_elements))
        valid_events = []
        for event in candidate_events:
            valid = True
            for element, required_rank in element_ranks.items():
                if element in event:
                    actual_rank = event.index(element) + 1
                    if actual_rank > required_rank:
                        valid = False
                        break
            if valid:
                valid_events.append(event)
        return valid_events

    def inverse_order_code_mapping(self, v: Tuple[str, ...]) -> List[Tuple[Any, ...]]:
        decomposed_codes = self.decompose_order_code(v)
        result_events = []
        for code in decomposed_codes:
            pes_events = self.map_to_pes(code)
            result_events.extend(pes_events)
        return list(set(result_events))

    def new_orthogonal_sum(self, mu1: Dict[Tuple[Any, ...], float],
                           mu2: Dict[Tuple[Any, ...], float]) -> Dict[Tuple[Any, ...], float]:
        order_codes = {}
        all_events = set(mu1.keys()) | set(mu2.keys())
        for event in all_events:
            order_codes[event] = self.get_order_code(event)

        # Calculate conflict coefficient K
        K = 0.0
        for event_b, mass_b in mu1.items():
            v_b = order_codes[event_b]
            for event_c, mass_c in mu2.items():
                v_c = order_codes[event_c]
                v_intersection = self.order_code_intersection(v_b, v_c)
                if any(x != '0' for x in v_intersection):
                    K += mass_b * mass_c

        result_mass = defaultdict(float)
        for event_b, mass_b in mu1.items():
            v_b = order_codes[event_b]
            for event_c, mass_c in mu2.items():
                v_c = order_codes[event_c]
                v_intersection = self.order_code_intersection(v_b, v_c)
                candidate_events = self.inverse_order_code_mapping(v_intersection)
                if not candidate_events:
                    continue
                mass_contribution = mass_b * mass_c / len(candidate_events)
                for candidate in candidate_events:
                    result_mass[candidate] += mass_contribution

        # Normalize
        total_mass = sum(result_mass.values())
        if total_mass > 0:
            normalized_result = {}
            for event, mass in result_mass.items():
                normalized_result[event] = mass / total_mass
            return normalized_result
        else:
            return {event: 0.0 for event in result_mass}


def convert_fuzzy_set_to_pes_mass(fuzzy_set: Dict[str, float], fod: List[str]) -> Dict[Tuple[str, ...], float]:
    """
    Convert fuzzy set representation to PES mass function representation.
    In fuzzy sets, each element is treated as a singleton event.
    """
    mass_function = {}
    for element, mass in fuzzy_set.items():
        # Each fuzzy element corresponds to a singleton permutation event
        mass_function[(element,)] = mass
    return mass_function


# 根据您提供的图片数据进行融合
if __name__ == "__main__":
    print("=== 模糊集融合示例 ===")

    # 定义辨识框架和固定顺序
    fod = ['θ1', 'θ2', 'θ3']
    fixed_order = ['θ1', 'θ2', 'θ3']

    # 定义模糊集 μ₁ 和 μ₂（根据图片数据）
    fuzzy_mu1 = {'θ1': 0.4, 'θ2': 0.3, 'θ3': 0.3}
    fuzzy_mu2 = {'θ1': 0.5, 'θ2': 0.1, 'θ3': 0.4}

    print("输入模糊集:")
    print(f"μ₁ = {fuzzy_mu1}")
    print(f"μ₂ = {fuzzy_mu2}")

    # 将模糊集转换为PES质量函数
    mu1_pes = convert_fuzzy_set_to_pes_mass(fuzzy_mu1, fod)
    mu2_pes = convert_fuzzy_set_to_pes_mass(fuzzy_mu2, fod)

    # 确保质量函数归一化
    total_mu1 = sum(mu1_pes.values())
    total_mu2 = sum(mu2_pes.values())
    mu1_pes = {k: v / total_mu1 for k, v in mu1_pes.items()}
    mu2_pes = {k: v / total_mu2 for k, v in mu2_pes.items()}

    print("\n转换为PES质量函数:")
    print(f"μ₁ (PES): {mu1_pes}")
    print(f"μ₂ (PES): {mu2_pes}")

    # 初始化RPS融合器
    combiner = NewOrthogonalSumRPS(fod, fixed_order)

    # 执行新正交和融合
    result = combiner.new_orthogonal_sum(mu1_pes, mu2_pes)

    # 输出融合结果
    print("\n=== 融合结果 ===")
    print("新正交和融合结果 μ₁ ⊕ μ₂:")

    # 按事件长度和字母顺序排序输出
    sorted_events = sorted(result.items(), key=lambda x: (len(x[0]), x[0]))
    total_mass = 0.0

    for event, mass in sorted_events:
        event_str = "∅" if len(event) == 0 else "(" + ", ".join(event) + ")"
        print(f"{event_str}: {mass:.6f}")
        total_mass += mass

    print(f"总质量: {total_mass:.6f}")

    # 计算各元素的置信度（将包含该元素的所有事件的质量相加）
    print("\n各元素的置信度:")
    for element in fod:
        belief = sum(mass for event, mass in result.items() if element in event)
        print(f"Bel({element}): {belief:.6f}")

    # 计算各元素的似然度（将与该元素不冲突的所有事件的质量相加）
    print("\n各元素的似然度:")
    for element in fod:
        plausibility = sum(mass for event, mass in result.items()
                           if not any(e != element and e in event for e in fod))
        print(f"Pl({element}): {plausibility:.6f}")