import numpy as np
from scipy.integrate import quad
from sklearn.datasets import load_iris
from itertools import chain, combinations


class IrisBPAAnalyzer:
    def __init__(self, train_samples=20, test_samples=10):
        self.iris = load_iris()
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.class_models = {}
        self.test_models = {}
        self.fod = ['S', 'E', 'V']  # Setosa, Versicolor, Virginica
        self.feature_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

    def _prepare_data(self, target_class='E'):
        """准备训练集和测试集"""
        data = self.iris.data
        target_idx = self.fod.index(target_class) if target_class in self.fod else 1  # 默认为E

        # 为每个类别创建训练模型
        for i, class_name in enumerate(self.iris.target_names):
            class_data = data[self.iris.target == i]
            train_indices = np.random.choice(len(class_data), self.train_samples, replace=False)
            self.class_models[self.fod[i]] = {
                'mean': np.mean(class_data[train_indices], axis=0),
                'std': np.std(class_data[train_indices], axis=0, ddof=1),
                'data': class_data[train_indices]
            }

        # 为指定类别创建测试模型（每个属性单独建模）
        target_data = data[self.iris.target == target_idx]
        remaining_indices = [i for i in range(len(target_data))
                             if i not in self.class_models[target_class]['data']]
        test_indices = np.random.choice(remaining_indices, self.test_samples, replace=False)

        for i in range(4):  # 四个属性
            self.test_models[self.feature_names[i]] = {
                'mean': np.mean(target_data[test_indices, i]),
                'std': np.std(target_data[test_indices, i], ddof=1),
                'data': target_data[test_indices, i]
            }

    def _gaussian_mf(self, x, mean, std):
        """高斯隶属度函数"""
        return np.exp(-0.5 * ((x - mean) / std) ** 2)

    def _calculate_integral_interval(self, *params):
        """计算积分区间"""
        bounds = []
        for mean, std in params:
            bounds.extend([mean + 3 * std, mean - 3 * std])
        return min(bounds), max(bounds)

    def _compute_intersection_area(self, mf_funcs, params):
        """计算曲线交集面积"""

        def min_mf(x):
            return min(mf(x) for mf in mf_funcs)

        a, b = self._calculate_integral_interval(*params)
        area, _ = quad(min_mf, a, b, limit=100)
        return area

    def _powerset(self, elements):
        """生成幂集（不包括空集）"""
        return list(chain.from_iterable(combinations(elements, r)
                                        for r in range(1, len(elements) + 1)))

    def calculate_bpa_for_feature(self, feature_idx):
        """
        计算单个特征的BPA
        :param feature_idx: 使用的特征索引（0-3）
        :return: 格式化后的BPA字典
        """
        feature_name = self.feature_names[feature_idx]
        test_mean = self.test_models[feature_name]['mean']
        test_std = self.test_models[feature_name]['std']
        test_area = np.sqrt(2 * np.pi) * test_std

        # 计算各命题支持度
        sup_dict = {}
        elements = self.fod.copy()
        propositions = self._powerset(elements)

        # 计算所有命题的支持度
        for prop in propositions:
            params = [(self.class_models[c]['mean'][feature_idx],
                       self.class_models[c]['std'][feature_idx]) for c in prop]
            params.append((test_mean, test_std))

            test_mf = lambda x: self._gaussian_mf(x, test_mean, test_std)
            fault_mfs = [lambda x, m=self.class_models[c]['mean'][feature_idx],
                                s=self.class_models[c]['std'][feature_idx]:
                         self._gaussian_mf(x, m, s) for c in prop]

            area = self._compute_intersection_area([test_mf, *fault_mfs], params)
            sup_dict[tuple(sorted(prop))] = area / test_area  # 使用排序后的元组作为键

        # 归一化
        total_sup = sum(sup_dict.values())
        bpa = {k: v / total_sup for k, v in sup_dict.items()}

        return bpa

    def generate_evidence(self, target_class='E', num_tests=1):
        """
        生成证据体数组
        :param target_class: 目标类别 ('S', 'E', 'V')
        :param num_tests: 要生成的证据体数量
        :return: 证据体数组，每个元素是一个包含4个BPA的列表
        """
        evidence_array = []
        for _ in range(num_tests):
            self._prepare_data(target_class)
            evidence = []
            for i in range(4):  # 四个特征
                bpa = self.calculate_bpa_for_feature(i)
                evidence.append(bpa)
            evidence_array.append(evidence)
        return evidence_array


# 使用示例
if __name__ == "__main__":
    analyzer = IrisBPAAnalyzer(train_samples=20, test_samples=10)

    # 生成单个证据体（包含4个BPA）
    single_evidence = analyzer.generate_evidence('E', 1)
    print("单个证据体（包含4个BPA）:")
    print(single_evidence)

    # 生成多个证据体（每个包含4个BPA）
    # num_tests = 5  # 生成5个证据体
    # multiple_evidence = analyzer.generate_evidence('E', num_tests)

    # print(f"\n{num_tests}个证据体:")
    # print(multiple_evidence)