import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sklearn.datasets import load_iris
from itertools import permutations
import pandas as pd


class IrisBPAAnalyzer:
    def __init__(self, train_samples=20, test_samples=10):
        self.iris = load_iris()
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.class_models = {}
        self.test_models = {}
        self.fod = ['S', 'E', 'V']  # Setosa, Versicolor, Virginica
        self.feature_names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
        self.feature_abbr = ['SL', 'SW', 'PL', 'PW']  # 特征缩写

    def _prepare_data(self, target_class='E'):
        """准备训练集和测试集"""
        data = self.iris.data
        self.target_class = target_class  # 保存目标类别
        target_idx = self.fod.index(target_class) if target_class in self.fod else 1

        # 为每个类别创建训练模型
        for i, class_name in enumerate(self.iris.target_names):
            class_data = data[self.iris.target == i]
            train_indices = np.random.choice(len(class_data), self.train_samples, replace=False)
            self.class_models[self.fod[i]] = {
                'mean': np.mean(class_data[train_indices], axis=0),
                'std': np.std(class_data[train_indices], axis=0, ddof=1),
                'data': class_data[train_indices]
            }

        # 为测试数据建模
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

    def generate_table1_data(self):
        """
        生成Table 1的数据（均值和标准差）
        """
        table_data = []

        # 添加表头
        headers = ['Category'] + [f'({abbr}_mean, {abbr}_std)' for abbr in self.feature_abbr]
        table_data.append(headers)

        # 添加训练模型数据
        for class_name in self.fod:
            row = [class_name]
            for i in range(4):
                mean_val = self.class_models[class_name]['mean'][i]
                std_val = self.class_models[class_name]['std'][i]
                row.append(f'({mean_val:.4f}, {std_val:.4f})')
            table_data.append(row)

        # 添加测试模型数据
        test_row = ['TestModel']
        for i, feature in enumerate(self.feature_names):
            mean_val = self.test_models[feature]['mean']
            std_val = self.test_models[feature]['std']
            test_row.append(f'({mean_val:.4f}, {std_val:.4f})')
        table_data.append(test_row)

        return table_data

    def plot_figure3(self):
        """
        绘制Figure 3的四个子图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        colors = {'S': 'red', 'E': 'green', 'V': 'blue', 'TestModel': 'black'}
        linestyles = {'S': '-', 'E': '--', 'V': ':', 'TestModel': '-.'}

        for i, (feature, abbr) in enumerate(zip(self.feature_names, self.feature_abbr)):
            ax = axes[i]

            # 生成x轴范围
            all_means = []
            all_stds = []

            # 收集所有均值和标准差来确定x轴范围
            for class_name in self.fod:
                all_means.append(self.class_models[class_name]['mean'][i])
                all_stds.append(self.class_models[class_name]['std'][i])
            all_means.append(self.test_models[feature]['mean'])
            all_stds.append(self.test_models[feature]['std'])

            x_min = min(all_means) - 3 * max(all_stds)
            x_max = max(all_means) + 3 * max(all_stds)
            x = np.linspace(x_min, x_max, 1000)

            # 绘制训练模型曲线
            for class_name in self.fod:
                mean = self.class_models[class_name]['mean'][i]
                std = self.class_models[class_name]['std'][i]
                y = self._gaussian_mf(x, mean, std)
                ax.plot(x, y, color=colors[class_name], linestyle=linestyles[class_name],
                        label=f'{class_name} (μ={mean:.2f}, σ={std:.2f})', linewidth=2)

            # 绘制测试模型曲线
            test_mean = self.test_models[feature]['mean']
            test_std = self.test_models[feature]['std']
            y_test = self._gaussian_mf(x, test_mean, test_std)
            ax.plot(x, y_test, color=colors['TestModel'], linestyle=linestyles['TestModel'],
                    label=f'TestModel (μ={test_mean:.2f}, σ={test_std:.2f})', linewidth=3)

            ax.set_xlabel(f'{feature} Value', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(f'({chr(97 + i)}) {feature} Distribution', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def print_table1(self):
        """
        打印Table 1格式的数据
        """
        table_data = self.generate_table1_data()

        print("Table 1. Mean value and standard deviation of training models (S, E, V) and test models.")
        print("-" * 90)

        # 打印表头
        header = table_data[0]
        print(f"{header[0]:<10} {header[1]:<20} {header[2]:<20} {header[3]:<20} {header[4]:<20}")
        print("-" * 90)

        # 打印数据行
        for row in table_data[1:]:
            print(f"{row[0]:<10} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20}")

        print("-" * 90)

    def statistical_analysis(self):
        """
        进行统计分析
        """
        print("\n=== Statistical Analysis ===")

        # 计算测试模型与各类别的距离
        for i, feature in enumerate(self.feature_names):
            print(f"\nFeature {feature}:")
            test_mean = self.test_models[feature]['mean']
            test_std = self.test_models[feature]['std']

            for class_name in self.fod:
                class_mean = self.class_models[class_name]['mean'][i]
                class_std = self.class_models[class_name]['std'][i]

                # 计算均值差异（标准化）
                mean_diff = abs(test_mean - class_mean)
                std_combined = np.sqrt(test_std ** 2 + class_std ** 2)
                z_score = mean_diff / std_combined if std_combined > 0 else 0

                print(f"  TestModel vs {class_name}: Z-score = {z_score:.3f}")

                if z_score < 1:
                    print(f"    → Close match with {class_name}")
                elif z_score < 2:
                    print(f"    → Moderate match with {class_name}")
                else:
                    print(f"    → Poor match with {class_name}")

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

    def _generate_propositions(self):
        """生成所有命题排列（保持顺序差异）"""
        elements = range(len(self.fod))
        propositions = []

        # 单元素命题
        propositions.extend([(i,) for i in elements])

        # 双元素排列（区分顺序）
        propositions.extend(permutations(elements, 2))

        # 三元素排列（区分顺序）
        propositions.extend(permutations(elements, 3))

        return propositions

    def _calculate_proposition_weight(self, prop, feature_idx):
        """基于RPS思想的动态权重计算（无固定参数）"""
        # 1. 计算基本交集面积
        test_mean = self.test_models[self.feature_names[feature_idx]]['mean']
        test_std = self.test_models[self.feature_names[feature_idx]]['std']

        # 测试样本的隶属函数
        test_mf = lambda x: self._gaussian_mf(x, test_mean, test_std)

        # 类别的隶属函数
        fault_mfs = []
        params = []
        for idx in prop:
            c = self.fod[idx]
            mean = self.class_models[c]['mean'][feature_idx]
            std = self.class_models[c]['std'][feature_idx]
            fault_mfs.append(lambda x, m=mean, s=std: self._gaussian_mf(x, m, s))
            params.append((mean, std))

        params.append((test_mean, test_std))
        area = self._compute_intersection_area([test_mf, *fault_mfs], params)

        # 2. RPS风格的位置权重计算
        target_idx = self.fod.index(self.target_class)
        if target_idx in prop:
            position = prop.index(target_idx)

            # 动态计算位置衰减系数（基于类间距离）
            target_mean = self.class_models[self.target_class]['mean'][feature_idx]
            other_means = [self.class_models[c]['mean'][feature_idx]
                           for c in self.fod if c != self.target_class]

            # 计算目标类与最近类的距离比率
            min_dist = min(abs(target_mean - m) for m in other_means)
            max_dist = max(abs(target_mean - m) for m in other_means)
            dist_ratio = min_dist / (max_dist + 1e-10)

            # 动态衰减系数（距离越小衰减越快）
            # 当目标类与其他类区别明显时（dist_ratio小），位置影响更大
            dynamic_decay = 1.0 + (1.0 - dist_ratio)  # 范围1.0-2.0
            position_weight = dynamic_decay ** (-position)  # 指数衰减
        else:
            position_weight = 1.0

        # 3. RPS风格的长度权重计算
        if len(prop) > 1:
            # 计算属性间的相对信息量
            feature_entropy = []
            for i in range(4):
                data = self.test_models[self.feature_names[i]]['data']
                hist = np.histogram(data, bins=10)[0]
                prob = hist / (np.sum(hist) + 1e-10)
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                feature_entropy.append(entropy)

            # 当前属性的相对信息量比率
            current_entropy = feature_entropy[feature_idx]
            avg_entropy = np.mean(feature_entropy)
            entropy_ratio = current_entropy / (avg_entropy + 1e-10)

            # 动态长度惩罚（信息量越高惩罚越小）
            # 当当前属性信息量高时，更信任多元素组合
            length_penalty = 1.0 + (1.0 - entropy_ratio)  # 范围1.0-2.0
            length_weight = 1.0 / (len(prop) ** length_penalty)
        else:
            length_weight = 1.0

        # 4. 组合权重计算
        return area * position_weight * length_weight

    def calculate_bpa_for_feature(self, feature_idx):
        """计算改进后的BPA（区分顺序）"""
        propositions = self._generate_propositions()
        sup_dict = {}

        for prop in propositions:
            sup_dict[prop] = self._calculate_proposition_weight(prop, feature_idx)

        # 归一化
        total_sup = sum(sup_dict.values())
        bpa = {k: v / total_sup for k, v in sup_dict.items()}

        return bpa

    def generate_evidence(self, target_class='E', num_tests=1):
        """生成证据体数组"""
        evidence_array = []
        for _ in range(num_tests):
            self._prepare_data(target_class)
            evidence = []
            for i in range(4):  # 四个特征
                bpa = self.calculate_bpa_for_feature(i)
                evidence.append(bpa)
            evidence_array.append(evidence)
        return evidence_array

    def print_evidence(self, evidence):
        """打印格式化后的证据体"""
        gen_rps = []
        for i, bpa in enumerate(evidence[0]):
            print(f"\n特征 {self.feature_names[i]} 的BPA:")
            list = {}
            for prop, value in sorted(bpa.items(), key=lambda x: (len(x[0]), x[0])):
                prop_labels = tuple(self.fod[idx] for idx in prop)
                print(f"{prop_labels}: {value}")
                list[prop_labels] = value
            gen_rps.append(list)
        return gen_rps

# 修改使用示例
if __name__ == "__main__":
    analyzer = IrisBPAAnalyzer(train_samples=20, test_samples=10)

    # 准备数据
    analyzer._prepare_data('E')

    # 打印Table 1
    analyzer.print_table1()

    # 绘制Figure 3
    fig = analyzer.plot_figure3()
    plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 进行统计分析
    analyzer.statistical_analysis()

    # 生成BPA证据
    print("\n" + "=" * 50)
    print("以E为主体的BPA分布:")
    evidence_E = analyzer.generate_evidence('E', 1)
    gen_rps = analyzer.print_evidence(evidence_E)