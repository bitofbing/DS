import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from string import ascii_uppercase
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import itertools
from collections import defaultdict

# 生成所有按顺序选择的排列组合
def get_ordered_permutations(num_classes):
    result = []
    # 逐步增加元素数量
    for i in range(1, num_classes + 1):
        # 生成i个元素的全排列
        result.extend(itertools.permutations(range(i), i))
    return result

# 计算 weighted PMF
def calculate_weighted_pmf(weight_matrix, sorted_nmv):
    num_combinations, num_attributes = weight_matrix.shape
    num_classes = sorted_nmv.shape[0]  # 获取类的数量（classes）

    # 获取排列组合
    # all_combinations = get_ordered_permutations(num_classes)

    # 初始化 weighted_pmf 矩阵
    weighted_pmf = np.zeros_like(weight_matrix)

    # 记录当前组合数对应的起始位置
    current_row = 0

    # 遍历组合大小 i（从 1 到 num_classes）
    for i in range(1, num_classes + 1):
        num_permutations = len(list(itertools.permutations(range(i), i)))  # 当前大小的排列组合数量

        # 遍历每个属性 j
        for j in range(num_attributes):
            # 对于当前大小 i 的排列组合，使用 sorted_nmv[i-1, j]
            # 组合有几个类，就乘于多少行
            weighted_pmf[current_row:current_row + num_permutations, j] = (
                    weight_matrix[current_row:current_row + num_permutations, j] * sorted_nmv[i - 1, j]
            )

        # 更新起始行
        current_row += num_permutations

    return weighted_pmf

# 获取按顺序选择的排列组合
def gen_rps_fun(evidence):
    # 1. 加载Iris数据集并划分为训练集和测试集
    iris = load_iris()
    X = iris.data  # 四个属性
    y = iris.target  # 三个类 (0, 1, 2)
    num_classes = len(np.unique(iris.target))
    num_attributes = iris.data.shape[1]
    # 将数据集划分为训练集和测试集，乱序
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True, random_state=42)
    # 2. 计算每个类中每个属性的 mean value and standard deviation (无偏估计)
    mean_std_by_class = []
    for class_label in np.unique(y_train):
        X_class = X_train[y_train == class_label]
        mean_std = [(np.mean(X_class[:, i]), np.std(X_class[:, i], ddof=1)) for i in range(X_class.shape[1])]
        mean_std_by_class.append(mean_std)

    mean_std_by_class = np.array(mean_std_by_class)
    # print("每个类中每个属性的均值和标准差:\n", mean_std_by_class)
    # print("Shape of mean_std_by_class:\n", mean_std_by_class.shape)
    # 3. 为每个类和每个属性建立高斯分布函数，并对测试集中随机选取的一个样本进行预测

    # 保存下(3,4)个Gaussian distribution函数
    # 创建一个(3,4)的函数数组，用来存储每个类中每个属性的高斯分布函数
    # 对于每个类中的每个属性，计算该样本在该属性下的高斯分布的概率密度值。
    gaussian_functions = np.empty((3, 4), dtype=object)

    # 初始化并保存高斯分布函数
    for class_label in range(num_classes):
        for i in range(num_attributes):  # 四个属性
            mean, std = mean_std_by_class[class_label, i]
            # 保存高斯分布函数
            gaussian_functions[class_label, i] = norm(loc=mean, scale=std)

    # 随机选择一个测试集中的样本
    test_sample = X_test[np.random.randint(0, len(X_test))]

    # 计算该测试样本在每个类中每个属性的高斯分布结果
    gaussian_results = []
    for class_label in range(num_classes):
        class_results = []
        for i in range(num_attributes):  # 四个属性
            # 调用保存的高斯分布函数，计算概率密度值
            # pdf是SciPy的norm对象的方法，用于计算给定值在该正态分布下的概率密度值。
            pdf_value = gaussian_functions[class_label, i].pdf(test_sample[i])
            class_results.append(pdf_value)
        gaussian_results.append(class_results)

    gaussian_results = np.array(gaussian_results)
    # print("\n测试集中选取的样本:", test_sample)
    # print("\n每个类中每个属性的高斯分布函数值:\n", gaussian_results)
    column_sums = np.sum(gaussian_results, axis=0)
    normalized_results = gaussian_results / column_sums
    # print("\n每个属性针对所有类的归一化后的MV (归一化后的高斯分布值):\n", normalized_results)
    # 对归一化后的MV（normalized membership vector）进行降序排序，并保留原始顺序的索引
    sorted_indices = np.argsort(-normalized_results, axis=0)  # 降序排序，使用负号实现降序
    sorted_nmv = np.take_along_axis(normalized_results, sorted_indices, axis=0)  # 按照索引排序后的值
    sorted_gaussian_functions = np.take_along_axis(gaussian_functions, sorted_indices, axis=0)  # 按照索引排序后的GDM

    # 打印结果
    # print("\n归一化后的MV降序排序的结果:\n", sorted_nmv)
    # print("\n每个元素排序前的原始类索引:\n", sorted_indices)
    x_mean_ord = np.empty((3, 4))
    std_ord = np.empty((3, 4))

    # mean_std_by_class 的 shape 是 (3, 4, 2)，索引 [class, attribute, 0] 获取均值，索引 [class, attribute, 1] 获取标准差
    for attr_idx in range(num_attributes):  # 对每个属性进行操作
        for class_idx in range(num_classes):  # 对每个类进行操作
            sorted_class_idx = sorted_indices[class_idx, attr_idx]  # 获取排序后的类索引
            x_mean_ord[class_idx, attr_idx] = mean_std_by_class[sorted_class_idx, attr_idx, 0]  # 获取排序后的均值
            std_ord[class_idx, attr_idx] = mean_std_by_class[sorted_class_idx, attr_idx, 1]  # 获取排序后的标准差

    # print("\n排序后的 x_mean_ord:\n", x_mean_ord)
    # print("\n排序后的 std_ord:\n", std_ord)
    supporting_degree = np.exp(-np.abs(test_sample - x_mean_ord))

    # print("\nSupporting degree (支持度):\n", supporting_degree)
    all_combinations = get_ordered_permutations(num_classes)
    # 初始化权重矩阵 weight_matrix
    num_combinations = len(all_combinations)  # 所有按顺序排列组合的数量 (应该是9)
    weight_matrix = np.zeros((num_combinations, num_attributes))  # (9, 4)

    # 对每个属性计算权重,也就是每个属性下的权重
    for attr_idx in range(num_attributes):
        # 取第一列，第一列对应属性1
        s = supporting_degree[:, attr_idx]  # 取出该属性对应的支持度 (3,)

        # 遍历每个组合，计算 w(i1...iu...iq)
        for comb_idx, combination in enumerate(all_combinations):
            q = len(combination)  # 该组合的长度
            weight = 1.0  # 初始化权重

            # 根据公式 (19) 计算权重
            for u in range(q):
                i_u = combination[u]  # 当前排列项 i_u
                numerator = s[i_u]  # 分子支持度
                denominator_sum = np.sum(s[list(combination[u:])])  # 分母，从 u 到 q 的支持度和
                weight *= numerator / denominator_sum  # 按公式累乘

            # 将计算好的权重保存到 weight_matrix
            weight_matrix[comb_idx, attr_idx] = weight

    # 输出权重矩阵
    # print("\n权重矩阵 (Weight matrix):\n", weight_matrix)
    weighted_pmf = calculate_weighted_pmf(weight_matrix, sorted_nmv)
    RPS_w = []
    for j in range(num_attributes):
        RPS_w_j = set()

        thetas = sorted_indices[:, j]
        weighted_pmf_j = weighted_pmf[:, j]

        for idx, combination in enumerate(all_combinations):
            A = thetas[list(combination)]
            M_A = weighted_pmf_j[idx]
            A = tuple((A))
            RPS_w_j.add((A, M_A))

        RPS_w.append(RPS_w_j)
    return RPS_w