# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 设置 Matplotlib 的字体和负号显示，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 避免负号显示异常

# 定义数据文件路径
file_path = r"Cluster_sample_data.csv"

# 读取数据，假设数据文件中没有表头，且只有两列（x 和 y）
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 定义要测试的聚类数范围（从 2 到 10）
k_values = range(2, 11)

# 初始化用于存储结果的列表
vrc_values = []       # 存储方差比准则（Variance Ratio Criterion，VRC）值
avg_compactness = []  # 存储平均簇内紧密度
avg_separation = []   # 存储平均簇间分离度

# 遍历不同的聚类数 k
for k in k_values:
    # 使用 KMeans 算法进行聚类，固定随机种子以保证结果可复现
    kmeans = KMeans(n_clusters=k, random_state=500)
    kmeans.fit(data)  # 对数据进行聚类

    # 获取聚类标签和聚类中心
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 计算 Within-Cluster Sum of Squares (WSS)：簇内平方和
    WSS = np.sum([np.sum(cdist(data[labels == i], [center]) ** 2) for i, center in enumerate(centers)])

    # 计算 Between-Cluster Sum of Squares (BSS)：簇间平方和
    overall_mean = np.mean(data, axis=0)  # 计算数据的全局均值
    BSS = np.sum([len(data[labels == i]) * np.sum((center - overall_mean) ** 2) for i, center in enumerate(centers)])

    # 计算方差比准则（VRC）
    vrc = (BSS / (k - 1)) / (WSS / (len(data) - k))
    vrc_values.append(vrc)  # 将当前 k 的 VRC 值存储到列表中

    # 计算平均簇内紧密度
    compactness = np.mean([np.mean(cdist(data[labels == i], [center])) for i, center in enumerate(centers)])
    avg_compactness.append(compactness)  # 将当前 k 的平均簇内紧密度存储到列表中

    # 计算平均簇间分离度
    separation = np.mean(cdist(centers, centers)[np.triu_indices(k, 1)])
    avg_separation.append(separation)  # 将当前 k 的平均簇间分离度存储到列表中

# 绘制结果图表
plt.figure(figsize=(18, 6))  # 设置整个图表的大小

# 绘制方差比准则（VRC）随聚类数 k 的变化
plt.subplot(1, 3, 1)  # 第一个子图
plt.plot(k_values, vrc_values, marker='o', linestyle='-', color='blue')
plt.title("K-means聚类方差比准则（VRC）")  # 图表标题
plt.xlabel("聚类数（k）")  # x 轴标签
plt.ylabel("方差比准则（VRC）")  # y 轴标签
plt.xticks(k_values)  # 设置 x 轴刻度
plt.grid(True)  # 添加网格线

# 绘制平均簇内紧密度随聚类数 k 的变化
plt.subplot(1, 3, 2)  # 第二个子图
plt.plot(k_values, avg_compactness, marker='o', linestyle='-', color='green')
plt.title("K-means聚类簇内紧密度")  # 图表标题
plt.xlabel("聚类数（k）")  # x 轴标签
plt.ylabel("平均簇内紧密度")  # y 轴标签
plt.xticks(k_values)  # 设置 x 轴刻度
plt.grid(True)  # 添加网格线

# 绘制平均簇间分离度随聚类数 k 的变化
plt.subplot(1, 3, 3)  # 第三个子图
plt.plot(k_values, avg_separation, marker='o', linestyle='-', color='orange')
plt.title("K-means聚类簇间分离度")  # 图表标题
plt.xlabel("聚类数（k）")  # x 轴标签
plt.ylabel("平均簇间分离度")  # y 轴标签
plt.xticks(k_values)  # 设置 x 轴刻度
plt.grid(True)  # 添加网格线

# 调整子图布局，确保图表美观
plt.tight_layout()

# 显示图表
plt.show()