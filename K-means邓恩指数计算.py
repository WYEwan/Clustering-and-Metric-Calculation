# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.cluster import KMeans  # 用于K-means聚类
from scipy.spatial.distance import cdist  # 用于计算距离

# 设置Matplotlib的字体和负号显示，确保中文和负号可以正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号可以正常显示

# 指定数据文件路径
file_path = r"Cluster_sample_data.csv"

# 读取数据，假设数据文件中没有表头，列名为 'x' 和 'y'
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 定义要测试的聚类数范围，从2到10
k_values = range(2, 11)

# 初始化用于存储结果的列表
dunn_values = []  # 存储邓恩指数
max_compactness = []  # 存储簇内最大紧密度
min_separation = []  # 存储簇间最小分离度

# 遍历每个聚类数 k
for k in k_values:
    # 使用KMeans进行聚类，指定聚类数为k，随机种子为500以确保结果可复现
    kmeans = KMeans(n_clusters=k, random_state=500)
    kmeans.fit(data)  # 拟合数据

    # 获取聚类标签和聚类中心
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 计算簇内最大紧密度
    # 对每个簇，计算簇内所有点到簇中心的最大距离
    compactness = np.max([np.max(cdist(data[labels == i], [center])) for i, center in enumerate(centers)])
    max_compactness.append(compactness)  # 将结果存储到列表中

    # 计算簇间最小分离度
    # 计算所有聚类中心之间的距离矩阵
    inter_cluster_distances = cdist(centers, centers)
    # 将对角线元素（即自身到自身的距离）设置为无穷大，避免干扰
    np.fill_diagonal(inter_cluster_distances, np.inf)
    # 找到最小的簇间距离
    separation = np.min(inter_cluster_distances)
    min_separation.append(separation)  # 将结果存储到列表中

    # 计算邓恩指数
    dunn_index = separation / compactness
    dunn_values.append(dunn_index)  # 将结果存储到列表中

# 绘制结果
plt.figure(figsize=(18, 6))  # 设置图形大小

# 绘制邓恩指数随聚类数k的变化
plt.subplot(1, 3, 1)  # 创建子图1
plt.plot(k_values, dunn_values, marker='o', linestyle='-', color='blue')  # 绘制折线图
plt.title("K-means聚类邓恩指数")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("邓恩指数")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 绘制簇内最大紧密度随聚类数k的变化
plt.subplot(1, 3, 2)  # 创建子图2
plt.plot(k_values, max_compactness, marker='o', linestyle='-', color='green')  # 绘制折线图
plt.title("K-means聚类簇内最大紧密度")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("簇内最大紧密度")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 绘制簇间最小分离度随聚类数k的变化
plt.subplot(1, 3, 3)  # 创建子图3
plt.plot(k_values, min_separation, marker='o', linestyle='-', color='orange')  # 绘制折线图
plt.title("K-means聚类簇间最小分离度")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("簇间最小分离度")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 自动调整子图布局
plt.tight_layout()

# 显示图形
plt.show()