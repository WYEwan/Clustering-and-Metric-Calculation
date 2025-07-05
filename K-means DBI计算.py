# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.cluster import KMeans  # 导入KMeans聚类算法
from sklearn.metrics import davies_bouldin_score  # 导入戴维斯堡丁指数（DBI）评估指标
from scipy.spatial.distance import cdist  # 导入计算距离的函数

# 设置matplotlib的字体和负号显示，确保中文和负号可以正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据文件路径
file_path = r"Cluster_sample_data.csv"

# 读取数据，假设数据文件中没有表头，指定列名为'x'和'y'
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 定义要测试的聚类数范围，从2到10
k_values = range(2, 11)

# 初始化用于存储评估指标的列表
dbi_values = []  # 存储戴维斯堡丁指数（DBI）值
avg_separation = []  # 存储平均簇间分离度
avg_compactness = []  # 存储平均簇内紧密度

# 遍历每个聚类数k
for k in k_values:
    # 创建KMeans模型，指定聚类数为k，随机种子为500以保证结果可复现
    kmeans = KMeans(n_clusters=k, random_state=500)
    # 对数据进行聚类拟合
    kmeans.fit(data)
    # 获取聚类标签
    labels = kmeans.labels_
    # 获取聚类中心
    centers = kmeans.cluster_centers_

    # 计算戴维斯堡丁指数（DBI），并将其添加到列表中
    dbi = davies_bouldin_score(data, labels)
    dbi_values.append(dbi)

    # 计算平均簇内紧密度
    # 对每个簇，计算簇内点到簇中心的平均距离，然后取所有簇的平均值
    compactness = np.mean([np.mean(cdist(data[labels == i], [center])) for i, center in enumerate(centers)])
    avg_compactness.append(compactness)

    # 计算平均簇间分离度
    # 计算所有簇中心之间的距离，取上三角矩阵的平均值（避免重复计算）
    separation = np.mean(cdist(centers, centers)[np.triu_indices(k, 1)])
    avg_separation.append(separation)

# 创建一个图形窗口，设置大小为18x6
plt.figure(figsize=(18, 6))

# 绘制戴维斯堡丁指数（DBI）随聚类数k的变化曲线
plt.subplot(1, 3, 1)  # 在1x3的子图网格中绘制第1个子图
plt.plot(k_values, dbi_values, marker='o', linestyle='-', color='blue')  # 绘制折线图
plt.title("K-means聚类戴维斯堡丁指数")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("戴维森堡丁指数（DBI）")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 绘制平均簇内紧密度随聚类数k的变化曲线
plt.subplot(1, 3, 2)  # 在1x3的子图网格中绘制第2个子图
plt.plot(k_values, avg_compactness, marker='o', linestyle='-', color='green')  # 绘制折线图
plt.title("K-means聚类簇内紧密度")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("平均簇内紧密度")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 绘制平均簇间分离度随聚类数k的变化曲线
plt.subplot(1, 3, 3)  # 在1x3的子图网格中绘制第3个子图
plt.plot(k_values, avg_separation, marker='o', linestyle='-', color='orange')  # 绘制折线图
plt.title("K-means聚类簇间分离度")  # 设置标题
plt.xlabel("聚类数（k）")  # 设置x轴标签
plt.ylabel("平均簇间分离度")  # 设置y轴标签
plt.xticks(k_values)  # 设置x轴刻度
plt.grid(True)  # 添加网格

# 调整子图布局，避免重叠
plt.tight_layout()

# 显示图形
plt.show()