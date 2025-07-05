# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 设置 Matplotlib 的字体和符号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 数据路径
data_path = r"Cluster_sample_data.csv"

# 读取数据
data = pd.read_csv(data_path, header=None, names=['x', 'y'])  # 读取 CSV 文件，指定列名为 'x' 和 'y'
X = data[['x', 'y']].values  # 提取数据的 x 和 y 列，转换为 NumPy 数组

# 设置 DBSCAN 的 min_samples 参数
min_samples = 5  

# 使用 NearestNeighbors 计算每个点到其 min_samples 最近邻的距离
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)  # 拟合数据
distances, indices = neighbors_fit.kneighbors(X)  # 获取每个点到其最近的 min_samples 个点的距离和索引

# 提取每个点到第 min_samples 个最近邻的距离，并按升序排列
distances = np.sort(distances[:, min_samples - 1])

# 绘制 K 距离图，用于选择合适的 eps 值
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title(f'K距离图 (min_samples={min_samples})')  # 设置标题
plt.xlabel('升序排列点位次')  # 设置 x 轴标签
plt.ylabel(f'到第{min_samples}近邻点的距离')  # 设置 y 轴标签
plt.show()

# 设置 DBSCAN 的 eps 参数范围
eps = 0.18
eps_values = np.linspace(0.1, eps * 2, 50)  # 生成从 0.1 到 0.36 的 50 个等间距值

# 初始化用于存储评估指标的列表
vrc_scores = []  # 方差比准则（Variance Ratio Criterion）
separation_scores = []  # 簇间分离度
compactness_scores = []  # 簇内紧密度

# 遍历不同的 eps 值，评估 DBSCAN 聚类效果
for eps_val in eps_values:
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)  # 初始化 DBSCAN 模型
    labels = dbscan.fit_predict(X)  # 对数据进行聚类，获取每个点的标签

    # 获取非噪声点的簇标签（去除标签为 -1 的噪声点）
    unique_labels = set(labels) - {-1}  

    # 如果存在多个簇，则计算评估指标
    if len(unique_labels) > 1:
        clusters = [X[labels == label] for label in unique_labels]  # 提取每个簇的数据点
        n_clusters = len(clusters)  # 簇的数量

        # 计算簇内平方和（Within-Cluster Sum of Squares, WSS）
        WSS = sum(np.sum(cdist(cluster, [np.mean(cluster, axis=0)])**2) for cluster in clusters)

        # 计算簇间平方和（Between-Cluster Sum of Squares, BSS）
        overall_mean = np.mean(X, axis=0)  # 计算整体均值
        BSS = sum(len(cluster) * np.sum((np.mean(cluster, axis=0) - overall_mean)**2) for cluster in clusters)

        # 计算方差比准则（Variance Ratio Criterion, VRC）
        vrc_score = (BSS / (n_clusters - 1)) / (WSS / (len(X) - n_clusters))
        vrc_scores.append(vrc_score)

        # 计算簇间分离度
        if n_clusters > 1:
            centers = [np.mean(cluster, axis=0) for cluster in clusters]  # 计算每个簇的中心
            avg_inter_cluster_distance = np.mean(cdist(centers, centers)[np.triu_indices(n_clusters, k=1)])
            separation_scores.append(avg_inter_cluster_distance)
        else:
            separation_scores.append(np.nan)

        # 计算簇内紧密度
        compactness = [np.mean(cdist(cluster, [np.mean(cluster, axis=0)])) for cluster in clusters]
        compactness_scores.append(np.mean(compactness))
    else:
        # 如果只有一个簇或没有簇，将评估指标设置为 NaN
        vrc_scores.append(np.nan)
        separation_scores.append(np.nan)
        compactness_scores.append(np.nan)

# 绘制评估指标随 eps 变化的图
plt.figure(figsize=(12, 6))

# 绘制方差比准则（VRC）图
plt.subplot(1, 3, 1)
plt.plot(eps_values, vrc_scores, marker='o')
plt.title('DBSCAN聚类方差比准则（VRC）')  # 设置标题
plt.xlabel('eps')  # 设置 x 轴标签
plt.ylabel('方差比准则（VRC）')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴下限为 0

# 绘制簇内紧密度图
plt.subplot(1, 3, 2)
plt.plot(eps_values, separation_scores, marker='o', color='green')
plt.title('DBSCAN聚类簇内紧密度')  # 设置标题
plt.xlabel('eps')  # 设置 x 轴标签
plt.ylabel('平均簇内紧密度')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴下限为 0

# 绘制簇间分离度图
plt.subplot(1, 3, 3)
plt.plot(eps_values, compactness_scores, marker='o', color='red')
plt.title('DBSCAN聚类簇间分离度')  # 设置标题
plt.xlabel('eps')  # 设置 x 轴标签
plt.ylabel('平均簇间分离度')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴下限为 0

# 调整布局并显示图像
plt.tight_layout()
plt.show()