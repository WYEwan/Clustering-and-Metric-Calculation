# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

# 设置matplotlib的字体和显示负号的样式，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False    # 确保负号显示正常

# 定义数据文件路径
data_path = r"D:\通用文件夹\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"

# 读取数据，假设数据文件没有表头，只有两列（x和y）
data = pd.read_csv(data_path, header=None, names=['x', 'y'])

# 将数据转换为NumPy数组，方便后续处理
X = data[['x', 'y']].values

# 使用Ward方法进行层次聚类，计算链接矩阵Z
Z = linkage(X, method='ward')

# 定义一系列用于划分簇的阈值，用于评估不同阈值下的聚类效果
dendrogram_cutoff_values = np.linspace(0, 15, 30) 

# 初始化用于存储评估指标的列表
db_scores = []  # 戴维斯-布尔丁指数（Davies-Bouldin Score）
separation_scores = []  # 簇间分离度
compactness_scores = []  # 簇内紧密度

# 遍历每个阈值，计算对应的聚类效果
for dendrogram_cutoff in dendrogram_cutoff_values:
    # 使用fcluster函数根据当前阈值划分簇
    labels = fcluster(Z, t=dendrogram_cutoff, criterion='distance')
    
    # 计算簇的数量
    n_clusters = len(np.unique(labels)) 
    
    # 如果簇的数量小于2或大于数据点数减1，则跳过当前阈值
    if n_clusters < 2 or n_clusters > X.shape[0] - 1:
        continue
    
    # 计算戴维斯-布尔丁指数
    db_score = davies_bouldin_score(X, labels)
    db_scores.append(db_score)
    
    # 提取每个簇的数据点
    clusters = [X[labels == i] for i in range(1, n_clusters + 1)]
    
    # 计算每个簇的中心点
    cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
    
    # 如果簇的数量大于1，计算簇间分离度
    if len(cluster_centers) > 1:
        avg_inter_cluster_distance = np.mean([cdist([cluster_centers[i]], [cluster_centers[j]], metric='euclidean').min()
                                              for i in range(len(cluster_centers)) 
                                              for j in range(i + 1, len(cluster_centers))])
        separation_scores.append(avg_inter_cluster_distance)
    else:
        # 如果只有一个簇，簇间分离度无意义，记为NaN
        separation_scores.append(np.nan)
    
    # 计算簇内紧密度
    compactness = [np.mean(cdist(cluster, [center])) for cluster, center in zip(clusters, cluster_centers)]
    compactness_scores.append(np.mean(compactness))

# 绘制评估指标的图表
plt.figure(figsize=(12, 6))

# 绘制戴维斯-布尔丁指数随阈值变化的曲线
plt.subplot(1, 3, 1)
plt.plot(dendrogram_cutoff_values[:len(db_scores)], db_scores, marker='o')
plt.title('层次聚类戴维斯-布尔丁指数')  # 图表标题
plt.xlabel('Ward方法阈值')  # x轴标签
plt.ylabel('戴维斯-布尔丁指数')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制簇间分离度随阈值变化的曲线
plt.subplot(1, 3, 2)
plt.plot(dendrogram_cutoff_values[:len(separation_scores)], separation_scores, marker='o', color='green')
plt.title('层次聚类簇间分离度')  # 图表标题
plt.xlabel('Ward方法阈值')  # x轴标签
plt.ylabel('平均簇间分离度')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制簇内紧密度随阈值变化的曲线
plt.subplot(1, 3, 3)
plt.plot(dendrogram_cutoff_values[:len(compactness_scores)], compactness_scores, marker='o', color='red')
plt.title('层次聚类簇内紧密度')  # 图表标题
plt.xlabel('Ward方法阈值')  # x轴标签
plt.ylabel('平均簇内紧密度')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 调整布局并显示图表
plt.tight_layout()
plt.show()