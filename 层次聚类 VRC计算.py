# 导入必要的库
import pandas as pd  # 用于数据处理和读取
import numpy as np   # 用于数值计算
from scipy.spatial.distance import cdist  # 用于计算距离
from scipy.cluster.hierarchy import linkage, fcluster  # 用于层次聚类
import matplotlib.pyplot as plt  # 用于绘图

# 设置 Matplotlib 的字体和样式，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据文件路径
data_path = r"Cluster_sample_data.csv"

# 读取数据文件，假设数据文件中只有两列（x 和 y），没有表头
data = pd.read_csv(data_path, header=None, names=['x', 'y'])

# 提取数据中的 x 和 y 列，并将其转换为 NumPy 数组
X = data[['x', 'y']].values

# 使用 Ward 方法进行层次聚类，计算聚类的连接矩阵 Z
Z = linkage(X, method='ward')

# 定义一系列用于划分聚类的阈值
dendrogram_cutoff_values = np.linspace(0, 15, 30)

# 初始化用于存储各种评估指标的列表
vrc_scores = []  # 方差比准则（Variance Ratio Criterion）
bss_scores = []  # 组间平方和（Between-Cluster Sum of Squares）
wss_scores = []  # 组内平方和（Within-Cluster Sum of Squares）
separation_scores = []  # 簇间分离度
compactness_scores = []  # 簇内紧密度

# 遍历每个阈值，计算对应的聚类效果评估指标
for dendrogram_cutoff in dendrogram_cutoff_values:
    # 根据当前阈值划分聚类
    labels = fcluster(Z, t=dendrogram_cutoff, criterion='distance')
    
    # 计算聚类的数量
    n_clusters = len(set(labels))
    
    # 如果聚类数量大于 1 且不是每个点都属于一个单独的簇，则进行评估
    if n_clusters > 1 and n_clusters != len(labels):
        # 提取每个簇的数据点
        clusters = [X[labels == i] for i in set(labels)]
        
        # 计算每个簇的中心点
        cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        
        # 计算组内平方和（WSS）
        wss = sum([np.sum((cluster - center)**2) for cluster, center in zip(clusters, cluster_centers)])
        wss_scores.append(wss)
        
        # 计算整体均值
        overall_mean = np.mean(X, axis=0)
        
        # 计算组间平方和（BSS）
        bss = sum([len(cluster) * np.sum((center - overall_mean)**2) for cluster, center in zip(clusters, cluster_centers)])
        bss_scores.append(bss)
        
        # 计算方差比准则（VRC）
        if n_clusters > 1:
            vrc = (bss / (n_clusters - 1)) / (wss / (len(X) - n_clusters))
        else:
            vrc = np.nan
        vrc_scores.append(vrc)
        
        # 计算簇间分离度
        cluster_centers = np.array(cluster_centers)
        if len(cluster_centers) > 1:
            avg_inter_cluster_distance = np.mean([cdist([cluster_centers[i]], [cluster_centers[j]], metric='euclidean').min()
                                                  for i in range(len(cluster_centers)) 
                                                  for j in range(i + 1, len(cluster_centers))])
            separation_scores.append(avg_inter_cluster_distance)
        else:
            separation_scores.append(np.nan)
        
        # 计算簇内紧密度
        compactness = [np.mean(cdist(cluster, [center])) for cluster, center in zip(clusters, cluster_centers)]
        compactness_scores.append(np.mean(compactness))
    else:
        # 如果聚类数量不符合要求，则将所有评估指标设置为 NaN
        vrc_scores.append(np.nan)
        bss_scores.append(np.nan)
        wss_scores.append(np.nan)
        separation_scores.append(np.nan)
        compactness_scores.append(np.nan)

# 绘制评估指标的图表
plt.figure(figsize=(15, 8))  # 设置图表大小

# 绘制方差比准则（VRC）的图表
plt.subplot(1, 3, 1)  # 创建子图 1
plt.plot(dendrogram_cutoff_values[:len(vrc_scores)], vrc_scores, marker='o')  # 绘制折线图
plt.title('层次聚类方差比准则（VRC）')  # 设置标题
plt.xlabel('ward方法阈值')  # 设置 x 轴标签
plt.ylabel('方差比准则（VRC）')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴范围

# 绘制簇内紧密度的图表
plt.subplot(1, 3, 2)  # 创建子图 2
plt.plot(dendrogram_cutoff_values[:len(compactness_scores)], compactness_scores, marker='o', color='orange')  # 绘制折线图
plt.title('层次聚类簇内紧密度')  # 设置标题
plt.xlabel('ward方法阈值')  # 设置 x 轴标签
plt.ylabel('平均簇内紧密度')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴范围

# 绘制簇间分离度的图表
plt.subplot(1, 3, 3)  # 创建子图 3
plt.plot(dendrogram_cutoff_values[:len(separation_scores)], separation_scores, marker='o', color='purple')  # 绘制折线图
plt.title('层次聚类簇间分离度')  # 设置标题
plt.xlabel('ward方法阈值')  # 设置 x 轴标签
plt.ylabel('平均簇间分离度')  # 设置 y 轴标签
plt.ylim(bottom=0)  # 设置 y 轴范围

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()