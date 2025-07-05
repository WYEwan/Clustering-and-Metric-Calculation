# 导入必要的库
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist  # 用于计算距离矩阵
from scipy.cluster.hierarchy import linkage, fcluster  # 用于层次聚类
import matplotlib.pyplot as plt  # 用于绘图

# 设置 Matplotlib 的字体和显示负号的配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

# 定义数据文件路径
data_path = r"D:\通用文件夹\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"

# 读取数据文件，假设数据文件中没有表头，且只有两列（x 和 y 坐标）
data = pd.read_csv(data_path, header=None, names=['x', 'y'])

# 将数据转换为 NumPy 数组，方便后续计算
X = data[['x', 'y']].values

# 使用层次聚类的 Ward 方法计算聚类的连接矩阵 Z
Z = linkage(X, method='ward')

# 定义一系列的树状图截断值，用于后续分析不同阈值下的聚类效果
dendrogram_cutoff_values = np.linspace(0, 15, 30)

# 初始化用于存储邓恩指数、分离度和紧密度的列表
dunn_indices = []
separation_scores = []
compactness_scores = []

# 遍历每个树状图截断值
for dendrogram_cutoff in dendrogram_cutoff_values:
    # 根据当前截断值对数据进行聚类，生成聚类标签
    labels = fcluster(Z, t=dendrogram_cutoff, criterion='distance')
    
    # 计算聚类的数量
    n_clusters = len(set(labels))
    
    # 如果聚类数量大于1且小于数据点总数，则进行后续计算
    if n_clusters > 1 and n_clusters != len(labels):
        # 根据聚类标签将数据点分组
        clusters = [X[labels == i] for i in range(1, n_clusters + 1)]
        
        # 如果聚类数量大于1，则计算分离度
        if len(clusters) > 1:
            min_inter_cluster_distance = np.inf  # 初始化最小簇间距离为无穷大
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算两个簇之间的最小距离
                    distance = np.min(cdist(clusters[i], clusters[j]))
                    min_inter_cluster_distance = min(min_inter_cluster_distance, distance)
            separation_scores.append(min_inter_cluster_distance)  # 存储当前分离度
        else:
            separation_scores.append(np.nan)  # 如果只有一个簇，则分离度为NaN
        
        # 初始化用于存储每个簇的紧密度的列表
        compactness = []
        for cluster in clusters:
            # 计算簇内距离矩阵
            intra_cluster_distances = cdist(cluster, cluster)
            np.fill_diagonal(intra_cluster_distances, np.nan)  # 将对角线元素（自身距离）设为NaN
            # 计算簇内的最大距离
            max_intra_cluster_distance = np.nanmax(intra_cluster_distances)
            compactness.append(max_intra_cluster_distance)
        
        # 计算平均紧密度
        compactness_scores.append(np.mean(compactness))
        
        # 如果紧密度大于0，则计算邓恩指数
        if compactness_scores[-1] > 0:
            dunn_index = separation_scores[-1] / compactness_scores[-1]
            dunn_indices.append(dunn_index)
        else:
            dunn_indices.append(np.nan)  # 如果紧密度为0，则邓恩指数为NaN
    else:
        # 如果聚类数量不符合要求，则将相关指标设为NaN
        dunn_indices.append(np.nan)
        separation_scores.append(np.nan)
        compactness_scores.append(np.nan)

# 绘制结果图
plt.figure(figsize=(12, 6))

# 绘制邓恩指数随截断值变化的曲线
plt.subplot(1, 3, 1)
plt.plot(dendrogram_cutoff_values[:len(dunn_indices)], dunn_indices, marker='o')
plt.title('层次聚类邓恩指数')  # 图标题
plt.xlabel('ward方法阈值')  # x轴标签
plt.ylabel('邓恩指数')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制分离度随截断值变化的曲线
plt.subplot(1, 3, 2)
plt.plot(dendrogram_cutoff_values[:len(separation_scores)], separation_scores, marker='o', color='green')
plt.title('层次聚类簇间最小分离度')  # 图标题
plt.xlabel('ward方法阈值')  # x轴标签
plt.ylabel('簇间最小分离度')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制紧密度随截断值变化的曲线
plt.subplot(1, 3, 3)
plt.plot(dendrogram_cutoff_values[:len(compactness_scores)], compactness_scores, marker='o', color='red')
plt.title('层次聚类簇内最大紧密度')  # 图标题
plt.xlabel('ward方法阈值')  # x轴标签
plt.ylabel('簇内最大紧密度')  # y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 调整布局并显示图像
plt.tight_layout()
plt.show()