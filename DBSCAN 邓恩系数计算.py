# 导入所需的库
import pandas as pd  # 数据处理库
import numpy as np  # 数学计算库
from sklearn.cluster import DBSCAN  # DBSCAN聚类算法
from sklearn.neighbors import NearestNeighbors  # 近邻搜索工具
import matplotlib.pyplot as plt  # 数据可视化库
from scipy.spatial.distance import cdist  # 计算距离矩阵的工具

# 设置matplotlib的字体和显示负号的样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义数据文件路径
data_path = r"F:\课程库存\大三课程库存\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"

# 读取数据，假设数据文件中没有表头，且只有两列（x和y）
data = pd.read_csv(data_path, header=None, names=['x', 'y'])

# 将数据转换为二维数组，用于后续的聚类分析
X = data[['x', 'y']].values

# 设置DBSCAN算法的最小样本数参数
min_samples = 5  

# 使用NearestNeighbors计算每个点到其第min_samples个近邻的距离
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)  # 对数据进行拟合
distances, indices = neighbors_fit.kneighbors(X)  # 获取每个点的近邻距离和索引

# 提取每个点到其第min_samples个近邻的距离，并按升序排列
distances = np.sort(distances[:, min_samples - 1]) 

# 绘制K距离图，用于选择合适的eps值
plt.figure(figsize=(8, 4))  # 设置图像大小
plt.plot(distances)  # 绘制距离曲线
plt.title(f'K距离图 (min_samples={min_samples})')  # 设置标题
plt.xlabel('升序排列点位次')  # 设置x轴标签
plt.ylabel(f'到第{min_samples}近邻点的距离')  # 设置y轴标签
plt.show()  # 显示图像

# 设置DBSCAN的eps值范围
eps = 0.18  # 基准eps值
eps_values = np.linspace(0.1, eps * 2, 50)  # 在0.1到0.36之间生成50个eps值

# 初始化用于存储评估指标的列表
dunn_indices = []  # 邓恩系数
separation_scores = []  # 簇间最小分离度
compactness_scores = []  # 簇内最大紧密度

# 遍历不同的eps值，评估DBSCAN聚类效果
for eps_val in eps_values:
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)  # 创建DBSCAN模型
    labels = dbscan.fit_predict(X)  # 对数据进行聚类并获取标签

    # 如果聚类结果中存在多个簇（且不是所有点都被标记为噪声）
    if len(set(labels)) > 1 and len(set(labels)) != len(labels):
        clusters = [X[labels == i] for i in set(labels) if i != -1]  # 提取每个簇的数据

        # 如果存在多个簇，计算簇间最小分离度
        if len(clusters) > 1:
            min_inter_cluster_distance = np.inf  # 初始化最小簇间距离为无穷大
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = np.min(cdist(clusters[i], clusters[j]))  # 计算两个簇之间的最小距离
                    min_inter_cluster_distance = min(min_inter_cluster_distance, distance)  # 更新最小簇间距离
            separation_scores.append(min_inter_cluster_distance)
        else:
            separation_scores.append(np.nan)  # 如果只有一个簇，簇间分离度无意义

        # 计算每个簇的紧密度
        compactness = []
        for cluster in clusters:
            cluster_center = np.mean(cluster, axis=0)  # 计算簇的中心
            compactness.append(np.max(cdist(cluster, [cluster_center])))  # 计算簇内最大距离
        compactness_scores.append(np.mean(compactness))  # 计算所有簇的平均紧密度

        # 如果簇内紧密度大于0，计算邓恩系数
        if compactness_scores[-1] > 0:
            dunn_index = separation_scores[-1] / compactness_scores[-1]  # 邓恩系数 = 簇间分离度 / 簇内紧密度
            dunn_indices.append(dunn_index)
        else:
            dunn_indices.append(np.nan)  # 如果簇内紧密度为0，邓恩系数无意义
    else:
        # 如果聚类结果无效（只有一个簇或全是噪声），将所有指标标记为NaN
        dunn_indices.append(np.nan)  
        separation_scores.append(np.nan)
        compactness_scores.append(np.nan)

# 绘制评估指标随eps变化的图像
plt.figure(figsize=(12, 6))  # 设置图像大小

# 绘制邓恩系数随eps变化的图像
plt.subplot(1, 3, 1)  # 创建子图1
plt.plot(eps_values, dunn_indices, marker='o')  # 绘制邓恩系数曲线
plt.title('DBSCAN聚类邓恩系数')  # 设置标题
plt.xlabel('eps')  # 设置x轴标签
plt.ylabel('邓恩系数')  # 设置y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制簇内最大紧密度随eps变化的图像
plt.subplot(1, 3, 2)  # 创建子图2
plt.plot(eps_values, separation_scores, marker='o', color='green')  # 绘制簇间分离度曲线
plt.title('DBSCAN聚类簇间最小分离度')  # 设置标题
plt.xlabel('eps')  # 设置x轴标签
plt.ylabel('簇间最小分离度')  # 设置y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 绘制簇间最小分离度随eps变化的图像
plt.subplot(1, 3, 3)  # 创建子图3
plt.plot(eps_values, compactness_scores, marker='o', color='red')  # 绘制簇内紧密度曲线
plt.title('DBSCAN聚类簇内最大紧密度')  # 设置标题
plt.xlabel('eps')  # 设置x轴标签
plt.ylabel('簇内最大紧密度')  # 设置y轴标签
plt.ylim(bottom=0)  # 设置y轴下限为0

# 调整子图布局并显示图像
plt.tight_layout()
plt.show()