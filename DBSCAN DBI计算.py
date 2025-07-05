# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
data_path = r"F:\课程库存\大三课程库存\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"
data = pd.read_csv(data_path, header=None, names=['x', 'y'])

# 提取坐标信息
X = data[['x', 'y']].values

# 设置DBSCAN聚类的最小样本数
min_samples = 5

# 计算K距离图（用于确定DBSCAN的eps参数）
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# 提取第min_samples-1个邻居的距离（即第min_samples近邻距离）
distances = np.sort(distances[:, min_samples - 1])

# 绘制K距离图
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title(f'K距离图 (min_samples={min_samples})')
plt.xlabel('升序排列点位次')
plt.ylabel(f'到第{min_samples}近邻点的距离')
plt.show()  # 显示K距离图

# 设置DBSCAN聚类的eps范围
eps = 0.18
eps_values = np.linspace(0.1, eps * 2, 50)

# 初始化DBSCAN聚类评估指标列表
db_scores = []           # 戴维森堡丁指数（DBI），值越小越好
separation_scores = []   # 簇间分离度，值越大越好
compactness_scores = []  # 簇内紧密度，值越小越好

# 对不同eps值进行DBSCAN聚类并计算评估指标
for eps_val in eps_values:
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # 如果聚类标签满足条件则计算评估指标
    if len(set(labels)) > 1 and len(set(labels)) != len(labels):
        # 计算戴维森堡丁指数（DBI）
        db_score = davies_bouldin_score(X, labels)
        db_scores.append(db_score)
        
        # 计算簇内紧密度和簇间分离度
        clusters = [X[labels == i] for i in set(labels) if i != -1]  # 获取聚类簇
        
        if len(clusters) > 1:
            # 计算簇间分离度（平均簇间距离）
            centers = [np.mean(cluster, axis=0) for cluster in clusters]  # 计算簇心
            avg_inter_cluster_distance = np.mean(
                cdist(centers, centers)[np.triu_indices(len(centers), k=1)])
            separation_scores.append(avg_inter_cluster_distance)
        else:
            separation_scores.append(np.nan)  # 若只有一个聚类簇则分离度为NaN
            
        # 计算簇内紧密度（平均簇内距离）
        compactness = []
        for cluster in clusters:
            cluster_center = np.mean(cluster, axis=0)
            compactness.append(np.mean(cdist(cluster, [cluster_center])))
        compactness_scores.append(np.mean(compactness))
    else:
        # 如果聚类标签不满足条件（如只有噪声点或所有点为一个簇）则指标为NaN
        db_scores.append(np.nan)
        separation_scores.append(np.nan)
        compactness_scores.append(np.nan)

# 绘制DBSCAN聚类评估指标随eps变化的图像
plt.figure(figsize=(12, 6))

# 绘制戴维森堡丁指数（DBI）图
plt.subplot(1, 3, 1)
plt.plot(eps_values, db_scores, marker='o')
plt.title('DBSCAN聚类戴维森堡丁指数')
plt.xlabel('eps')
plt.ylabel('戴维森堡丁指数（DBI）')
plt.ylim(bottom=0)

# 绘制簇间分离度图
plt.subplot(1, 3, 2)
plt.plot(eps_values, separation_scores, marker='o', color='green')
plt.title('DBSCAN聚类簇间分离度')
plt.xlabel('eps')
plt.ylabel('平均簇间分离度')
plt.ylim(bottom=0)

# 绘制簇内紧密度图
plt.subplot(1, 3, 3)
plt.plot(eps_values, compactness_scores, marker='o', color='red')
plt.title('DBSCAN聚类簇内紧密度')
plt.xlabel('eps')
plt.ylabel('平均簇内紧密度')
plt.ylim(bottom=0)

plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示图像