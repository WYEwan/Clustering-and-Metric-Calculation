# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
from sklearn.cluster import DBSCAN  # 导入DBSCAN聚类算法
from sklearn.neighbors import NearestNeighbors  # 用于计算最近邻距离
import matplotlib.pyplot as plt  # 用于绘图
# 设置Matplotlib的字体和负号显示，确保中文和负号能够正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据文件路径
data_path = r"F:\课程库存\大三课程库存\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"
# 读取CSV文件数据，假设数据文件中没有表头，且只有两列（x和y坐标）
data = pd.read_csv(data_path, header=None, names=['x', 'y'])
# 将数据的x和y列提取为NumPy数组，用于后续计算
X = data[['x', 'y']].values

# 设置DBSCAN算法中的最小样本数参数
min_samples = 5  
# 创建NearestNeighbors对象，用于计算每个点到其第min_samples个最近邻点的距离
neighbors = NearestNeighbors(n_neighbors=min_samples)
# 对数据进行拟合
neighbors_fit = neighbors.fit(X)
# 计算每个点到其min_samples个最近邻点的距离和对应的索引
distances, indices = neighbors_fit.kneighbors(X)
# 提取每个点到其第min_samples个最近邻点的距离，并按升序排列
distances = np.sort(distances[:, min_samples - 1])

# 绘制K距离图，用于确定DBSCAN算法中的eps参数
plt.figure(figsize=(8, 4))  # 设置图形大小
plt.plot(distances)  # 绘制距离曲线
plt.title(f'K距离图 (min_samples={min_samples})')  # 设置标题
plt.xlabel('升序排列点位次')  # 设置x轴标签
plt.ylabel(f'到第{min_samples}近邻点的距离')  # 设置y轴标签
plt.show()  # 显示图形

# 根据K距离图选择合适的eps值
eps = 0.18
print(f"eps={eps}")  # 打印eps值
print(f"min_samples={min_samples}")  # 打印min_samples值

# 创建DBSCAN对象并设置参数
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
# 对数据进行聚类，并获取每个点的聚类标签
labels = dbscan.fit_predict(X)

# 绘制DBSCAN聚类结果
plt.figure(figsize=(8, 6))  # 设置图形大小
# 获取所有唯一的聚类标签
unique_labels = set(labels)
# 为每个聚类生成颜色
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# 遍历每个聚类标签
for k, col in zip(unique_labels, colors):
    if k == -1:  # 如果是噪声点（标签为-1），使用黑色表示
        col = [0, 0, 0, 1]  
    # 获取当前聚类的所有点的布尔索引
    class_member_mask = (labels == k)
    # 提取当前聚类的所有点
    xy = X[class_member_mask]
    # 绘制当前聚类的点
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
# 设置图形标题和坐标轴标签
plt.title(f'DBSCAN聚类结果 (eps={eps:.2f}, min_samples={min_samples})')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()  # 显示图形