# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np   # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # 用于层次聚类
from sklearn.preprocessing import StandardScaler  # 用于数据标准化

# 设置matplotlib的字体和显示负号的样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei，用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 定义数据文件路径
data_path = r"D:\通用文件夹\数据科学与数据挖掘导论\作业1\Cluster_sample_data.csv"

# 读取数据文件
data = pd.read_csv(data_path, header=None, names=['x', 'y'])  # 读取CSV文件，无表头，指定列名为x和y

# 提取数据中的x和y列，并将其转换为NumPy数组
X = data[['x', 'y']].values

# 数据标准化
scaler = StandardScaler()  # 创建标准化器对象
X_scaled = scaler.fit_transform(X)  # 对数据进行标准化处理

# 进行层次聚类
Z = linkage(X_scaled, method='ward')  # 使用ward方法计算层次聚类的连接矩阵

# 绘制层次聚类树状图
plt.figure(figsize=(10, 6))  # 设置图形大小
dendrogram(Z)  # 绘制树状图
plt.title("层次聚类树状图")  # 设置标题
plt.xlabel('数据点排列')  # 设置x轴标签
plt.ylabel('ward方法阈值')  # 设置y轴标签
plt.gca().set_xticks([])  # 隐藏x轴刻度
plt.show()  # 显示图形

# 设置层次聚类的截断阈值
dendrogram_cutoff = 30

# 根据截断阈值对数据进行聚类
labels = fcluster(Z, t=dendrogram_cutoff, criterion='distance')  # 使用距离作为标准进行聚类

# 绘制聚类结果图
plt.figure(figsize=(8, 6))  # 设置图形大小
unique_labels = set(labels)  # 获取所有唯一的聚类标签
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]  # 生成颜色列表

# 绘制每个聚类的点
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)  # 获取当前聚类的布尔索引
    xy = X[class_member_mask]  # 提取当前聚类的数据点
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),  # 绘制散点图
             markeredgecolor='k', markersize=6)

# 设置图形标题和轴标签
plt.title(f'层次聚类结果 (ward方法阈值={dendrogram_cutoff})')  # 显示标题
plt.xlabel('X')  # 设置x轴标签
plt.ylabel('Y')  # 设置y轴标签
plt.show()  # 显示图形