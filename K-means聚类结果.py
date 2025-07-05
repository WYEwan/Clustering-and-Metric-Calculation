# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.cluster import KMeans  # 从sklearn库中导入KMeans聚类算法
from matplotlib.font_manager import FontProperties  # 用于设置字体属性

# 设置matplotlib的字体和负号显示，以支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei，支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据文件路径
file_path = r"Cluster_sample_data.csv"
# 使用pandas读取CSV文件，文件没有表头，指定列名为['x', 'y']
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 设置聚类的类别数
k = 7
# 初始化KMeans聚类模型，设置聚类数量为k，随机种子为42以保证结果可复现
kmeans = KMeans(n_clusters=k, random_state=42)
# 对数据进行聚类，并将聚类结果存储在新列'cluster'中
data['cluster'] = kmeans.fit_predict(data[['x', 'y']])

# 创建一个图形窗口，设置图形大小为8x6英寸
plt.figure(figsize=(8, 6))

# 定义颜色列表，用于区分不同的聚类结果
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'brown']
# 遍历每个聚类类别，绘制每个类别的数据点
for i in range(k):
    cluster_points = data[data['cluster'] == i]  # 提取当前类别的数据点
    plt.scatter(cluster_points['x'], cluster_points['y'], s=10, color=colors[i], label=f'类别 {i+1}')

# 获取聚类的质心坐标
centroids = kmeans.cluster_centers_
# 绘制质心，使用黑色X标记，大小为100
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color='black', marker='X', label='质心')

# 添加图形的标题、坐标轴标签和图例
plt.title(f"K-means 聚类（K={k}）")  # 设置图形标题
plt.xlabel("X")  # 设置X轴标签
plt.ylabel("Y")  # 设置Y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

# 显示图形
plt.show()