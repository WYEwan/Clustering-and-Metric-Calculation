# 导入必要的库
import pandas as pd  # 用于数据处理
import matplotlib.pyplot as plt  # 用于绘图
from sklearn.cluster import KMeans  # 用于K-means聚类
from matplotlib.font_manager import FontProperties  # 用于设置字体

# 设置matplotlib的字体和显示参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体，以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指定数据文件路径
file_path = r"Cluster_sample_data.csv"

# 使用pandas读取CSV文件，假设数据文件中没有表头，故设置header=None，并指定列名为['x', 'y']
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 定义要测试的聚类数范围（从1到10）
k_values = range(1, 11)

# 初始化一个列表，用于存储每个聚类数对应的误差平方和（SSE）
sse = []

# 遍历每个聚类数k
for k in k_values:
    # 创建KMeans模型，指定聚类数为k，随机种子为500以确保结果可复现
    kmeans = KMeans(n_clusters=k, random_state=500)
    # 使用KMeans模型拟合数据
    kmeans.fit(data)
    # 将当前聚类数的SSE值追加到列表中
    sse.append(kmeans.inertia_)

# 创建一个图形窗口，设置大小为8x6英寸
plt.figure(figsize=(8, 6))

# 绘制SSE随聚类数k变化的折线图
# 使用红色线条，带圆圈标记
plt.plot(k_values, sse, marker='o', linestyle='-', color='red')

# 设置图表标题
plt.title("K-means聚类手肘法图表")

# 设置x轴标签
plt.xlabel("聚类数（k）")

# 设置y轴标签
plt.ylabel("误差平方和（SSE）")

# 设置x轴刻度为聚类数范围
plt.xticks(k_values)

# 添加网格线
plt.grid(True)

# 显示图表
plt.show()