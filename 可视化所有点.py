# 导入所需的库
import pandas as pd  # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于绘图
from matplotlib.font_manager import FontProperties  # 用于字体管理

# 设置 Matplotlib 的字体和符号显示
# 设置字体为 SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 定义数据文件路径
file_path = r"Cluster_sample_data.csv"

# 使用 pandas 读取 CSV 文件
# 文件没有表头，因此指定 header=None
# 并为数据列指定列名 ['x', 'y']
data = pd.read_csv(file_path, header=None, names=['x', 'y'])

# 创建一个图形窗口，设置图形大小为 8x6 英寸
plt.figure(figsize=(8, 6))

# 绘制散点图
# 使用数据中的 'x' 和 'y' 列作为坐标
# 设置点的颜色为蓝色，大小为 10，透明度为 0.6
plt.scatter(data['x'], data['y'], color='blue', s=10, alpha=0.6)

# 添加标题和坐标轴标签
plt.title("数据点分布图")  # 图形标题
plt.xlabel("X")  # X 轴标签
plt.ylabel("Y")  # Y 轴标签

# 添加网格线
plt.grid(True)

# 显示图形
plt.show()