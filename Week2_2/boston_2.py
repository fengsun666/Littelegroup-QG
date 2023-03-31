import math
from sklearn import datasets
import matplotlib.pyplot as plt

house = datasets.load_boston()
x = house.data
y = house.target
nums = len(house.feature_names)
columns = 3
rows = math.ceil(nums / columns)
plt.figure(figsize=(10, 12))   # 指定宽和高（单位英寸）
for i in range(nums):
    plt.subplot(rows, columns, i + 1)   # 一图排列多个子图
    plt.plot(x[:, i], y, "b+")
    plt.title(house.feature_names[i])
plt.subplots_adjust(hspace=0.8)    # 调整子图布局
plt.show()
