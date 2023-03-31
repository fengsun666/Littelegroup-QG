import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

house = datasets.load_boston()
x = house.data
y = house.target
stand = StandardScaler()   # 化为标准差形式，正态分布
stand_x = stand.fit_transform(x)
best = SelectKBest(f_regression, k=3)
best.fit_transform(stand_x, y)
best_index = best.get_support()
print(best_index)
best_features = house.feature_names[best_index]
print(best_features)
x_best = x[:, best_index]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr = LinearRegression()   # 最小二乘回归，线性近似
lr.fit(x_train, y_train)
y_test_predit = lr.predict(x_test)
error_1 = mean_squared_error(y_test, y_test_predit)
y_train_predit = lr.predict(x_train)
error_2 = mean_squared_error(y_train, y_train_predit)
print('测试误差为：', error_1, '训练误差', error_2)
score = r2_score(y_test, y_test_predit).round(5)    # 计算相关系数
plt.plot(y_test_predit, "r-", label="predit_value")
plt.plot(y_test, "b-", label="true_value")
plt.legend()
plt.show()
