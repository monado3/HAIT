# In[]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# In[]
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# In[] quadratic function
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
boston
df_data = pd.DataFrame(boston.data, columns=boston.feature_names)
df_target = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([df_data, df_target], axis=1)
df.head()

lstat = df.loc[:, ['LSTAT']].values
rm = df.loc[:, ['RM']].values
y = df.loc[:, ['MEDV']].values

lstat_train, lstat_test, y_train, y_test = train_test_split(
    lstat, y, test_size=0.3, random_state=0)

model_lin = LinearRegression()
model_lin.fit(lstat_train, y_train)
# In[]
plt.scatter(lstat, y, color='lightgray', label='data')
x = np.arange(0, 40, 1)[:, np.newaxis]
plt.plot(x, model_lin.predict(x), color='red', label='linear')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(loc='upper right')
plt.show()

quad = PolynomialFeatures(degree=2)
lstat_quad = quad.fit_transform(lstat)
lstat_quad

lstat_quad_train, lstat_quad_test, _, _ = train_test_split(
    lstat_quad, y, test_size=0.3, random_state=0)

model_quad = LinearRegression()
model_quad.fit(lstat_quad_train, y_train)
# In[]
plt.scatter(lstat, y, color='lightgray', label='data')
plt.plot(x, model_lin.predict(x), color='red', label='linear')
x_quad = quad.fit_transform(x)
plt.plot(x, model_quad.predict(x_quad), color='green', label='quad')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

# In[] cubic function
cubic = PolynomialFeatures(degree=3)
lstat_cubic = cubic.fit_transform(lstat)
lstat_cubic
lstat_cubic_train, lstat_cubic_test, _, _ = train_test_split(
    lstat_cubic, y, test_size=0.3, random_state=0)

model_cubic = LinearRegression()
model_cubic.fit(lstat_cubic_train, y_train)
# In[]
plt.scatter(lstat, y, color='lightgray', label='data')
plt.plot(x, model_lin.predict(x), color='red', label='linear')
plt.plot(x, model_quad.predict(x_quad), color='green', label='linear')
x_cubic = cubic.fit_transform(x)
x_cubic
plt.plot(x, model_cubic.predict(x_cubic), color='blue', label='cubic')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(loc='upper right')
plt.show()

# In[]
def adjusted(score, n_sample, n_features):
   adjusted_score = 1 - (1 - score) * ((n_sample - 1) /
                                       (n_sample - n_features - 1))
   return adjusted_score


print('adjusted R^2')
print('')

# 線形回帰
print('model_linear')
print('train: %.3f' % adjusted(model_lin.score(
    lstat_train, y_train), len(y_train), 1))
print('test : %.3f' % adjusted(model_lin.score(lstat_test, y_test), len(y_test), 1))
print('')

# 2次関数
print('model_quad')
print('train: %.3f' % adjusted(model_quad.score(
    lstat_quad_train, y_train), len(y_train), 2))
print('test : %.3f' % adjusted(model_quad.score(
    lstat_quad_test, y_test), len(y_test), 2))
print('')

# 3次関数
print('model_cubic')
print('train: %.3f' % adjusted(model_cubic.score(
    lstat_cubic_train, y_train), len(y_train), 3))
print('test : %.3f' % adjusted(model_cubic.score(
    lstat_cubic_test, y_test), len(y_test), 3))

# In[]
X_lin = np.hstack((lstat,rm))
X_lin_train, X_lin_test, _, _ = train_test_split(X_lin, y, test_size = 0.3, random_state = 0)
model_lin_2 = LinearRegression()
model_lin_2.fit(X_lin_train, y_train)

X_quad = np.hstack((lstat_quad, rm))
X_quad_train, X_quad_test, _, _ = train_test_split(X_quad, y, test_size = 0.3 , random_state= 0)
model_quad_2 = LinearRegression()
model_quad_2.fit(X_quad_train, y_train)

X_cubic = np.hstack((lstat_cubic, rm))
X_cubic_train, X_cubic_test, _ , _ = train_test_split(X_cubic, y, test_size = 0.3, random_state = 0)
model_cubic_2 = LinearRegression()
model_cubic_2.fit(X_cubic_train, y_train)
# 自由度調整済み決定係数をtrainとtestに分けて出力
print('adjusted R^2')
print('')

# 線形回帰
print('model_linear_2')
print('train: %.3f' % adjusted(model_lin_2.score(X_lin_train, y_train), len(y_train), 2))
print('test : %.3f' % adjusted(model_lin_2.score(X_lin_test, y_test), len(y_test), 2))
print('')

# 2次関数
print('model_quad_2')
print('train: %.3f' % adjusted(model_quad_2.score(X_quad_train, y_train), len(y_train), 3))
print('test : %.3f' % adjusted(model_quad_2.score(X_quad_test, y_test), len(y_test), 3))
print('')

# 3次関数
print('model_cubic_2')
print('train: %.3f' % adjusted(model_cubic_2.score(X_cubic_train, y_train), len(y_train), 4))
print('test : %.3f' % adjusted(model_cubic_2.score(X_cubic_test, y_test), len(y_test), 4))
