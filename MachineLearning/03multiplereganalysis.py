import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
print(boston.DESCR)
columns = boston.feature_names
columns
boston.data
boston.target[:50]
df_data = pd.DataFrame(boston.data, columns=boston.feature_names)
df_data.head()

df_target = pd.DataFrame(boston.target, columns=['MEDV'])
df_target.head()

df = pd.concat([df_data, df_target], axis=1)
df.head()


# In[]
df_pickup = df.loc[:, ['LSTAT', 'INDUS', 'DIS', 'RM', 'MEDV']]


sns.pairplot(df_pickup, size=2.0)
plt.show()

df.corr()

# In[]
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, square=True, fmt='.2f')
plt.show()

df.describe()

# In[]
X = df.loc[:, ['LSTAT', 'RM']].values
X

y = df.loc[:, ['MEDV']].values
y[:10]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X.shape
y.shape

X_train.shape
y_train.shape

# In[]
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)

# In[]
lr.intercept_
lr.coef_

X_new = np.array([[12,3]])
X_new

y_prop = 15
y_pred = lr.predict(X_new)
y_pred

price_ratio = y_prop / y_pred
price_ratio

print(f'y_prop       :{y_prop:.2f}')
print(f'y_pred       :{y_pred[0][0]:.2f}')
print(f'price_ratio  :{price_ratio[0][0]:.2f}')

print('R^2')
print(f'train: {lr.score(X_train, y_train): .3f}')
print(f'test: {lr.score(X_test, y_test): .3f}')

# In[]
from sklearn.metrics import mean_squared_error as mse
"RMSE"
f'train: {(mse(y_train, lr.predict(X_train)))**(1/2): .3f}'
f'test : {mse(y_test, lr.predict(X_test))**0.5: .3f}'

df.describe()

# In[] Residuals Plot
def res_plot(y_train, y_train_pred, y_test, y_test_pred):
   res_train= y_train_pred - y_train
   res_test = y_test_pred - y_test

   plt.figure(figsize = (8,8))
   plt.scatter(y_train_pred, res_train, color = 'blue', marker = 'o', label = 'train', alpha = 0.5)
   plt.scatter(y_test_pred, res_test, color = 'green', marker = 's', label = 'test', alpha = 0.5)

   plt.xlabel('Predicted Values')
   plt.ylabel('Residuals')
   plt.legend(loc='upper left')
   plt.hlines(y=0, xmin=-10, xmax=50, color ='red')
   plt.xlim([-10,50])
   plt.show()

res_plot(y_train,lr.predict(X_train), y_test, lr.predict(X_test))

# In[] 3D plot by mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

ax = Axes3D(plt.figure(figsize = (8,5)))
ax.scatter3D(df['LSTAT'], df['RM'], df['MEDV'])

X_grid, Y_grid = np.meshgrid(np.arange(0, 40, 2.5), np.arange(1, 10, 0.5))
w0 = lr.intercept_
w1 = lr.coef_[0,0]
w2 = lr.coef_[0,1]
Z = w0 + w1*X_grid + w2*Y_grid

ax.plot_wireframe(X_grid, Y_grid, Z, alpha = 0.3, color ='red')

ax.set_xlabel('LSTAT')
ax.set_ylabel('RM')
ax.set_zlabel('MEDV')

plt.show()

# In[]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_std = ss.fit_transform(boston.data)
y_std = ss.fit_transform(y)

X_std[:5]
y_std[:5]
X_std.mean(axis=0)

y_std.mean()
X_std.std(axis=0)
y_std.std()
lr_std = LinearRegression()
lr_std.fit(X_std, y_std)
lr_std.coef_
