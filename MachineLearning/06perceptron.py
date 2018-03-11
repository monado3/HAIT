# In[]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# In[]
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
# In[]
from sklearn.model_selection import train_test_split
# In[] preprcessing
from sklearn.preprocessing import StandardScaler

iris = load_iris()
print(iris.DESCR)
columns = iris.feature_names
columns
X = iris.data
X[:10]
y = iris.target
y

X_df = pd.DataFrame(X, columns=columns)
y_df = pd.DataFrame(y, columns=['species'])
df = pd.concat([X_df, y_df], axis=1)
df.head()
df.describe()

sns.pairplot(df, hue='species', size=2.0)
plt.show()

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_1 = X_std[0:100, [2, 3]]
y_1 = y[:100]

# In[]
plt.scatter(X_std[:50, 2], X_std[:50, 3],
            color='red', marker='o', label='setosa')
plt.scatter(X_std[50:100, 2], X_std[50:100, 3],
            color='blue', marker='x', label='versicolor')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(
    X_1, y_1, test_size=0.3, random_state=0)
ppn_1 = Perceptron(eta0=0.1)
ppn_1.fit(X_1_train, y_1_train)

# In[]


def plot_decision_regions(X, y, classifier, resolution=0.02):
  from matplotlib.colors import ListedColormap
  markers = ('s', 'x', 'o', '^', 'v')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                c=cmap(idx), marker=markers[idx], label=cl)


# In[]
plot_decision_regions(X_1, y_1, ppn_1)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Setosa or Versicolor')
plt.show()

# In[]
print(f'train acc: {ppn_1.score(X_1_train, y_1_train):3f}')
print(f'test acc: {ppn_1.score(X_1_test, y_1_test):3f}')

# In[]
index = 28
print(f'answer :{y_1_test[index]:d}')
print(f'predict:{ppn_1.predict(X_1_test[index].reshape(1,-1))}')

# In[]
X_2 = X_std[:, [2, 3]]
y_2 = y

X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(
    X_2, y_2, test_size=0.3, random_state=0)
# In[] my check
X_2_train[:5]
y_2_train

len(X_2_train)
len(y_2_train)

y_2_train == 0
X_2_train[y_2_train == 0, :]

for idx, cl in enumerate(np.unique(y)):
  plt.scatter(x=X_2_train[y_2_train == cl, 0],
              y=X_2_train[y_2_train == cl, 1], label=cl)
plt.show()
# ↑ why the Perceptron algorithm solved this question?


# In[]
ppn_2 = Perceptron(eta0=0.1)
ppn_2.fit(X_2_train, y_2_train)

# In[]
plot_decision_regions(X_2, y_2, ppn_2)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Setosa or Versicolor')
plt.show()
print(f'train acc:{ppn_2.score(X_2_train,y_2_train):.3f}')
print(f'test acc: {ppn_2.score(X_2_test, y_2_test):.3f}')

index = 35
print(f'answer :{y_2_test[index]:d}')
print(f'predict:{ppn_2.predict(X_2_test[index].reshape(1,-1))} ')

# In[] my check
from matplotlib.colors import ListedColormap
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])
