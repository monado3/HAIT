# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()

print(bc.DESCR)

df_data = pd.DataFrame(bc.data, columns=bc.feature_names)
df_target = pd.DataFrame(bc.target, columns=['class'])
df = pd.concat([df_data,df_target], axis=1)
df.head()
df.describe()

# In[]
df_pickup = df.loc[:, ['worst perimeter','worst concave points','worst radius','mean concave points','class']]
sns.pairplot(df_pickup, size=2.0, hue='class',markers='+')
plt.show()

# In[]
X = df.loc[:,['worst perimeter', 'mean concave points']].values
y = df.loc[:,['class']].values
y = y.reshape(-1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0)
lr.fit(X_train, y_train)

# In[]
def plot_decision_regions(X, y, classifier, resolution=0.02):
  from matplotlib.colors import ListedColormap
  markers = ("s", "x", "o", "^", "v")
  colors = ("red", "blue", "lightgreen", "gray", "cyan")
  cmap = ListedColormap(colors[:len(np.unique(y))])

  x1_min, x1_max = X[:, 0].min()-1 , X[:, 0].max() +1
  x2_min, x2_max = X[:, 1].min()-1 , X[:, 1].max() +1

  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())

  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

# In[]
plot_decision_regions(X_std, y, lr)
plt.xlabel('petal length [standardizd]')
plt.ylabel('petal width [standardizd]')
plt.legend(loc = 'upper left')
plt.show()

# In[]
print(f'train acc:{lr.score(X_train,y_train):.3f}')
print(f'test  acc:{lr.score(X_test ,y_test ):.3f}')
len(y_test)
index = 0
lr.predict_proba(X_test[index].reshape(1, -1))

print(f'answer :{y_test[index]:d}')
print(f'predict:{lr.predict(X_test[index].reshape(1,-1))}')
