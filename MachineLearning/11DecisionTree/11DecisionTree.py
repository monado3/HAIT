# In[]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_1 = iris.data[:, [2, 3]]
y_1 = iris.target
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(
    X_1, y_1, test_size=0.3, random_state=0)
tree_1 = DecisionTreeClassifier(random_state=0)
tree_1.fit(X_1_train, y_1_train)

# In[]
def plot_decision_regions(X, y, classifier, resolution=0.02):
  from matplotlib.colors import ListedColormap
  markers = ("s", "x", "o", "^", "v")
  colors = ("red", "blue", "lightgreen", "gray", "cyan")
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
plot_decision_regions(X_1,y_1,tree_1)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='best')
plt.show()
# In[]
print(f'train: {tree_1.score(X_1_train, y_1_train):.3f}')
print(f'test : {tree_1.score(X_1_test, y_1_test):.3f}')

from sklearn.tree import export_graphviz
export_graphviz(tree_1, out_file='tree_1.dot', feature_names=['petal length','petal width'],class_names=['setosa','versicolour','virsinica'],impurity=False,filled=True)

# In[]
import cv2
img = cv2.imread('tree_1.png')
plt.figure(figsize=(12,12))
plt.imshow(img)
plt.show()
# In[]
from sklearn.tree import DecisionTreeClassifier
tree_2 = DecisionTreeClassifier(random_state=0,max_depth=3)
tree_2.fit(X_1_train,y_1_train)
plot_decision_regions(X_1,y_1,tree_2)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='best')
plt.show()

# In[]
from sklearn.ensemble import RandomForestClassifier
rfc_1 = RandomForestClassifier(random_state=0, n_estimators=10)
rfc_1.fit(X_1_train,y_1_train)

plot_decision_regions(X_1,y_1,rfc_1)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
# In[]
print(f'train accuracy:{rfc_1.score(X_1_train,y_1_train):.3f}')
print(f'test  accuracy:{rfc_1.score(X_1_test,y_1_test):.3f}')

# In[]
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X_2_train,X_2_test,y_2_train,y_2_test = train_test_split(bc.data,bc.target,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfc_2=RandomForestClassifier(random_state=0,n_estimators=100)
rfc_2.fit(X_2_train,y_2_train)
# In[]
importances = rfc_2.feature_importances_
importances
n_features=len(bc.feature_names)
plt.figure(figsize=(12,8))
plt.barh(range(n_features),importances,align='center')
plt.yticks(np.arange(n_features),bc.feature_names)
plt.show()
rfc_2.predict_proba(X_2_test[0].reshape(1,-1))

# In[]
from sklearn.ensemble import GradientBoostingClassifier
gbct=GradientBoostingClassifier(random_state=0,max_depth=3,learning_rate=0.1)
gbct.fit(X_2_train,y_2_train)
print(f'train accuracy:{gbct.score(X_2_train,y_2_train):.3f}')
print(f'test accuracy:{gbct.score(X_2_test,y_2_test):.3f}')
# In[]
plt.figure(figsize=(12,8))
plt.barh(range(n_features),gbct.feature_importances_,align='center')
plt.yticks(np.arange(n_features),bc.feature_names)
plt.show()
gbct.predict_proba(X_2_test[0].reshape(1,-1))
