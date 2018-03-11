import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X_1 = iris.data[50:,[0,2]]
y_1 = iris.target[50:]

from sklearn.model_selection import train_test_split
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size=0.3, random_state=0)

from sklearn.svm import SVC
svc_1 = SVC(kernel='linear', C=1)
svc_1.fit(X_1_train, y_1_train)

# In[]
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_1_test, svc_1.predict(X_1_test))
cm_1
# In[]
def plot_cm(confmat):
  fig, ax = plt.subplots(figsize=(5,5))
  ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
  for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
      ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
  plt.title('predict label')
  plt.ylabel('true label')
  plt.tight_layout()
  plt.show()

# In[]
plot_cm(cm_1)

# In[]
from sklearn import metrics
metrics.accuracy_score(y_1_test, svc_1.predict(X_1_test))

# In[]
metrics.recall_score(y_1_test, svc_1.predict(X_1_test))
metrics.precision_score(y_1_test, svc_1.predict(X_1_test))
metrics.f1_score(y_1_test, svc_1.predict(X_1_test))

# In[]
iris = load_iris()
X_2 = iris.data[:,[0,2]]
y_2 = iris.target


X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2,y_2,test_size=0.3,random_state=0)
svc_2 = SVC(kernel='linear', C=1.0)
svc_2.fit(X_2_train, y_2_train)
cm_2 = confusion_matrix(y_2_test, svc_2.predict(X_2_test))
cm_2
plot_cm(cm_2)
metrics.precision_score(y_2_test, svc_2.predict(X_2_test), average='macro')
metrics.recall_score(y_2_test, svc_2.predict(X_2_test), average='macro')
metrics.f1_score(y_2_test, svc_2.predict(X_2_test), average='macro')

# In[]
from sklearn.datasets import load_boston
boston = load_boston()
df_data = pd.DataFrame(boston.data, columns=boston.feature_names)
df_target = pd.DataFrame(boston.target, columns=['MEDV'])
df = pd.concat([df_data, df_target], axis=1)
df.head()
X_3 = df.loc[:,['LSTAT','RM']].values
y_3 = df.loc[:,['MEDV']].values
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.model_selection import KFold
kf_3 = KFold(n_splits=5, shuffle=True, random_state=0)
from sklearn.model_selection import cross_val_score
scores_3 = cross_val_score(lr, X_3,y_3,cv=kf_3)
scores_3
scores_3.mean()
scores_3.std()

# In[]
from sklearn.datasets import load_iris
iris = load_iris()
X_4 = iris.data[:,[2,3]]
y_4 = iris.target
svc_4 = SVC(kernel='rbf',gamma=0.1,C=1.0)
from sklearn.model_selection import StratifiedKFold
kf_4 = StratifiedKFold(n_splits=5, shuffle=True,random_state=0)
scores_4 = cross_val_score(svc_4, X_4,y_4,cv=kf_4)
scores_4.mean()
scores_4.std()

# In[]
np.random.seed(0)
X_xor=np.random.randn(200,2)
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor=np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show()

# In[]
X_xor_train,X_xor_test,y_xor_train,y_xor_test=train_test_split(X_xor,y_xor,test_size=0.3,random_state=0)
param_grid={'C':[0.1,1.0,10,100,1000,10000],
            'gamma':[0.001,0.01,0.1,1,10]}
kf_5 = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
from sklearn.model_selection import GridSearchCV
gs_svc = GridSearchCV(SVC(), param_grid,cv=kf_5)
gs_svc.fit(X_xor_train, y_xor_train)
gs_svc.best_params_
gs_svc.best_score_
gs_svc.score(X_xor_test, y_xor_test)

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
    plt.show()

# In[]
plot_decision_regions(X_xor, y_xor, gs_svc)
