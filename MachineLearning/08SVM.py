# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X_1 = iris.data[:100,[2,3]]
y_1 = iris.target[:100]
y_1
iris.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_1)
X_1_std = scaler.transform(X_1)

# In[]
plt.scatter(X_1_std[:50,[0]], X_1_std[:50,[1]], color='red', marker='s')
plt.scatter(X_1_std[50:100,[0]], X_1_std[50:100,[1]], color='blue', marker='x')
plt.show()

# In[]
from sklearn.model_selection import train_test_split
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1_std, y_1, test_size=0.3, random_state=0)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_1_train, y_1_train)

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
plot_decision_regions(X_1_std, y_1, svc)

# In[]
X_2 = iris.data[50:150,[2,3]]
y_2 = iris.target[50:150]
scaler_2 = StandardScaler()
scaler_2.fit(X_2)
X_2_std = scaler.transform(X_2)

# In[]
plt.scatter(X_2_std[:50,[0]],X_2_std[:50,[1]],color='red', marker='s')
plt.scatter(X_2_std[50:100,[0]],X_2_std[50:100,[1]],color='blue',marker='x')
plt.show()

# In[]
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2_std, y_2, test_size=0.3, random_state=0)
svc_slack = SVC(kernel='linear',C=1.0)
svc_slack.fit(X_2_train, y_2_train)
plot_decision_regions(X_2_std, y_2, svc_slack)
svc_slack.score(X_2_test, y_2_test)

# In[]
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor, 1, -1)

# In[]
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor== -1,0], X_xor[y_xor== -1,1],c='r',marker='s',label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show()

# In[]
X_xor_train, X_xor_test, y_xor_train, y_xor_test = train_test_split(X_xor, y_xor, test_size=0.3, random_state=0)
linear_svm = SVC(kernel='linear', C=0.1)
linear_svm.fit(X_xor_train, y_xor_train)
plot_decision_regions(X_xor, y_xor, classifier=linear_svm)
linear_svm.score(X_xor_test, y_xor_test)

# In[]
rbf_svm = SVC(kernel='rbf', gamma=0.1, C=10)
rbf_svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=rbf_svm)
rbf_svm.score(X_xor_test, y_xor_test)
