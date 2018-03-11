# In[]
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# In[]
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
feature_names
X[:10]

# In[]
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
plt.figure(figsize=(12, 9))
for i, (p0, p1) in enumerate(pairs):
  plt.subplot(2, 3, i + 1)
  for target, marker, color in zip(list(range(3)), '>ox', 'rgb'):
    plt.scatter(X[iris.target == target, p0],
                X[iris.target == target, p1], marker=marker, c=color)
  plt.xlabel(feature_names[p0])
  plt.ylabel(feature_names[p1])
  plt.xticks([])
  plt.yticks([])

plt.show()

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.fit_transform(X)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_std)
X_pca[:10]
# In[]
plt.figure(figsize=(6, 6))
for target, marker, color in zip(range(3), '>ox', 'rgb'):
  plt.scatter(X_pca[iris.target == target, 0],
              X_pca[iris.target == target, 1], marker=marker, color=color)
plt.xlabel('PC 0')
plt.ylabel('PC 1')
plt.xticks([])
plt.yticks([])
plt.show()
# In[]
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

plt.figure(figsize=(6, 6))
for target, marker, color in zip(range(3), '>ox', 'rgb'):
  plt.scatter(X_pca[iris.target == target, 2],
              X_pca[iris.target == target, 3], marker=marker, color=color)
plt.xlabel('PC 2')
plt.ylabel('PC 3')
plt.xticks([])
plt.yticks([])
plt.show()

# In[]
pca.components_
np.sqrt(pca.explained_variance_)
pca.components_*np.sqrt(pca.explained_variance_)[:,np.newaxis]

# In[]
X_std[:10]
iris.feature_names
X_std[:,[0,2]][:10] #this differs from text In[28]
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
scores_1 = cross_val_score(SVC(),X_std[:,[0,2]],y,cv=5)
scores_1.mean()
X_pca[:10]
X_pca[:,0:2][:10]
scores_2 = cross_val_score(SVC(),X_pca[:,[0,2]],y,cv=5)
scores_2.mean()
print(f'特徴選択:{scores_1.mean():.4f}')
print(f'特徴抽出:{scores_2.mean():.4f}')
