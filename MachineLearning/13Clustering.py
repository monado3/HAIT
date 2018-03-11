# In[]
import matplotlib.pyplot as plt
import numpy as np
# In[]
# シルエット図を出力
from matplotlib import cm
from numpy.random import *
# In[] k-means
from sklearn.cluster import KMeans
# In[]
# In[]
# サンプル数が大きく異なるクラスタを生成
# サンプル数50の超球状のクラスタを生成
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.fit_transform(X)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_std)
km = KMeans(n_clusters=3, init='random', n_init=10,
            max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X_pca[:, 0:2])

# In[]


def kmeans_plot(n_clusters, km, X):
  # クラスタの予測値を算出
  y_km = km.fit_predict(X)

  # クラスタごとに散布図をプロット
  # 5クラスまでプロットできる
  for i, color, marker in zip(range(n_clusters), 'rgbcm', '>o+xv'):
    plt.scatter(X[y_km == i, 0],            # 横軸の値
                X[y_km == i, 1],            # 縦軸の値
                color=color,              # プロットの色
                marker=marker,            # プロットの形
                label='cluster ' + str(i)  # ラベル
                )

  # クラスタの中心をプロット
  plt.scatter(km.cluster_centers_[:, 0],    # 横軸の値
              km.cluster_centers_[:, 1],    # 縦軸の値
              color='y',                    # プロットの色
              marker='*',                   # プロットの形
              label='centroids',            # ラベル
              s=300,                        # プロットのサイズを大きくして見やすくする
              )

  plt.legend()
  plt.grid()
  plt.show()


# In[]
kmeans_plot(3, km, X_pca[:, 0:2])
y_km
y_correct = np.hstack((y[0:50], y[100:150]))
y_correct = np.hstack((y_correct, y[50:100]))
y_correct
ans = y_km == y_correct
ans
correct_ans = len(np.where(ans == True)[0])
correct_ans / len(y)

# In[]
# 超級上でないクラスタを作成
X_1, _ = make_blobs(n_samples=50,
                    n_features=2,
                    centers=1,
                    cluster_std=0.4,
                    center_box=[0, -2],
                    random_state=6
                    )

# サンプル数50の超球状のクラスタを生成
X_2, _ = make_blobs(n_samples=50,
                    n_features=2,
                    centers=1,
                    cluster_std=0.4,
                    center_box=[0, 2],
                    random_state=9
                    )

seed(2)
X_31 = np.array([i / 15 for i in range(-50, 0)])
X_32 = np.array([0.5 * float(i) + 2 + float(randint(100)) / 100 for i in X_31])
X_3 = np.hstack((X_31.reshape(-1, 1), X_32.reshape(-1, 1)))

# 2種類のクラスタを色分けして表示
plt.scatter(X_1[:, 0], X_1[:, 1], c='r', marker='s', s=50)
plt.scatter(X_2[:, 0], X_2[:, 1], c='y', marker='o', s=50)
plt.scatter(X_3[:, 0], X_3[:, 1], c='b', marker='x', s=50)
plt.show()

# In[]
X = np.vstack((X_1, X_2))
X = np.vstack((X, X_3))
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=0)
y_km = km.fit_predict(X)
kmeans_plot(3, km, X)



# サンプル数25のクラスタを2つ生成
X_1, _ = make_blobs(n_samples=50,
                    n_features=2,
                    centers=2,
                    cluster_std=0.5,
                    random_state=3
                    )

# サンプル数300のクラスタを1つ生成
X_2, _ = make_blobs(n_samples=300,
                    n_features=2,
                    centers=1,
                    cluster_std=1.0,
                    center_box=(-5, 5),
                    random_state=5
                    )

# 2種類のクラスタを色分けして表示
plt.scatter(X_1[:, 0], X_1[:, 1], c='y', marker='o', s=50)
plt.scatter(X_2[:, 0], X_2[:, 1], c='b', marker='x', s=50)
plt.show()

# In[]
X_blobs = np.vstack((X_1, X_2))
km2 = KMeans(n_clusters=3,
             init='random',
             n_init=10,
             max_iter=300,
             tol=1e-04,
             random_state=0,
             )
y_km = km2.fit_predict(X_blobs)
kmeans_plot(3, km2, X_blobs)

# In[]
distortions = []
for k in range(1, 11):
  km = KMeans(n_clusters=k,
              init='random',
              n_init=10,
              max_iter=300,
              random_state=0,
              )
  km.fit(X_pca[:, 0:2])
  distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xticks([i for i in range(1, 11)])
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')
plt.show()

# In[] silhouette coefficient
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
            )
y_km = km.fit_predict(X_pca[:, 0:2])
s = silhouette_samples(X_pca[:, 0:2], y_km, metric='euclidean')
s[:10]
len(s)

# In[] シルエット図を出力
from matplotlib import cm

# シルエット図を出力する関数を定義
def draw_silhouettes(X, y_km):
  cluster_labels = np.unique(y_km)                     # クラスラベルを重複なく抽出
  n_clusters = cluster_labels.shape[0]                 # クラスタの数を算出

  # シルエット係数を計算
  # (X, y_km, 距離の種類)
  s = silhouette_samples(X, y_km, metric='euclidean')

  # 各ラベルごとにシルエット図を描画
  y_ax_lower, y_ax_upper = 0, 0                         # シルエット図の上端と下端の初期値を設定
  yticks = []                                          # 縦軸のメモリ位置を格納するリストを生成
  for i, label in enumerate(cluster_labels):
    label_s = s[y_km == label]                     # 該当するクラスタについて､シルエット係数を算出
    label_s.sort()                               # シルエット係数を小さい順に並べ替える
    y_ax_upper += len(label_s)                   # シルエット図の上端を､サンプルの数だけ引き上げる
    color = cm.jet(float(i) / n_clusters)        # color mapから色を取得
    plt.barh(range(y_ax_lower, y_ax_upper),      # 横軸の範囲を指定
             label_s,                    # バーの幅を指定
             height=1.0,                 # バーの厚みを指
             color=color)                # バーの色を指定
    yticks.append((y_ax_lower + y_ax_upper) / 2)  # クラスタラベルの表示位置を追加
    y_ax_lower += len(label_s)                   # シルエット図の下端を､サンプルの数だけ引き上げる

  # 係数の平均値に破線を引く(横軸の値, 色, 線の形式)
  plt.axvline(np.mean(s), color="red", linestyle="--")
  plt.yticks(yticks, cluster_labels + 1)               # クラスタレベルを表示(位置, 縦軸の値)
  plt.ylabel('Cluster')
  plt.xlabel('silhouette coefficient')
  plt.show()


# シルエット図を出力
draw_silhouettes(X_pca[:, 0:2], y_km)

# In[]
# k-means法を実行
# 2クラスタに分ける
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,   # クラスタの個数を指定
            init='random',  # セントロイドの初期値の決め方を決定
            n_init=10,      # 異なるセントロイドの初期値を用いた実行回数
            max_iter=300,   # ひとつのセントロイドを用いたときの最大イテレーション回数
            tol=1e-04,      # 収束と判定するための相対的な許容誤差
            random_state=0, # セントロイドの初期化に用いる乱数生成器の状態
           )
y_km = km.fit_predict(X_pca[:, 0:2])
draw_silhouettes(X_pca[:,0:2],y_km)

# In[]
# k-means法を実行
# 4クラスタに分ける
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,   # クラスタの個数を指定
            init='random',  # セントロイドの初期値の決め方を決定
            n_init=10,      # 異なるセントロイドの初期値を用いた実行回数
            max_iter=300,   # ひとつのセントロイドを用いたときの最大イテレーション回数
            tol=1e-04,      # 収束と判定するための相対的な許容誤差
            random_state=0, # セントロイドの初期化に用いる乱数生成器の状態
           )
y_km = km.fit_predict(X_pca[:, 0:2])
draw_silhouettes(X_pca[:,0:2],y_km)
