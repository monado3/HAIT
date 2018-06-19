# In[]
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer import serializers

# In[]
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home='.')
X = mnist.data
X
y = mnist.target
y
X = X.astype(np.float32)
y = y.astype(np.int32)
X
len(X)
X /= 255.
X
X.shape
y.shape

# In[]
def draw_digit(data):
  """数値の行列データを画像表示"""
  plt.figure(figsize=(3,3))
  X, Y = np.meshgrid(np.arange(28),np.arange(28))
  Z = data.reshape(28,28)
  Z = Z[::-1,:]
  plt.pcolor(X, Y, Z)
  plt.tick_params(labelbottom='off')
  plt.tick_params(labelleft='off')
  plt.gray()
  plt.show()
idx = 50000
draw_digit(X[idx])
y[idx]

# In[]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0)
N_train = len(X_train)
N_train
N_test = len(X_test)
N_test

# In[]
class MLP(chainer.Chain):
  '''ニューラルネットの構造を定義'''
  def __init__(self):
    super(MLP, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(784, 900)
      self.l2 = L.Linear(900, 1000)
      self.l3 = L.Linear(1000, 500)
      self.l4 = L.Linear(500, 10)

  def __call__(self, X):
    h1 = F.relu(self.l1(X))
    h2 = F.relu(self.l2(h1))
    h3 = F.relu(self.l3(h2))
    return self.l4(h3)

# In[]
model = L.Classifier(MLP())
gpu = 0
if gpu >= 0:
  import cuda
  xp = cuda.cupy
  cupy.cuda.Device(gpu).use()
  model.to_gpu()
else:
  xp = np
# 数値をChainerが扱える型に変換
X = X.astype(xp.float32)
y = y.astype(xp.int32)
# データをxpの32bit小数の型に変換
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)
batch_size = 100
n_epoch = 20

# In[]
from chainer.datasets import tuple_dataset
from chainer import iterators, training
from chainer.training import extensions
train = tuple_dataset.TupleDataset(X_train, y_train)
train_iter = iterators.SerialIterator(train, batch_size=batch_size, shuffle=True)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (n_epoch,'epoch'), out='result')
test = tuple_dataset.TupleDataset(X_test, y_test)
test_iter = iterators.SerialIterator(test, batch_size=batch_size, shuffle=False, repeat=False)
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch','main/loss','main/accuracy','validation/main/loss','validation/main/accuracy']))
trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'], 'epoch', file_name='loss.png'))
trainer.run()

# In[]
from pylab import box
def show_graph(src):
  img = plt.imread(src)
  xpixels,ypixels = img.shape[0],img.shape[1]
  dpi = 100
  margin = 0.01
  figsize = (1+margin)*ypixels / dpi, (1+margin)*xpixels / dpi

  fig = plt.figure(figsize=figsize, dpi=dpi)
` ax = fig.add_axes([margin,margin,1-2*margin,1-2*margin])
  ax.tick_params(labelbottom='off',bottom='off')
  ax.tick_params(labelleft='off',left='off')

  ax.imshow(img, interpolation='none')
  box('off')
  plt.show()

show_graph('result/loss.png')
show_graph('result/accuracy.png')

# In[]
# モデルを利用して予測をする関数を定義
def predict(model, X):
    # データ数が1の場合は、バッチサイズ分の次元を追加
    if len(X.shape) == 1:
        pred = model.predictor(X[None, ...]).data.argmax()
    # データ数が2以上の場合はそのまま
    else:
        pred = model.predictor(X).data.argmax(axis=1)
    return pred
index = 123
draw_digit(X_test[index])
pred = predict(model, X_test[index])
ans = y_test[index]

print('predict: ', pred)
print('answer :', ans)
if pred==ans:
  print('correct')
else:
  print('incorrect')

# In[]
from sklearn.metrics import confusion_matrix as cm
result = predict(model, X_test)
cm(result, y_test)
# 混同行列をグラフで出力する関数
def plot_cm(y_true, y_pred):
    confmat = cm(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xticks(xp.arange(0, 10, 1)) # x軸の目盛りを指定
    plt.yticks(xp.arange(0, 10, 1)) # y軸の目盛りを指定
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

plot_cm(result, y_test)

# 性能指標を確認
from sklearn import metrics
print('accuracy: %.3f' % metrics.accuracy_score(y_test, predict(model, X_test)))
print('recall: %.3f' % metrics.recall_score(y_test, predict(model, X_test), average='macro'))
print('precision: %.3f' % metrics.precision_score(y_test, predict(model, X_test), average='macro'))
print('f1_score: %.3f' % metrics.f1_score(y_test, predict(model, X_test), average='macro'))

# In[]
# 予想が外れたデータを表示
# 今回は3つだけ表示
count = 0
for i in range(len(y_test)):
    pre = predict(model, X_test[i]) # 予測結果
    ans =  y_test[i]                # 正解

    # 正解が4か9のサンプルについてだけ確認
    if (ans != 9) and (ans != 4):
        continue

    # 予測が間違っていたらリストへ格納
    if pre != ans:
        count += 1
        # 予測を間違えた画像を3枚だけ表示
        if count > 3:
            break
        draw_digit(X_test[i])
        print("正解：{}  予測：{}".format(ans, pre))

# In[]
serializers.save_npz('mnist.model', model)
print('Saved the model.')


serializers.load_npz('mnist.model', model_reloaded)
print('Loaded the model.')
