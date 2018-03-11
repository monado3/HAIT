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
gpu = -1
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
