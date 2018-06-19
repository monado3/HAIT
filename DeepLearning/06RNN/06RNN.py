import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# In[]
df = pd.read_csv('./international-airline-passengers.csv')
df.head()
df.tail()
print(df.iloc[144])

# In[]
df.columns = ['Month', 'Passengers']
df = df.iloc[:-1]
df.tail()

# In[]
plt.plot(df['Passengers'])
plt.xticks(np.arange(0,145,12))
plt.grid()
plt.show()

# In[]
from statsmodels.tsa.seasonal import seasonal_decompose
sd = seasonal_decompose(df['Passengers'].values, freq=12)
sd.plot()
plt.show()

# In[]
data = df['Passengers'].values
data = data.astype(np.float32)
scale = data.max()
data /= scale

# In[]
print(data.shape)
# In[]
data = data[:,np.newaxis]
print(data.shape)

# In[]
X = data[:-1]
y = data[1:]
print(f'X: {len(X)}')
print(f'y: {len(y)}')

# In[]
train_size = int(len(data)*0.7)
print(train_size)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
print('X_train:', X_train.shape)
print('X_test :', X_test.shape)
print('y_train:', y_train.shape)
print('y_test :', y_test.shape)

# In[]
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from chainer import datasets, iterators, serializers, training, report, Variable

# In[]
class MyLSTM(chainer.Chain):
  # ネットワークの構造を定義する関数
  # (入力層のノード数, 中間層のノード数, 出力層のノード数)
  def __init__(self, n_input=1, n_units=5, n_output=1):
    super(MyLSTM, self).__init__()
    with self.init_scope():
      self.xh = L.Linear(n_input, n_units) # 入力層
      self.hh = L.LSTM(n_units, n_units) # 中間層（LSTM Block）
      self.hy = L.Linear(n_units, n_output)

  # 中間層の記憶を初期化する関数
  def reset_state(self):
    self.hh.reset_state()

  # 順伝播計算の計算規則を定義する関数
  def __call__(self, x):
    h1 = self.xh(x)
    h2 = self.hh(h1)
    y = self.hy(h2)
    return y

# In[]
class LossFunc(chainer.Chain):
  def __init__(self, predictor):
    super(LossFunc, self).__init__(predictor=predictor)

  def __call__(self, x, t):
    y = self.predictor(x)
    loss = F.mean_squared_error(y, t)
    report({'loss':loss}, self)
    return loss

# In[]
model = LossFunc(MyLSTM())
optimizer = optimizers.Adam()
optimizer.setup(model)
n_epoch = 1000

# In[]
# LSTM用のIterator
class LSTM_test_Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size = 10, seq_len = 5, repeat = True):
        self.seq_length = seq_len
        self.dataset = dataset
        self.nsamples =  len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat
        self.epoch = 0
        self.iteration = 0
        self.offsets = np.random.randint(0, len(dataset),size=batch_size)
        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
            raise StopIteration
        x, t = self.get_data()
        self.iteration += 1
        epoch = self.iteration // self.batch_size
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            self.offsets = np.random.randint(0, self.nsamples,size=self.batch_size)
        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_data(self):
        tmp0 = [self.dataset[(offset + self.iteration)%self.nsamples][0]
               for offset in self.offsets]
        tmp1 = [self.dataset[(offset + self.iteration + 1)%self.nsamples][0]
               for offset in self.offsets]
        return tmp0,tmp1

    def serialzie(self, serialzier):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch     = serializer('epoch', self.epoch)

# LSTM用のUpdater
class LSTM_updater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
        self.seq_length = train_iter.seq_length

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        for i in range(self.seq_length):
            batch = np.array(train_iter.__next__()).astype(np.float32)
            x, t  = batch[:,0].reshape((-1,1)), batch[:,1].reshape((-1,1))
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

# In[]
# TupleDatasetを生成
train = tuple_dataset.TupleDataset(X_train, y_train)
test = tuple_dataset.TupleDataset(X_test, y_test)

# Iteratorを生成
# SerialIteratorではなく､LSTM_test_Iteratorを利用
# seq_lenはミニバッチの系列の長さを表す
train_iter = LSTM_test_Iterator(train, batch_size = 10, seq_len = 10)
test_iter  = LSTM_test_Iterator(test,  batch_size = 10, seq_len = 10, repeat = False)

# Updaterの生成
# StandardUpdaterではなく､LSTM_updaterを利用
updater = LSTM_updater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (n_epoch, 'epoch'), out = 'result')

# Evaluatorの定義
eval_model = model.copy() # 記憶を初期化した別のモデルを複製して検証
eval_rnn = eval_model.predictor # 検証用のモデルで予測値を出力
eval_rnn.train = False # 検証用のモデルは学習する必要がないことを指示
trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=-1,
        eval_hook=lambda _: eval_rnn.reset_state()))

# Adamの学習率を指定
trainer.extend(extensions.ExponentialShift("alpha", 1.00000001))
# 学習ログを出力
trainer.extend(extensions.LogReport())
# 学習ログを画像で出力
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='dl6_loss.png'))

# In[]
trainer.run()

# In[]
from pylab import box
def show_graph(src):
  img =  plt.imread(src)
  xpixels, ypixels = img.shape[0],img.shape[1]
  dpi = 100
  margin = 0.01
  figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

  fig = plt.figure(figsize=figsize, dpi=dpi)
  ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
  ax.tick_params(labelbottom="off",bottom="off")
  ax.tick_params(labelleft="off",left="off")

  ax.imshow(img, interpolation='none')
  box("off")
  plt.show()

# In[]
show_graph('result/dl6_loss.png')

# In[]
# LSTMの記憶をリセットして､系列を順番に読み込ませて予測を行う｡
pred = [] # 予測値を格納するための空リストを定義
model.predictor.reset_state() # 新しく系列を入力させ直すので､記憶を初期化する

# 系列を順番に読み込み､要素をひとつずつ予測
# 予測値はリストに格納
for i in range(len(X)):
  pred.append(model.predictor(X[i].reshape((-1,1)))[0].data[0])
pred = np.array(pred)

# In[]
plt.plot(pred)
plt.plot(data)
plt.xticks(np.arange(0,145,12))
plt.grid()
plt.show()

# In[]
def pred_n_passengers(pred, scale, year, month):
  index = ((year - 1949) * 12) + (month - 1)
  return pred[index] * scale

# In[]
print(pred_n_passengers(pred, scale, 1950, 4))
