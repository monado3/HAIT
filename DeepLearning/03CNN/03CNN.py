import pickle

# In[]
import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
from chainer import iterators, optimizers, serializers, training
# In[]
from chainer.datasets import tuple_dataset
from chainer.training import extensions

'''
# In[]
from urllib import request
url_train = 'https://s3-ap-northeast-1.amazonaws.com/ai-std/train.pickle'
request.urlretrieve(url_train, 'train.pickle')

url_test =  'https://s3-ap-northeast-1.amazonaws.com/ai-std/test.pickle'
request.urlretrieve(url_test, 'test.pickle')

url = 'https://s3-ap-northeast-1.amazonaws.com/ai-std/label.pickle'
request.urlretrieve(url, 'label.pickle')
'''

# In[] practice pickle
obj = 'the object I try saving'

# save
with open('sample.pickle', 'wb') as f:
  pickle.dump(obj, f)

# load
with open('sample.pickle', 'rb') as f:
  loaded_obj = pickle.load(f)

print(loaded_obj)

# In[]


def unpickle(file):
  with open(file, 'rb') as f:
    return pickle.load(f, encoding='bytes')


# In[]
train = unpickle('train.pickle')
test = unpickle('test.pickle')
label = unpickle('label.pickle')

# In[]
for obj in train, test, label:
  print(type(obj))

# In[]
label
X_train = train['data']
y_train = train['label']
X_test = test['data']
y_test = test['label']
N_train = len(X_train)
N_test = len(X_test)
print(N_train, N_test)

# In[]
print(train['data'].shape)
print(test['data'].shape)

# In[]
X_train /= 255.0
X_test /= 255.0

# In[]


def show_test_sample_info(index):
  img = X_test[index].transpose(1, 2, 0)
  plt.imshow(img)
  plt.show()

  print('class ' + label[y_test[index]])
  print('label :' + str(y_test[index]))


show_test_sample_info(402)


# In[]


class CNN(chainer.Chain):
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(
          in_channels=3, out_channels=64, ksize=4, stride=1, pad=2)
      self.conv2 = L.Convolution2D(
          in_channels=64, out_channels=128, ksize=4, stride=1, pad=2)
      self.conv3 = L.Convolution2D(
          in_channels=None, out_channels=128, ksize=4, stride=1, pad=2)
      self.fc4 = L.Linear(None, 512)
      self.fc5 = L.Linear(None, 5)

  def __call__(self, X):
    h = F.relu(F.max_pooling_2d(self.conv1(X), ksize=2))
    h = F.relu(F.max_pooling_2d(self.conv2(h), ksize=2))
    h = F.relu(F.max_pooling_2d(self.conv3(h), ksize=2))
    h = F.relu(self.fc4(h))
    return self.fc5(h)



# In[]
model = L.Classifier(CNN())

optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

batchsize = 100
n_epoch = 100

# In[]
train = tuple_dataset.TupleDataset(X_train, y_train)
train_iter = iterators.SerialIterator(
    train, batch_size=batchsize, shuffle=True)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')\

# In[]
test = tuple_dataset.TupleDataset(X_test, y_test)
test_iter = iterators.SerialIterator(
    test, batch_size=batchsize, shuffle=False, repeat=False)
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                     'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/loss'],
                                     'epoch', file_name='loss.png'))
# line141, main/accuracy -> main/loss

# In[]
trainer.run()

# In[] save
serializers.save_npz('mnist.model', model)

# In[] load
# serializers.load_npz('mnist.model', model_reloaded)

# In[]
# モデルを利用して予測をする関数を定義
def predict(model, X):
  if len(X.shape) == 3:  # データ数が1の場合は、バッチサイズ分の次元を追加
    pred = model.predictor(X[None, ...]).data.argmax()
  else:  # データ数が2以上の場合はそのまま
    pred = model.predictor(X).data.argmax(axis=1)
  return pred

# In[]
from sklearn import metrics
print('accuracy: %.3f' % metrics.accuracy_score(y_test, predict(model, X_test)))
print('recall: %.3f' % metrics.recall_score(
    y_test, predict(model, X_test), average='macro'))
print('precision: %.3f' % metrics.precision_score(
    y_test, predict(model, X_test), average='macro'))
print('f1_score: %.3f' % metrics.f1_score(
    y_test, predict(model, X_test), average='macro'))

# In[]
from pylab import box
def show_graph(src):
  img = plt.imread(src)
  xpixels, ypixels = img.shape[0], img.shape[1]
  dpi = 100
  margin = 0.01
  figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi

  fig = plt.figure(figsize=figsize, dpi=dpi)
  ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
  ax.tick_params(labelbottom="off", bottom="off")
  ax.tick_params(labelleft="off", left="off")

  ax.imshow(img, interpolation='none')
  box("off")
  plt.show()


# In[]
# 精度と誤差をグラフ描画
show_graph('result/loss.png')
show_graph('result/accuracy.png')

# In[]
# indexを指定
index = 1

# 画像を出力
show_test_sample_info(index)

# 指定のindexが与えられたtestデータについて確認
pred = predict(model, X_test[index])
print('predict: {}'.format(pred))

# 正解か不正解かを出力
if pred == y_test[index]:
    print('正解です｡')
else:
    print('間違いです｡')

# In[]
from sklearn.metrics import confusion_matrix as cm

# 混同行列きれいに出力する関数
def plot_cm(y_true, y_pred):
  confmat = cm(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
  for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
      ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
  plt.xticks(np.arange(0, 5, 1))                               # x軸の目盛りを指定
  plt.yticks(np.arange(0, 5, 1))
  plt.xlabel('true label')
  plt.ylabel('predicted label')
  plt.show()

# 混同行列を出力
result = predict(model, X_test)
plot_cm(result, y_test)

# In[]
# ラベルを確認
print(label)

# In[]
errors = []

for i in range(len(y_test)):
  pred_1 = predict(model, X_test[i])
  if pred_1 != y_test[i]:
    errors.append((i, label[y_test[i]], label[pred_1]))

len(errors)

# In[]
for error_index, corr_label, pred_label in errors[:3]:
  show_test_sample_info(error_index)
  print(corr_label)
  print(pred_label)
  
