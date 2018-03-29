import matplotlib.pyplot as plt
import numpy as np

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
import pickle
obj = 'the object I try saving'

# save
with open('sample.pickle','wb') as f:
  pickle.dump(obj, f)

# load
with open('sample.pickle','rb') as f:
  loaded_obj = pickle.load(f)

print(loaded_obj)

# In[]
def unpickle(file):
  with open(file,'rb') as f:
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
  img = X_test[index].transpose(1,2,0)
  plt.imshow(img)
  plt.show()

  print('class ' + label[y_test[index]])
  print('label :' + str(y_test[index]))

show_test_sample_info(402)

# In[]
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

# In[]
class CNN(chainer.Chain):
  def __init__(self):
    super().__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=4, stride=1, pad=2)
      self.conv2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=4, stride=1, pad=2)
      self.conv3 = L.Convolution2D(in_channels=None, out_channels=128, ksize=4, stride=1, pad=2)
      self.fc4 = L.linear(None, 512)
      self.fc5 = L.linear(None, 5)

  def __call__(self, X):
    h = F.relu(F.max_pooling_2d(self.conv1(X), ksize=2))
    h = F.relu(F.max_pooling_2d(self.conv2(h), ksize=2))
    h = F.relu(F.max_pooling_2d(self.conv3(h), ksize=2))
    h = F.relu(self.fc4(h))
    return self.fc5(h)

# In[]
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from chainer import optimizers, serializers, training, iterators

# In[]
model = L.Classifier(CNN())

optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

batchsize = 100
n_epoch = 100

# In[]
train = tuple_dataset.TupleDataset(X_train, y_train)
train_iter = iterators.SerialIterator(train, batch_size=batchsize,shuffle=True)
updater = training.StandardUpdater(train_iter,optimizer)
trainer = training.Trainer(updater, (n_epoch,'epoch'), out='result')\

# In[]
test = tuple_datset.TupleDataset(X_test, y_test)
test_iter = iterators.SerialIterator(test, batch_size=batchsize, shuffle=False, repeat=False)
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch','main/accuracy','main/loss','validation/main/accuracy','validation/main/loss']))
trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'],
                          'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/loss'],
                          'epoch', file_name='loss.png'))

# In[]
trainer.run()
