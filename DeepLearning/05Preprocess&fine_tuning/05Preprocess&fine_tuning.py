import cv2
import numpy as np
import matplotlib.pyplot as plt

# In[]
img = cv2.imread('misc/4.1.04.tiff')
type(img)
img.shape

# In[]
plt.imshow(img)

# In[]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# In[]
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('misc/new_female.jpg', img)

# In[]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img, (224, 224))
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(resized)

# In[]
print(img.shape)
print(resized.shape)

# In[]
cropped_1 = img[50:175,120:220,:]
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(cropped_1)

# In[]
h, w, c = img.shape
cropped_2 = img[:,int(w*(2.5/5)):int(w*(4/5)),:]
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(cropped_2)

# In[]
grayed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(grayed)
plt.gray()

# In[]
th, binary = cv2.threshold(grayed, 125,255,cv2.THRESH_BINARY)
plt.subplot(1,2,1)
plt.imshow(grayed)
plt.subplot(1,2,2)
plt.imshow(binary)
plt.gray()

# In[]
blurred = cv2.GaussianBlur(binary, (11,11), 0)
plt.subplot(1,2,1)
plt.imshow(binary)
plt.subplot(1,2,2)
plt.imshow(blurred)

# In[]
img_2 = cv2.imread('misc/4.1.02.tiff', 0)
plt.imshow(img_2)

# In[]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img_2)

plt.subplot(1,2,1)
plt.imshow(img_2)
plt.subplot(1,2,2)
plt.imshow(cl1)

# In[]
import pickle
with open('../03CNN/train.pickle', 'rb') as f:
  train = pickle.load(f, encoding='bytes')
X_train = train['data']

X_train /= 255
X_train -= np.mean(X_train)

# In[]
flipped = cv2.flip(img, 1)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(flipped)

# In[]
size = (img.shape[0], img.shape[1])
center = (int(size[0]/2), int(size[1]/2))
angle = 30
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, rotation_matrix, size)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(rotated)

# In[]
gamma = 1.5
look_up_table = np.zeros((256,1), dtype='uint8')
for i in range(256):
  look_up_table[i][0] = 255*pow(float(i)/255, 1.0/gamma)
img_gamma = cv2.LUT(img, look_up_table)

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_gamma)

# In[]
row, col, ch = img.shape
mean = 0
sigma = 10
noise = np.random.normal(mean,sigma,(row,col,ch))
noise = noise.reshape(row,col,ch)
noised = img + noise
noised /= 255

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(noised)

# In[]
rows, cols, channels = img.shape
M = np.float32([[1,0,100],[0,1,50]])
moved = cv2.warpAffine(img, M, (cols,rows))
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(moved)

# In[]
zoomed_in = cv2.resize(img, None, fx=2.0, fy=2.0)
height_1,width_1,channel_1 = img.shape
height_2,width_2,channel_2 = zoomed_in.shape
x = int((width_2-width_1)/2)
y = int((height_2-height_1)/2)
zoomed_in = zoomed_in[x:x+width_1, y:y+height_1]
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(zoomed_in)

# In[] fine-tuning
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer.datasets import tuple_dataset,TransformDataset
from chainer.training import extensions
from chainer import optimizers, serializers, training, iterators
# In[]
def unpickle(file):
  with open(file,'rb') as fo:
    list = pickle.load(fo, encoding='bytes')
  return list
# In[]
train = unpickle('../03CNN/train.pickle')
test = unpickle('../03CNN/test.pickle')
label = unpickle('../03CNN/label.pickle')

# In[]
N_train = len(train)
N_test = len(test)
X_train = train['data']
X_test = test['data']
y_train = train['label']
y_test = test['label']

# In[]
class PretrainedVGG16(chainer.Chain):
  def __init__(self, n_class=5, lossfun=F.softmax_cross_entropy, accfun=F.accuracy):
    super(PretrainedVGG16, self).__init__()
    with self.init_scope():
      self.base = L.VGG16Layers()
      self.new_fc8 = L.Linear(None, n_class)
      self.lossfun = lossfun
      self.accfun = accfun

  def __call__(self, x, t):
    with chainer.using_config('enable_backprop',False):
      x = np.asarray(x, dtype=np.float32)

    h = F.relu(self.base(x, layers=['fc7'])['fc7'])
    y = self.new_fc8(h)
    return self.lossfun(y, t)

# In[]
model = PretrainedVGG16(n_class=5)
model.base.disable_update()

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
batchsize=100
n_epoch=1

# In[]
# data augmentationを行う関数
def get_augmented(img, random_crop=4):
  img = img.transpose(1, 2, 0)

  # 左右反転のノイズを加える
  if np.random.rand() > 0.5:
    img = np.fliplr(img)

  # 左右どちらかに30度回転させる
  if np.random.rand() > 0.5:
    size = (img.shape[0], img.shape[1])
    # 画像の中心位置(x, y)
    center = (int(size[0]/2), int(size[1]/2))
    # 回転させたい角度
    angle = np.random.randint(-30, 30)
    # 拡大比率
    scale = 1.0
    # 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # 並進移動
    img = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_CUBIC)

  # BGRをRGBに変換
  img = img.transpose(2, 0, 1)

  return img

# In[]
# 画像の前処理を行う関数
def process_image(inputs):
  image, label = inputs

  # サイズをVGG16指定のものに変換する
  # チャンネルをchainer指定の配置にする
  image = cv2.resize(image.transpose(1, 2, 0), (224, 224)).transpose(2, 0, 1)

  # RGBからそれぞれvgg指定の値を引く(mean-subtractionに相当)
  image[0, :, :] -= 100
  image[1, :, :] -= 116.779
  image[2, :, :] -= 123.68

  # 0-1正規化
  image /= image.max()

  # augmentation
  image = get_augmented(image)

  return image, label

# In[]
# trainerを利用して学習を行う
train_data = tuple_dataset.TupleDataset(X_train, y_train)
test_data = tuple_dataset.TupleDataset(X_test, y_test)

# データセットに前処理を加える
train_data = TransformDataset(train_data, process_image)
test_data = TransformDataset(test_data, process_image)

# In[]
# trainerを利用して学習
train_iter = iterators.SerialIterator(train_data,batch_size=batchsize,shuffle=True)
test_iter = iterators.SerialIterator(test_data,batch_size=1,shuffle=False,repeat=False)
updater = training.StandardUpdater(train_iter,optimizer)
trainer = training.Trainer(updater,(n_epoch,'epoch'),out = 'result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                          'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))

trainer.run()
