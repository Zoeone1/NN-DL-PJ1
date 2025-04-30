# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import data_augment as da
# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 应用数据增强
augmented_train_imgs = []
augmented_train_labs = []
for img, label in zip(train_imgs, train_labs):
    augmented_img = da.augment_image(img)
    augmented_train_imgs.append(augmented_img)
    augmented_train_labs.append(label)

# 合并原始数据和增强数据
#train_imgs = np.vstack([train_imgs, np.array(augmented_train_imgs)])
#train_labs = np.hstack([train_labs, np.array(augmented_train_labs)])

# normalize from [0, 255] to [0, 1]  linear
train_imgs = train_imgs.reshape(-1, 784) / train_imgs.max()
valid_imgs = valid_imgs.reshape(-1, 784) / valid_imgs.max()

# 在 test_train.py 中  conv
#train_imgs = train_imgs.reshape(-1, 1, 28, 28) / train_imgs.max()
#valid_imgs = valid_imgs.reshape(-1, 1, 28, 28) / train_imgs.max()

#train_imgs.shape[-1]的值通常为 28×28 = 784，代表输入图像的像素数量。
#512 和 128 分别是第一个和第二个隐藏层的神经元数量。
#10 是输出层的神经元数量，对应于 10 个类别。

#代码写好之后运行正确率总是0.1左右，完全是随机的。后面发现，是W和b初始化的问题，改了之后就没问题了

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 512, 128, 10], 'ReLU', [1e-4, 1e-4, 1e-4]) 
#linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4]) 
#conv_model = nn.models.Model_CNN(in_channels=1, out_channels1=3, kernel_size=3, fc_size=128, act_func='ReLU')

optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=linear_model, mu=0.95)
#optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)

#scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

#loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
loss_fn = nn.op.CombinedLoss(model=linear_model, max_classes=train_labs.max() + 1, weight_decay_lambda=1e-4)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=5, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()

