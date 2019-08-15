from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from IPython import display
import matplotlib.pyplot as plt

def main():
  net = nn.Sqeuential()
  net.add(nn.Conv2D(channels=6,kernel_size=5,activation='relu'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=3,activation='relu'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Flatten(),
        nn.Dense(120,activation='relu'),
        nn.Dense(84,activation='relu'),
        nn.Dense(10))

  net.load_parameters('net.params')

  transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.13,0.31)])

  mnist_valid = datasets.FashionMNIST(train=False)

  X,y = mnist_valid[:10]
  preds=[]

  for x in X:
    x = transformer(x).expand_dims(axis=0)
    pred = net(x).argmax(axis=1)
    preds.append(pred.astype('int32').asscalar())



