from __future__ import print_function # only relevant for Python 2
import mxnet as mx
from mxnet import nd,gluon,autograd
from mxnet.gluon import nn

def get_mislabeled(loader):
    mislabeled = []
    
    for inputs,labels in loader:
        inputs = inputs.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        outputs = net(inputs)

        preds = nd.argmax(output,axis=1)
        for i, p, l in zip(inputs,preds,labels):
            p,l = int(p.asscalar()),int(l.asscalar())
            if p != l:
               mislabeled.append((i.asnumpy(),p,l)
    
    return mislabeled


def data_xfrom(data):
   return nd.moveaxis(data,2,0).astype('float32')/255


def main():
  mx.random.seed(42) 
  train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
   val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)

  batch_size=100
  train_loader = mx.gluon.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)

  val_loader = mx.gluon.data.DataLoader(val_data,shuffle=False,batch_size=batch_size)
  net = nn.HybridSequential(prefix='MLP_')
  with net.name_scope():
      net.add(nn.Flatten(),
              nn.Dense(128, activation='relu'),
              nn.Dense(64, activation='relu'),
              nn.Dense(10, activation=None))  
            # loss function includes softmax already, see below

  ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
  net.initialize(mx.init.Xavier(), ctx=ctx)
  trainer = gluon.Trainer(params=net.collect_params(),optimizer='sgd',optimizer_params={'learning_rate':0.04})

  metric = mx.metric.Accuracy()
  loss_function = gluon.loss.SoftmaxCrossEntropyLoss()

  """
  steps in each epoch:
  1. get a minibatch of inputs and labels from the train_loader 
  2. feed the inputs to the network producing outputs
  3. compute minibatch loss value by comparing outputs to labels
  4. use backpropagation to compute the gradients of the loss with respect ot each of the network parameters by calling loss.backward()
  5. update the parameters of the network according to the optimizer rule with   trainer.step(batch_size=inputs.shape[0])
  6. print the current accuracy over the training data i.e. the fraction of correctly classified training examples
  """

# this is training
  num_epochs=10
  for epoch in range(num_epochs):
      for inputs,labels in train_loader:
          inputs = inputs.as_in_context(ctx)
          labels = labels.as_in_context(ctx)
          with autograd.record():
              outputs = net(inputs)
              loss = loss_function(outputs,labels)
          loss.backward()
          metric.update(labels,outputs)
          trainer.step(batch_size=inputs.shape[0])
   name, acc = metric.get()
   print('After epoch {}: {} = {}'.format(epoch+1,name,acc))
   metric.reset()

# validation data
  metric = mx.metric.Accuracy()
  for inputs,labels in val_loader:
      #Possibly copy inputs and labels to the GPU
      inputs = inputs.as_in_contect(ctx)
      labels = labels.as_in_contect(ctx)
      metric.update(labels, net(inputs))
  print('Validation: {} = {}'.format(*metric.get()))
  assert metric.get()[1]>0.96




