from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download 
from mxnet import image

def main():
    net=models.resnet50_v2(pretrained=True)
    url = 'http://data.mxnet.io/models/imagenet/synset.txt'
    fname = download(url)
    with open(fname, 'r') as f:
      text_labels = [' '.join(l.split()[1:]) for l in f]

    url2 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/\
        Golden_Retriever_medium-to-light-coat.jpg/\
        365px-Golden_Retriever_medium-to-light-coat.jpg'
    fname2 = download(url2)
    x = image.imread(fname2)

    x = image.resize_short(x, 256)
    x, _ = image.center_crop(x, (224,224))
    plt.imshow(x.asnumpy())
    plt.show()

    prob = net(transform(x)).softmax()
    idx = prob.topk(k=5)[0]
    for i in idx:
        i = int(i.asscalar())
        print('With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), text_labels[i]))


def transform(data):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std
