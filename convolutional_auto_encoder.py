import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
from chainer.dataset import concat_examples
import matplotlib.pyplot as plt


use_gpu = True
gpu_id = -1
if use_gpu:
    gpu_id = 0


class CAE(Chain):
    def __init__(self):
        super(CAE, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=4, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=4, out_channels=8, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=8, out_channels=16, ksize=4, stride=1)
            self.l4 = L.Linear(None, 128)
            self.l5 = L.Linear(None, 4624)
            self.deconv5 = L.Deconvolution2D(
                in_channels=16, out_channels=8, ksize=4, stride=1)
            self.deconv6 = L.Deconvolution2D(
                in_channels=8, out_channels=4, ksize=5, stride=1)
            self.deconv7 = L.Deconvolution2D(
                in_channels=4, out_channels=1, ksize=5, stride=1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.l4(h))
        h = F.relu(self.l5(h))
        h = F.reshape(h, (-1, 16, 17, 17))
        h = F.relu(self.deconv5(h))
        h = F.relu(self.deconv6(h))
        return F.sigmoid(self.deconv7(h))


def test_model(model, test_iter):
    test_losses = []
    while True:
        test_batch = test_iter.next()
        image_test, _ = concat_examples(test_batch, gpu_id)
        image_test = image_test.reshape(-1, 1, 28, 28)
        prediction_test = model(image_test)

        loss_test = F.mean_squared_error(image_test, prediction_test)
        test_losses.append(cuda.to_cpu(loss_test.data))

        if test_iter.is_new_epoch:
            test_iter.epoch = 0
            test_iter.current_position = 0
            test_iter.is_new_epoch = False
            test_iter._pushed_position = None
            break
    print('val_loss:{:.04f}'.format(np.mean(test_losses)))


def train_model():
    train, test = mnist.get_mnist(withlabel=True, ndim=1)

    x, t = train[0]
    print('train[0] label: ', t)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    # plt.show() # uncomment to show image

    batch_size = 128

    train_iter = iterators.SerialIterator(
        train, batch_size, repeat=True, shuffle=True)
    test_iter = iterators.SerialIterator(
        test, batch_size, repeat=False, shuffle=False)

    model = CAE()

    if use_gpu:
        model.to_gpu(gpu_id)

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    max_epoch = 10
    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        image_train, _ = concat_examples(train_batch, gpu_id)
        image_train = image_train.reshape(-1, 1, 28, 28)
        # print('training image shape: ', image_train.shape)

        prediction_train = model(image_train)

        loss = F.mean_squared_error(image_train, prediction_train)
        # print('loss shape: ', loss.shape)
        model.cleargrads()
        loss.backward()

        optimizer.update()

        if train_iter.is_new_epoch:
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch, float(cuda.to_cpu(loss.data))), end='')
            test_model(model, test_iter)
    # save trained model
    return model


def save_model(path, model):
    serializers.save_npz(path, model)


def evaluate_model(path):
    model = CAE()
    serializers.load_npz(path, model)

    _, test = mnist.get_mnist(withlabel=True, ndim=1)

    x, t = test[10]

    # change the size of minibatch
    x = x.reshape(1, 1, 28, 28)
    y = model(x).data
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)
    plt.subplot(1,2,1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(y.reshape(28, 28), cmap='gray')
    plt.title('decoded')
    plt.show()


def main():
    path = 'cae.model'
    model = train_model()
    save_model(path, model)
    evaluate_model(path)


if __name__ == '__main__':
    main()
