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


class MLP(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def test_model(model, test_iter):
    test_losses = []
    test_accuracies = []
    while True:
        test_batch = test_iter.next()
        image_test, target_test = concat_examples(test_batch, gpu_id)
        prediction_test = model(image_test)

        loss_test = F.softmax_cross_entropy(
            prediction_test, target_test)
        test_losses.append(cuda.to_cpu(loss_test.data))

        accuracy = F.accuracy(prediction_test, target_test)
        test_accuracies.append(cuda.to_cpu(accuracy.data))

        if test_iter.is_new_epoch:
            test_iter.epoch = 0
            test_iter.current_position = 0
            test_iter.is_new_epoch = False
            test_iter._pushed_position = None
            break
    print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
        np.mean(test_losses), np.mean(test_accuracies)))


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

    model = MLP()

    if use_gpu:
        model.to_gpu(gpu_id)

    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    max_epoch = 10
    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, gpu_id)

        prediction_train = model(image_train)

        loss = F.softmax_cross_entropy(prediction_train, target_train)

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
    model = MLP()
    serializers.load_npz(path, model)

    _, test = mnist.get_mnist(withlabel=True, ndim=1)

    x, t = test[1]

    # change the size of minibatch
    x = x[None, :]
    y = model(x)
    prediction = np.argmax(y.data, axis=1)

    print('prediction: ', prediction)
    print('label: ', t)

    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()

def main():
    path = 'manual_mnist.model'
    # model = train_model()
    # save_model(path, model)
    evaluate_model(path)



if __name__ == '__main__':
    main()
