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

import matplotlib.pyplot as plt

use_gpu = True


class MLP(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def train_model():
    train, test = mnist.get_mnist()
    batchsize = 128

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    model = MLP()

    gpu_id = -1
    if use_gpu:
        gpu_id = 0
        model.to_gpu(gpu_id)

    max_epoch = 10

    # Wrap your model by Classifier and include the process of loss calculation within your model.
    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
    model = L.Classifier(model)

    optimizer = optimizers.MomentumSGD()
    optimizer.setup(model)
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(
        updater, (max_epoch, 'epoch'), out='mnist_result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


def evaluate_model():
    _, test = mnist.get_mnist()

    model = MLP()
    serializers.load_npz('mnist_result/model_epoch-10', model)

    image, label = test[0]

    print('image shape: ', image.shape)
    y = model(image[None, :])

    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

    print('label: ', label)
    print('predicted_label:', y.data.argmax(axis=1)[0])


def main():
    train_model()
    evaluate_model()


if __name__ == '__main__':
    main()
