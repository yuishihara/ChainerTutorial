import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)


def run_optimizer():
    np.random.seed(1)

    model = MyChain()
    learning_rate = 0.1
    optimizer = optimizers.SGD(lr=learning_rate).setup(model)

    print('Weight before optimization' + str(model.l1.W))

    x = np.random.uniform(-1, 1, (2, 4)).astype(np.float32)
    model.cleargrads()
    loss = F.sum(model(chainer.Variable(x)))
    loss.backward()
    optimizer.update()

    print('Weight after optimization' + str(model.l1.W))


def run_optimizer_simple():
    np.random.seed(1)

    model = MyChain()
    learning_rate = 0.1
    optimizer = optimizers.SGD(lr=learning_rate).setup(model)

    print('Weight before optimization' + str(model.l1.W))

    def loss_function(output, training_data):
        print('output: ' + str(output))
        print('training_data: ' + str(training_data))
        losses = F.loss.squared_error.squared_error(output, training_data)
        print('losses: ' + str(losses))
        return F.sum(losses)

    arg1 = np.random.uniform(-1, 1, (2, 4)).astype(np.float32)
    arg2 = np.random.uniform(-1, 1, (2, 2)).astype(np.float32)
#    arg2 = np.array([[0.193665, -1.004596], [-0.4661492, 0.7400142]], dtype = np.float32)
    output = model(chainer.Variable(arg1))
    training_data = chainer.Variable(arg2)

    print('output of model: ' + str(output))

    optimizer.update(loss_function, output, training_data)
    print('Weight after optimization' + str(model.l1.W))


def main():
    run_optimizer()
    run_optimizer_simple()


if __name__ == '__main__':
    main()
