import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def links():
    np.random.seed(1)
    f = L.Linear(3, 2)

    # should print 2 x 3 matrix
    print(f.W.data)
    assert(f.W.shape == (2, 3))

    # should print 2 x 1 matrix
    print(f.b.data)
    assert(f.b.shape == (2, ))

    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # NOTE: 2 is the number of mini batches and 3 is input size
    assert(x_data.shape == (2, 3))
    x = Variable(x_data)
    y = f(x)

    # should print 2 x 2 matrix
    print(y.data)
    assert(y.data.shape == (2, 2))

    # Must clear before calculation
    f.cleargrads()

    y.grad = np.ones((2, 2), dtype=np.float32)
    y.backward()

    # should print 2 x 3 matrix
    print(f.W.grad)
    assert(f.W.grad.shape == (2, 3))

    print(f.b.grad)
    assert(f.b.grad.shape == (2, ))


def main():
    links()


if __name__ == '__main__':
    main()
