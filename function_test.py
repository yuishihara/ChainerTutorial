import unittest
import numpy as np
import chainer
from chainer.backends import cuda
from chainer import testing
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class LinearFunction(Function):
    def forward(self, inputs):
        x, W, b = inputs
        return (x.dot(W.T) + b, )

    def backward(self, inputs, grad_outputs):
        x, W, b = inputs
        gy, = grad_outputs

        gx = gy.dot(W)
        gW = gy.T.dot(x)
        gb = gy.sum(axis=0)

        return (gx, gW, gb)


def linear(x, W, b):
    return LinearFunction()(x, W, b)


class Linear(Link):
    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        with self.init_scope():
            self.W = chainer.Parameter(
                1,
                (out_size, in_size))
            self.b = chainer.Parameter(1, (out_size, ))

    def __call__(self, x):
        return linear(x, self.W, self.b)


class TestLinear(unittest.TestCase):
    def test_backward_cpu(self):
        def f(x):
            W = chainer.Parameter(
                1,
                (2, 2))
            b = chainer.Parameter(1, (2, ))
            return linear(x, W, b)
        x = np.random.randn(3, 2).astype(np.float32)
        y_grad = np.random.rand(3, 2).astype(np.float32)

        gradient_check.check_backward(f, x, y_grad, atol=1e-4, rtol=1e-4)


def main():
    tester = TestLinear()
    tester.test_backward_cpu()


if __name__ == '__main__':
    main()
