import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class MulAdd(Function):
    def forward_cpu(self, inputs):
        print('using cpu!!')
        x, y, z = inputs
        w = x * y + z
        return (w, )

    def backward_cpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y * gw
        gy = x * gw
        gz = gw
        return (gx, gy, gz)

    def forward_gpu(self, inputs):
        print('using gpu!!')
        x, y, z = inputs
        w = x * y + z
        return (w, )

    def backward_gpu(self, inputs, grad_outputs):
        x, y, z = inputs
        gw, = grad_outputs

        gx = y * gw
        gy = x * gw
        gz = gw
        return (gx, gy, gz)


def muladd(x, y, z):
    return MulAdd()(x, y, z)


def differentiable_functions_cpu():
    print('will use cpu')
    x = Variable(np.ones((3, 2)))
    y = Variable(np.ones((3, 2)))
    z = Variable(np.ones((3, 2)))
    w = muladd(x, y, z)

    print(w)


def differentiable_functions_gpu():
    print('will use gpu')
    x = Variable(np.ones((3, 2)))
    y = Variable(np.ones((3, 2)))
    z = Variable(np.ones((3, 2)))
    x.to_gpu()
    y.to_gpu()
    z.to_gpu()
    w = muladd(x, y, z)

    print(w)


class ExpAdd(Function):
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        z = xp.exp(x) + xp.exp(y)
        return (z, )

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, y = inputs
        gz, = grad_outputs
        gx = gz * xp.exp(x)
        gy = gz * xp.exp(y)
        return (gx, gy)


def expadd(x, y):
    return ExpAdd()(x, y)


def unified_backward_forward_method_cpu():
    print('unified: will use cpu')
    x = Variable(np.ones((3, 2)))
    y = Variable(np.ones((3, 2)))
    w = expadd(x, y)

    print(w)


def unified_backward_forward_method_gpu():
    print('unified: will use gpu')
    x = Variable(np.ones((3, 2)))
    y = Variable(np.ones((3, 2)))
    x.to_gpu()
    y.to_gpu()
    w = expadd(x, y)

    print(w)


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


def links_and_wrap_functions():
    # (1: stands for the num of batches, 2: dimension of data)
    x_data = np.array([[1, 1], [2, 2]], dtype=np.float32)
    x = Variable(x_data)
    f = Linear(in_size=2, out_size=2)
    y = f(x)

    print('x: ' + str(x_data))
    print('W shape: ' + str(f.W.shape))
    print('W: ' + str(f.W))
    print('b shape: ' + str(f.b.shape))
    print('b: ' + str(f.b))
    print(y)


def main():
    differentiable_functions_cpu()
    differentiable_functions_gpu()
    unified_backward_forward_method_cpu()
    unified_backward_forward_method_gpu()
    links_and_wrap_functions()


if __name__ == '__main__':
    main()
