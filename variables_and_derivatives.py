import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def variables_and_derivatives_scalar():
    x_data = np.array([5], dtype=np.float32)
    x = Variable(x_data)
    y = x**2 - 2 * x + 1

    # should print [16.]
    print(y.data)
    # dy/dx should print [8.]
    print(x.grad)

    z = 2 * x
    y = x**2 - z + 1
    y.backward(retain_grad=True)

    # dy/dz should print [-1.]
    print(z.grad)


def variables_and_derivatives_matrix_cpu():
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Variable(x_data)
    y = x**2 - 2*x + 1
    y.grad = np.ones((2, 3), dtype=np.float32)

    # should print [[0, 1, 4][9, 16, 25]]
    print(y)

    y.backward()
    # dy/dx should print [[0, 2, 4][6, 8, 10]]
    print(x.grad)


def variables_and_derivatives_matrix_gpu():
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x = Variable(x_data)
    y = x**2 - 2*x + 1
    y.grad = np.ones((2, 3), dtype=np.float32)
    x.to_gpu()
    y.to_gpu()

    # should print [[0, 1, 4][9, 16, 25]]
    print(y)

    y.backward()
    # dy/dx should print [[0, 2, 4][6, 8, 10]]
    print(x.grad)


def main():
    variables_and_derivatives_scalar()
    variables_and_derivatives_matrix_cpu()
    variables_and_derivatives_matrix_gpu()


if __name__ == '__main__':
    main()
