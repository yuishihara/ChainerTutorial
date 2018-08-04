import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def gaussian_kl_divergence():
    mu_data = np.array([1, 2, 3], dtype=np.float32)
    mu = Variable(mu_data)
    var_data = np.array([1, 4, 9], dtype=np.float32) 
    var = Variable(var_data)

    ln_var = F.log(var)

    dim = len(mu_data)
    xp = cuda.get_array_module(var)
    expected_kld = (xp.trace(xp.diag(var)) + mu_data.dot(mu_data) - dim - xp.sum(ln_var)) * 0.5
    computed_kld = F.gaussian_kl_divergence(mu, ln_var)

    print('expected_kld: ', expected_kld)
    print('computed_kld: ', computed_kld)


def main():
    gaussian_kl_divergence()


if __name__ == '__main__':
    main()
