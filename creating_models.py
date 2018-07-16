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
        hidden = self.l1(x)
        out = self.l2(hidden)
        return out


def main():
    myChain = MyChain()
    x = Variable(np.ones((2, 4), dtype=np.float32))
    y = myChain(x)

    print(y)


if __name__ == '__main__':
    main()
