import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def distance(instance, target):
    return np.sqrt(np.sum(np.square(instance - target)))


class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num

    def get_knn_num(self, neighbor_num, instances, targets):


        return self.k


if __name__ == '__main__':
    # (1000, 784)1000个实例，图片拉伸为一维784
    n_x = np.loadtxt('mnist_x', delimiter=' ')
    n_y = np.loadtxt('mnist_y')

    x = n_x.reshape(-1, 28, 28)
    plt.figure()
    plt.imshow(x[0], cmap='gray')

    # plt.show()
