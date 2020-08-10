#!/usr/bin/env python
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import time
import argparse
from utils.model import LeNet5
from utils.layers import CrossEntropyLoss

from utils import config
from tqdm import tqdm
import time
import struct
import math
import random
from abc import ABCMeta, abstractmethod

import os
# read the images and labels
def readDataset(dataset):
    (image, label) = dataset
    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (img, lbl)


# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,), (pad,), (pad,), (0,)), 'constant', constant_values=(0, 0))
    return X_pad

def normalize(image):
    image -= image.min()
    image = image / image.max()
    image = (image-np.mean(image))/np.std(image)
    return image

def get_parser():
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

# initialization of the weights & bias
def initialize(kernel_shape):
    b_shape = (1, 1, 1, kernel_shape[-1]) if len(kernel_shape) == 4 else (kernel_shape[-1],)
    mu, sigma = 0, 0.1
    weight = np.random.normal(mu, sigma, kernel_shape)
    bias = np.ones(b_shape) * 0.01
    return weight, bias

def load_dataset(test_image_path, test_label_path, train_image_path, train_label_path):
    trainset = (train_image_path, train_label_path)
    testset = (test_image_path, test_label_path)

    # read data
    (train_image, train_label) = readDataset(trainset)
    (test_image, test_label) = readDataset(testset)

    # data preprocessing
    train_image_normalized_pad = normalize(zero_pad(train_image[:, :, :, np.newaxis], 2))
    test_image_normalized_pad = normalize(zero_pad(test_image[:, :, :, np.newaxis], 2))

    return (train_image_normalized_pad, train_label), (test_image_normalized_pad, test_label)

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_minibatch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def update_weights(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""
model = LeNet5()
print("model = lenet-5")

batch_size = 64
D_in = 784
D_out = 10
print("Begining training with batch size " ,  batch_size)

#mnist.init()
# X_train, Y_train, X_test, Y_test = load()
global args
args = get_parser()

train_data, test_data = load_dataset(args.test_image_path, args.test_label_path, args.train_image_path,
                                         args.train_label_path)

X_train, Y_train = train_data[0], train_data[1]
X_test, Y_test = test_data[0], test_data[1]
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

losses = []
GD = SGD(model.get_params(), lr=0.0001, reg=0)
Loss_func = CrossEntropyLoss()

lr_global_list = np.array([5e-2] * 2 + [2e-2] * 3 + [1e-2] * 3 + [5e-3] * 4 + [1e-3] * 8)
batches =2000

for epoch in range(4):
    cost = 0
    print("---------- epoch", epoch + 1, "start ------------")
    # GD.lr = lr_global_list[epoch]
    for i in range(batches):
        # get batch, make onehot
        X_batch, Y_batch = get_minibatch(X_train, Y_train, batch_size)
        Y_batch = MakeOneHot(Y_batch, D_out)

        # forward, loss, backward, step
        Y_pred = model.forward(X_batch)
        loss, dout = Loss_func.get(Y_pred, Y_batch)
        cost+= loss
        model.backward(dout)
        GD.update_weights()

        if i % 500 == 0:
                print("Batch no", i, " Loss =", loss )
                losses.append(loss)
    with open('model_data_' + str(epoch) + '.pkl', 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    print("Done, total cost of epoch {}: {}".format(epoch + 1, cost))
    print("---------- epoch", epoch + 1, "end ------------")
# save params

data = np.arange(len(losses))
plt.plot(data, losses)
plt.show()

X_train_min, Y_train_min = get_batch(X_train, Y_train, 100)

#train accuracy
Y_pred_min = model.forward(X_train_min)
result = np.argmax(Y_pred_min, axis=1) - Y_train_min
result = list(result)
print("Train accuracy:"   + str(result.count(0)/X_train_min.shape[0]))

#test accuracy
Y_pred= model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("Test accuracy" +  str(result.count(0)/X_test.shape[0]))