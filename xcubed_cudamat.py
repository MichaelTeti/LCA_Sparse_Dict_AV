from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import h5py
import glob
import sys, os
from sklearn.preprocessing import scale
from skimage.util import view_as_windows as vaw
from numpy import random as npr
from scipy.misc import imresize, bytescale
import time
import cudamat as cm
import cv2


# define useful variables
iters = 300 # number of training iterations
batch_sz = 1500  # training batch size
k = 1000
patch_sz = 15
imsz = 100



def plot2(a, e):
    # plot a histogram of the sparse coefficients
    # and the error rate over time
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    a1.plot(e[1:, 0])
    a1.set_ylabel('l2 error')
    a1.set_xlabel('Iteration')
    a2 = fig.add_subplot(122)
    a2.hist(a)
    a2.set_ylabel('Number of coefficients')
    a2.set_xlabel('Activation')
    plt.show()
    return


def mat2ten(X):
    zs=[X.shape[1], int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))]
    Z=np.zeros(zs)

    for i in range(X.shape[1]):
        Z[i, ...]=np.reshape(X[:,i],[zs[1],zs[2]])

    return Z


def montage(X):
    count, m, n = np.shape(X)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceN:sliceN + n, sliceM:sliceM + m] = bytescale(X[image_id, ...])
            image_id += 1

    return np.uint8(M)


def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(18, 18)
    plt.show()



def file_get(filename):
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    f.flush()
    f.close()
    return np.mean(X, 3)



def whiten(X):
    '''Function to ZCA whiten image matrix.'''

    sigma = np.cov(X, rowvar=True) # [M x M]
    U,S,V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return np.dot(ZCAMatrix, X)



def X3(y, iters, batch_sz, num_dict_features=None, D=None, white=False, scale=False):
    ''' Dynamical systems neural network used for sparse approximation of an
        input vector.

        Args:
            y: input signal or vector, or multiple column vectors.
            num_dict_features: number of dictionary patches to learn.
            iters: number of LCA iterations.
            batch_sz: number of samples to send to the network at each iteration.
            D: The dictionary to be used in the network.'''

    cm.cuda_set_device(0)
    cm.cuda_set_device(1)
    cm.init()

    e = np.zeros([iters, 1])

    r = np.random.permutation(y.shape[1])
    D = np.random.randn(y.shape[0], num_dict_features)
    D = cm.CUDAMatrix(D)
    a = cm.empty([num_dict_features, batch_sz])
    error = cm.empty([y.shape[0], batch_sz])
    AS = cm.empty([num_dict_features, batch_sz])
    DS = cm.empty([y.shape[0], num_dict_features])
    AS2 = cm.empty([1, batch_sz])
    DS2 = cm.empty([1, num_dict_features])

    for i in range(iters):

        # choose random examples this iteration
        batch = y[:, np.random.randint(0, y.shape[1], batch_sz)]
        
        if scale:
            batch = scale(batch, 1)
        if white:
            batch = whiten(batch)

        batch = cm.CUDAMatrix(batch)

        # scale the values in the dict to between 0 and 1
        cm.pow(D, 2, target=DS)
        cm.sum(DS, 0, target=DS2)
        cm.pow(cm.sqrt(DS2.add(1e-6)), -1)
        D.mult_by_row(DS2)

        # get similarity between each feature and each data patch
        cm.dot(D.T, batch, target=a)

        # scale the alpha coefficients (cosine similarity coefficients)
        cm.pow(a, 2, target=AS)
        cm.sum(AS, 0, target=AS2)
        cm.pow(cm.sqrt(AS2.add(1e-6)), -1)
        a.mult_by_row(AS2)

        # perform cubic activation on the alphas
        cm.pow(a, 3)
        a.mult(0.3) # learning rate

        # get the SSE between reconstruction and data batch
        error = batch.subtract(cm.dot(D, a))
        # save the error to plot later
        e[i, 0] = np.mean(error.asarray()**2)
        # modify the dictionary to reduce the error
        D.add(cm.dot(error, a.T))

        cv2.namedWindow('Dictionary', cv2.WINDOW_NORMAL)
        cv2.imshow('Dictionary', montage(mat2ten(D.asarray())))
        cv2.waitKey(1)

    fig = plt.figure(figsize=(16, 16))
    a1 = fig.add_subplot(121)
    a2 = fig.add_subplot(122)
    a1.imshow(montage(mat2ten(batch.asarray())), cmap='gray')
    a2.imshow(montage(mat2ten(cm.dot(D, a).asarray())), cmap='gray')
    plt.show()


    cm.pow(a, 2, target=AS)
    cm.sum(AS, 0, target=AS2)
    cm.pow(cm.sqrt(AS2.add(1e-6)), -1)
    a.mult_by_row(AS2)

    plt.plot(np.sort(np.absolute(a.asarray()[:, 0]))[::-1])
    plt.show()

    cm.shutdown()

    return D.asarray(), a.asarray(), e



f = h5py.File('rover_patches.h5', 'r')
X = np.asarray(f['X'])
print(X.shape)

Dict, alpha, error = X3(X, iters, batch_sz, k, white=True)
