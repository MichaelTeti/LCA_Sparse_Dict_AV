from __future__ import division, print_function, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import h5py
import glob
import sys, os
from torchvision.utils import make_grid
from torchvision.transforms import Pad
from sklearn.preprocessing import scale
from scipy.misc import imresize, bytescale
from skimage.util import view_as_windows as vaw
import time
import cv2


# define useful variables
iters = 120 # number of training iterations
batch_sz = 20  # training batch size
k = 500
patch_sz = 21

fnames = glob.glob('*.h5')


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
    zs=[X.shape[1], int(np.sqrt(X.shape[0]//3)), int(np.sqrt(X.shape[0]//3)), 3]
    Z=np.zeros(zs)

    for i in range(X.shape[1]):
        Z[i, ...] = X[:,i].reshape([zs[1],zs[2], 3])

    return Z


def montage(X):
    count, m, n, c = np.shape(X)
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n, c))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n, :] = bytescale(X[image_id, ...])
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
    return X



def whiten(X):
    '''Function to ZCA whiten image matrix.'''
    b, c, h, w = X.shape
    X = X.view(-1, c*h*w)
    U,S,V = torch.svd(torch.t(torch.mm(X, torch.t(X))))
    epsilon = 1e-5
    ZCAMatrix = torch.mm(U, torch.mm(torch.diag(1.0/torch.sqrt(S + epsilon)),
                                                torch.t(U)))
    return torch.mm(ZCAMatrix, X).view(b, c, h, w)



def X3(y, iters, batch_sz, num_dict_features=None, D=None):
    ''' Dynamical systems neural network used for sparse approximation of an
        input vector.

        Args:
            y: input signal or vector, or multiple column vectors.
            num_dict_features: number of dictionary patches to learn.
            iters: number of LCA iterations.
            batch_sz: number of samples to send to the network at each iteration.
            D: The dictionary to be used in the network.'''

    D = torch.randn(k, 1, patch_sz, patch_sz).float().cuda(0)
    D = Variable(D, requires_grad=False, volatile=True)

    for i in range(iters):
        # choose random examples this iteration
        x = y[np.random.randint(0, y.shape[0], batch_sz), ...]
        x = torch.from_numpy(x).float().cuda(0)
        x = whiten(x[:, None, ...])
        x = Variable(x, requires_grad=False, volatile=True)
        x2 = x[batch_sz//2:, ...].cuda(1)
        x = x[:batch_sz//2, ...]

        D = D*(1./(1e-5+torch.sqrt(torch.sum(
                  torch.sum(
                  torch.sum(D**2, -1), -1),
                  -1)[:, None, None, None].expand(D.size(0),
                  D.size(1), D.size(2), D.size(3)))))
        D2 = D.cuda(1)

        a = f.conv2d(x, D, padding=patch_sz//2).cuda(0)
        a2 = f.conv2d(x2, D2, padding=patch_sz//2).cuda(1)

        a = a*(1./(1e-5+torch.sqrt(torch.sum(a**2, 1)[:, None, :, :].expand(a.size(0),
                                             a.size(1), a.size(2), a.size(3)))))
        a2 = a2*(1./(1e-5+torch.sqrt(torch.sum(a2**2, 1)[:, None, :, :].expand(a2.size(0),
                                             a2.size(1), a2.size(2), a2.size(3)))))
        a = 0.1 * a ** 3
        a2 = 0.09 * a2 **3

        ashow = a[:, np.random.randint(0, a.size(1), 100), ...]
        cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
        cv2.imshow('dictionary',
                        montage(ashow.view(ashow.size(0)*ashow.size(1),
                        a.size(2), a.size(3), 1).data.cpu().numpy()))
        cv2.waitKey(1)

        x = f.conv_transpose2d(a, D, padding=patch_sz//2).double().cuda(0) - x.double()
        x2 = f.conv_transpose2d(a2, D2, padding=patch_sz//2).double().cuda(1) - x2.double()

        a = torch.cat((a, a2), 0).cuda(0)

        x = f.pad(x, (patch_sz//2, patch_sz//2, patch_sz//2, patch_sz//2))
        x = x.unfold(1, 1, 1).unfold(2, patch_sz, 1).unfold(3, patch_sz, 1)
        x2 = f.pad(x2, (patch_sz//2, patch_sz//2, patch_sz//2, patch_sz//2))
        x2 = x2.unfold(1, 1, 1).unfold(2, patch_sz, 1).unfold(3, patch_sz, 1)

        x = torch.cat((x, x2)).cuda(0)
        x = x.contiguous().view(-1, x.size(4)*x.size(5)*x.size(6))

        a = a.permute(1, 0, 2, 3).contiguous().view(-1,
                                               a.size(0)*a.size(2)*a.size(3))

        D = D + torch.mm(a, x.float()).view(k, 1, patch_sz, patch_sz)

        #cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
        #cv2.imshow('dictionary', montage(np.transpose(D.data.cpu().numpy(), (0, 2, 3, 1))))
        #cv2.waitKey(1)

    return D.data.cpu().numpy(), a.data.cpu().numpy()



X = file_get(fnames[0])
from progressbar import ProgressBar
Bar = ProgressBar()
for i in Bar(range(X.shape[0])):
    X[i, :240//3, :320//3, :] = imresize(X[i, ...], [240//3, 320//3])
X = X[:, :240//3, :320//3, :]
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / (sigma + 1e-6)
X = np.mean(X, 3)

d, a = X3(X, iters, batch_sz, k)


#x = torch.ones(4, 3, 6, 7).cuda()
#d = torch.zeros(10, 3, 2, 7).cuda()
#a = f.conv2d(Variable(x), Variable(d)).cuda()
