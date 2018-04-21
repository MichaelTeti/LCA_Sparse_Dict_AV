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
from progressbar import ProgressBar

# define useful variables
patch_sz = 15

fnames = glob.glob('*.h5')


def mat2ten(X, c=3):
    zs=[X.shape[1], int(np.sqrt(X.shape[0]//c)), int(np.sqrt(X.shape[0]//c)), c]
    Z=np.zeros(zs)

    for i in range(X.shape[1]):
        Z[i, ...] = X[:,i].reshape([zs[1],zs[2], c])

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
    U,S,V = torch.svd(torch.mm(X, torch.t(X)))
    epsilon = 1e-5
    ZCAMatrix = torch.mm(U, torch.mm(torch.diag(1.0/torch.sqrt(S + epsilon)),
                        torch.t(U)))

    return torch.mm(ZCAMatrix, X)



def X3(y, iters, batch_sz, num_dict_features, show=False):
    ''' Dynamical systems neural network used for sparse approximation of an
        input vector.

        Args:
            y: input signal or vector, or multiple column vectors.
            num_dict_features: number of dictionary patches to learn.
            iters: number of LCA iterations.
            batch_sz: number of samples to send to the network at each iteration.
            D: The dictionary to be used in the network.'''

    D = torch.randn(3*patch_sz**2, num_dict_features).float().cuda(0)
    D2 = D.cuda(1)
    #D = Variable(D, requires_grad=False, volatile=True)

    D_2 = torch.randn(3*patch_sz**2, 100).float().cuda(0)
    D2_2 = D_2.cuda(1)
    #D_2 = Variable(D_2, requires_grad=False, volatile=True)

    start = time.time()
    for i in range(iters):
        x = y[np.random.randint(0, y.shape[0], 1)[0], ...]
        x = torch.from_numpy(x).float().cuda(0)
        x = x.unfold(0, patch_sz, 1).unfold(1, patch_sz, 1).unfold(2, 3, 1)
        x = x.contiguous().view(x.size(0)*x.size(1)*x.size(2),
                                x.size(3)*x.size(4), x.size(-1))
        x = x - torch.mean(x, 0)
        x = torch.t(x.view(-1, x.size(1)*3))

        x = whiten(x)
        x2 = x[:, x.size(1)//2:].cuda(1)
        x = x[:, :x.size(1)//2]

        D = torch.mm(D, torch.diag(1./(torch.sqrt(torch.sum(D**2, 0))+1e-6)))
        D2 = D.cuda(1)

        a = torch.mm(torch.t(D), x).cuda(0)
        a2 = torch.mm(torch.t(D2), x2).cuda(1)

        a = torch.mm(a, torch.diag(1./(torch.sqrt(torch.sum(a**2, 0))+1e-6)))
        a2 = torch.mm(a2, torch.diag(1./(torch.sqrt(torch.sum(a2**2, 0))+1e-6)))

        a = .5 * a ** 3
        a2 = .5 * a2 ** 3

        x = x - torch.mm(D, a)
        x2 = x2 - torch.mm(D2, a2)

        # x = torch.cat((x, x2), 1).cuda(0)
        # a = torch.cat((a, a2), 1).cuda(0)

        D = D + torch.mm(x, torch.t(a))
        D = (D + (D2 + torch.mm(x2, torch.t(a2))).cuda(0)) / 2.

        D_2 = torch.mm(D_2,
                       torch.diag(1./(torch.sqrt(torch.sum(D_2**2, 0))+1e-6)))
        D2_2 = D_2.cuda(1)

        a_2 = torch.mm(torch.t(D_2),
            whiten(torch.t(torch.t(D) - torch.mean(torch.t(D), 0)))).cuda(0)
        a2_2 = torch.mm(torch.t(D2_2),
            whiten(torch.t(torch.t(D2) - torch.mean(torch.torch.t(D2), 0)))).cuda(1)

        a_2 = torch.mm(a_2,
                       torch.diag(1./(torch.sqrt(torch.sum(a_2**2, 0))+1e-6)))
        a2_2 = torch.mm(a2_2,
                        torch.diag(1./(torch.sqrt(torch.sum(a2_2**2, 0))+1e-6)))

        a_2 = .6 * a_2 ** 3
        a2_2 = .6 * a2_2 ** 3

        D_2 = D_2 + torch.mm(D - torch.mm(D_2, a_2), torch.t(a_2))
        D_2 = (D_2 + (D2_2 + torch.mm(D2 - torch.mm(D2_2,
                                            a2_2), torch.t(a2_2))).cuda(0)) / 2.

        if show:
            cv2.namedWindow('dictionary', cv2.WINDOW_NORMAL)
            cv2.imshow('dictionary', montage(mat2ten(D_2.cpu().numpy(), c=3)))
            cv2.waitKey(1)

    print(iters/(time.time() - start))
    return D.cpu().numpy(), a.cpu().numpy()



X = file_get(fnames[0])

Bar = ProgressBar()
for i in Bar(range(X.shape[0])):
    X[i, :240//3, :320//3, :] = imresize(X[i, ...], [240//3, 320//3])
X = X[:, :240//3, :320//3, :]

d, a = X3(X, iters=500, batch_sz=1, num_dict_features=64, show=True)


#x = torch.ones(4, 3, 6, 7).cuda()
#d = torch.zeros(10, 3, 2, 7).cuda()
#a = f.conv2d(Variable(x), Variable(d)).cuda()
