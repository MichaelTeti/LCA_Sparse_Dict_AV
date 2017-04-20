import numpy as np
import h5py
import tensorflow as tf
import os
from scipy.misc import *
from skimage.util import view_as_windows
from time import sleep
import sys

os.chdir('/home/mpcr/RaceTrackRover')
d=np.load('rover_dictionary.npy')
imshow(d)
sys.exit(0)
x=h5py.File('onehot_dataset.h5')
x=x['x_dataset']
x=x[:int(x.shape[0]-x.shape[0]*0.5):3, 135:, :, :]
x=np.mean(x, axis=3)
ps=8
x=view_as_windows(x, (1, ps, ps))
x=x[:, ::6, ::6, :, :, :]
x=np.transpose(np.reshape(x, [x.shape[0]*x.shape[1]*x.shape[2], -1]))
k=150
LCA_iters=1000
D=np.random.rand(ps**2, k)
error_vals=np.zeros([LCA_iters, ])
resize_ps=30
vis_d=np.zeros([10*resize_ps, 10*resize_ps])
batch_sz=65
assert(x.shape[0]==ps**2), 'check data shape before sending to model'

with tf.Session() as sess:
  for iters in range(LCA_iters):
    print(iters)
    batch=x[:, np.int32(np.floor(np.random.rand(batch_sz)*x.shape[1]))]
    batch=(batch-np.mean(batch, axis=0))/(np.std(batch, axis=0)+1e-6)
    D=tf.matmul(D, tf.diag(1/tf.sqrt(tf.reduce_sum(D**2, 0))))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/tf.sqrt(tf.reduce_sum(a**2, 0))))
    a=0.3*a**3
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))
    #reconstruction_error=tf.reduce_mean((batch-tf.matmul(D, a))**2)
    #e=sess.run(reconstruction_error)
    #error_vals[iters]=e

  d=sess.run(D)
  for row in range(10):
    for col in range(10):
      resized_patch=imresize(np.reshape(d[:, row*10+col], [ps, ps]), [resize_ps, resize_ps])
      vis_d[row*resize_ps:row*resize_ps+resize_ps, col*resize_ps:col*resize_ps+resize_ps]=resized_patch
  #imshow(vis_d)
  print('Saving learned dictionary...')
  np.save('rover_dictionary.npy', vis_d)
  #np.save('rover_error.npy', error_vals)

      
