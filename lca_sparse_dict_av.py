from skimage.util import view_as_windows
from scipy.io import savemat
import cv2
import numpy as np
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf
from scipy.io import *
from scipy.misc import *
import os

def create_batch(data, batch_size):
  return data[:, np.floor(np.random.rand(batch_size)*np.size(data, 1)).astype(int)]

#data=np.transpose(np.load('five_patches.npy')) # load data
data=imread('sf.jpg')
data=imresize(data, [200, 200])
data=view_as_windows(data, (16, 16, 1))
data_shape=data.shape
data=np.transpose(np.reshape(data, [data_shape[0]*data_shape[1]*data_shape[2], -1]))
k=300 # num of dictionary patches
batch_sz=200 # batch size
Dictionary=np.absolute(np.random.rand(256, k))
vis_d=np.zeros([20*40, 15*40])

sess=tf.InteractiveSession() # begin tensorflow session

X=tf.placeholder(tf.float32, shape=(256, None)) # data placeholder
D=tf.placeholder(tf.float32, shape=(256, k)) # dictionary placeholder

# Feature normalization
Dict=tf.matmul(D, tf.diag(tf.div(1.0, tf.sqrt(tf.reduce_sum(tf.square(D), 
  reduction_indices=0)))))

# compute sparse coefficients
a=tf.matmul(tf.transpose(Dict), X)

# Sparce coefficient normalization
a=tf.matmul(a, tf.diag(tf.div(1.0, tf.sqrt(tf.reduce_sum(tf.square(a), 
  reduction_indices=0)))))

# X cubed activation function
a=tf.mul(0.3, tf.pow(a, tf.mul(tf.ones([k, batch_sz]), 3.0)))

# Compute new dictionary using sparse coefficients
Dict=tf.add(Dict, tf.matmul(tf.sub(X, tf.matmul(Dict, a)), tf.transpose(a)))

# mean squared error
error=tf.reduce_mean(tf.square(tf.sub(X, tf.matmul(Dict, a))))

for i in range(200): # LCA iterations
  print('iteration: %d'%(i))
  batch=create_batch(data, batch_sz) # create new batch 
  # feed dictionary from last iteration to get new dictionary
  Dictionary=Dict.eval(feed_dict={X:batch, D:Dictionary}) 
  print Dictionary
  for row in range(20):
    for col in range(15):
      vis_d[row*40:row*40+40, col*40:col*40+40]=imresize(np.reshape(Dictionary[:, row*15+col],
        [16, 16]), [40, 40])
  cv2.imshow('dictionary', bytescale(vis_d))
  cv2.waitKey(1)
  e=error.eval(feed_dict={X:batch, D:Dictionary})
  print('mean squared error:%.4f'%(e))
Dic={}
Dic['dict']=Dictionary
savemat('dictionary.mat', Dic)

np.save('AVD.npy', Dictionary) # save dictionary
