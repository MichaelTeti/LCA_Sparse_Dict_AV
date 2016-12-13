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

data=imread('sf.jpg')
data=imresize(data, [200, 200])  # resize to a smaller size
data=view_as_windows(data, (16, 16, 1))  # take 16x16x1 patches from image
data_shape=data.shape
ps=256
# vectorize patches 
data=np.transpose(np.reshape(data, [data_shape[0]*data_shape[1]*data_shape[2], -1]))
# scale columns
data=(data-np.tile(np.mean(data, axis=0), (ps, 1)))/np.tile(np.std(data, axis=0), (ps, 1))
k=300 # num of dictionary patches
batch_sz=100 # batch size
Dictionary=np.random.rand(ps, k)
vis_d=np.zeros([20*40, 15*40])

sess=tf.InteractiveSession() # begin tensorflow session

X=tf.placeholder(tf.float32, shape=(ps, None)) # data placeholder
D=tf.placeholder(tf.float32, shape=(ps, k)) # dictionary placeholder

# Feature normalization
Dict=tf.matmul(D, tf.diag(1/tf.sqrt(tf.reduce_sum(D**2, 0))))

# compute sparse coefficients
a=tf.matmul(tf.transpose(Dict), X)

# Sparce coefficient normalization
a=tf.matmul(a, tf.diag(1/tf.sqrt(tf.reduce_sum(a**2, 0))))

# X cubed activation function
a=0.3*a**3

# Compute new dictionary using sparse coefficients
Dict=Dict+tf.matmul((X-tf.matmul(Dict, a)), tf.transpose(a))

# mean squared error
error=tf.reduce_mean(tf.square(tf.sub(X, tf.matmul(Dict, a))))

for i in range(10000): # LCA iterations
  print('iteration: %d'%(i))
  batch=create_batch(data, batch_sz) # create new batch
  # feed dictionary from last iteration to get new dictionary
  Dictionary=Dict.eval(feed_dict={X:batch, D:Dictionary}) 
  for row in range(20):
    for col in range(15):
      vis_d[row*40:row*40+40, col*40:col*40+40]=imresize(np.reshape(Dictionary[:, row*15+col], [16, 16]), [40, 40])
  cv2.imshow('dictionary', bytescale(vis_d))
  cv2.waitKey(1)
  e=error.eval(feed_dict={X:batch, D:Dictionary})
  print('mean squared error:%.4f'%(e))
Dic={}
Dic['dict']=Dictionary
savemat('dictionary.mat', Dic)

np.save('AVD.npy', Dictionary) # save dictionary
