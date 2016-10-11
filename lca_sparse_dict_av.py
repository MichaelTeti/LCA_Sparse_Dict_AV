from scipy.io import savemat
import cv2
import numpy as np
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf
from scipy.io import *
from scipy.misc import *
import os

if os.path.isfile('./AVD.npy')==True:
  if os.path.isfile('./avd.mat')==True:
    sys.exit()
  dict1=np.load('AVD.npy')
  comp_dict=np.zeros([20*11, 35*11, 5])
  for i in range(20):
    for j in range(35):
      comp_dict[i*11:i*11+11, j*11:j*11+11, :]=np.reshape(dict1[:, i*35+j], [11, 11, 5])
  cd={}
  cd['dict']=comp_dict
  savemat('avd.mat', cd)
  sys.exit()

def create_batch(data, batch_size):
  return data[:, np.floor(np.random.rand(batch_size)*np.size(data, 1)).astype(int)]

data=np.transpose(np.load('five_patches.npy')) # load data
k=700 # num of dictionary patches
batch_sz=1000 # batch size

# initialize dictionary with random patches from data
Dictionary=data[:, np.floor(np.random.rand(k)*np.size(data, 1)).astype(int)]

sess=tf.InteractiveSession() # begin tensorflow session
X=tf.placeholder(tf.float32, shape=(605, batch_sz)) # data placeholder
D=tf.placeholder(tf.float32, shape=(605, k)) # dictionary placeholder

# Feature normalization
Dict=tf.mul(D, tf.div(1.0, tf.sqrt(tf.reduce_sum(tf.square(D), 0))))

# compute sparse coefficients
a=tf.matmul(tf.transpose(Dict), X)

# sparse coefficient normalization
a=tf.mul(a, tf.div(1.0, tf.sqrt(tf.reduce_sum(tf.square(a), 0))))

# X cubed activation function
a=tf.mul(tf.constant(.3), tf.pow(a, tf.mul(tf.ones([k, batch_sz]),
  tf.constant(3.0))))

# Compute new dictionary using sparse coefficients
Dict=tf.add(Dict, tf.matmul(tf.sub(X, tf.matmul(Dict, a)), tf.transpose(a)))

# difference between the data and reconstruction
error=tf.reduce_mean(tf.square(tf.sub(X, tf.matmul(Dict, a))))

for i in range(5000): # LCA iterations
  print('iteration: %d'%(i))
  batch=create_batch(data, batch_sz) # create new batch 
  
  # feed dictionary from last iteration to get new dictionary
  Dictionary=Dict.eval(feed_dict={X:batch, D:Dictionary}) 

  # mean squared error between batch and reconstruction
  e=error.eval(feed_dict={X:batch, D:Dictionary})
  print('Error:%.4f'%(e))

np.save('AVD.npy', Dictionary) # save dictionary
