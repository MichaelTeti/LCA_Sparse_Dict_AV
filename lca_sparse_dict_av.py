from scipy.io import savemat
import cv2
import numpy as np
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf
from scipy.io import *
from scipy.misc import *

def create_batch(data, batch_size):
  r=np.random.permutation(np.ma.size(data, 1))
  batch=data[:, r[:batch_size]]
  return batch

data=np.transpose(np.load('five_patches.npy')) # load data
k=700 # num of dictionary patches
batch_sz=1000 # batch size
randp=np.random.permutation(np.ma.size(data, 1))
Dictionary=data[:, randp[:700]] # initial dictionary w/random patches

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

# Compute new dictionary
Dict=tf.add(Dict, tf.matmul(tf.sub(X, tf.matmul(Dict, a)), tf.transpose(a)))

# difference between the data and reconstruction
error=tf.reduce_sum(tf.abs(tf.sub(X, tf.matmul(Dict, a))))

for i in range(300): # LCA iterations
  print('iteration: %d'%(i))
  batch=create_batch(data, batch_sz) # create new batch 
  
  # feed dictionary from last iteration to get new dictionary
  Dictionary=Dict.eval(feed_dict={X:batch, D:Dictionary}) 

  # error function
  e=error.eval(feed_dict={X:batch, D:Dictionary})
  print('Error:%.4f'%(e))

np.save('AVD.npy', Dictionary) # save dictionary
