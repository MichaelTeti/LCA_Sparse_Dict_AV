#######################################################################################
#--------------------------------------------------------------------------------------
#
#                                Michael A. Teti
#
#               Machine Perception and Cognitive Robotics Laboratory
#
#                  Center for Complex Systems and Brain Sciences
#
#                          Florida Atlantic University
#
#--------------------------------------------------------------------------------------
#######################################################################################
#--------------------------------------------------------------------------------------
#
# This program is an attempt to combine sparse modeling, locally-competitive neural 
# networks, and dictionary learning to create a sparse multimodal dictionary. 
# 
#--------------------------------------------------------------------------------------
#######################################################################################
from skimage.util import view_as_windows
import cv2
import numpy as np
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf
from scipy.io import *
from scipy.misc import *
from extract_audio_and_video import *

audio_data=np.load('audio_data.npy') # load audio data
a=loadmat('video_frames.mat') # load video frames
video_frames=a['video_frames']
sess=tf.InteractiveSession()

# extract and compile video patches
five_patches=np.zeros([5*17*17, 100*(3620/3-3)])
extract_vid_patches=tf.extract_image_patches(video_frames[:, :, :, np.newaxis], 
  ksizes=[1, 10, 10, 1], strides=[1, 2, 2, 1], rates=[1, 3, 3, 1], padding='VALID')
video_patches=sess.run(extract_vid_patches)
i=range(3620)
i=i[0::3]
for i1 in range(3620/3-3):
  for j in range(100):
    five_patches[:, i1*100+j]=np.reshape(video_patches[i[i1]:i[i1]+5, :, :, j], [-1])

# extract and compile audio patches
audio_patches=view_as_windows(audio_data, (144, 2, 5))
audio_patches=np.reshape(audio_patches[0::30, :, 0::3, :, :, :], [45*1206, -1])
audio_patches=np.concatenate((audio_patches, np.zeros([45*1206, 5])), axis=1)
five_patches=np.concatenate((five_patches, np.transpose(audio_patches)), axis=1)
np.save('five_patches.npy', five_patches)
