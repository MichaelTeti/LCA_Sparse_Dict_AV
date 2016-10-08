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
from scipy.io import *
from scipy.misc import *
from extract_audio_and_video import *

audio_data=np.load('audio_data.npy') # load audio data
video_frames=np.load('frames.npy') # load video frames

# extract and compile video patches
video_patches=view_as_windows(video_frames, (11, 11, 5)) # extract 11x11x5 patches 
video_patches=video_patches[0::5, 0::5, 0::3, :, :, :] # sample every 5th pixel 
video_patches=np.reshape(video_patches, [10**2*1206, -1]) # reshape into vectors

# extract and compile audio patches
audio_patches=view_as_windows(audio_data, (60, 2, 5)) 
audio_patches=np.reshape(audio_patches[0::30, :, 0::3, :, :, :], [48*1206, -1])
audio_patches=np.concatenate((audio_patches, np.zeros([48*1206, 5])), axis=1)
video_patches=np.concatenate((video_patches, audio_patches), axis=0)
np.save('five_patches.npy', video_patches)
