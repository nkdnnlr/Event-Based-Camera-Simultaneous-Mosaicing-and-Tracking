# DVS image reconstruction demo
#
# Given DVS events and camera poses (rotational motion), reconstruct the
# gradient map that caused the events.
#
# Céline Nauer, Institute of Neuroinformatics, University of Zurich
# Joël Bachmann, ETH
# Nik Dennler, Institute of Neuroinformatics, University of Zurich
#
# Original MATLAB code by
#  Guillermo Gallego
#  Robotics and Perception Group
#  University of Zurich

import os

from scipy import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyquaternion

import sample.integration_methods as integration_methods
import sample.coordinate_transforms as coordinate_transforms


## ___Dataset___

# Data directories
data_dir = '../data/synth1'
calibration_dir = '../data/calibration'

# Calibration data
dvs_calibration = io.loadmat(os.path.join(calibration_dir, 'DVS_synth_undistorted_pixels.mat'))
# Dictionary with entries ['__header__', '__version__', '__globals__',
#                          'Kint', 'dist_coeffs', 'image_height', 'image_width',
#                          'pixels_grid', 'undist_pix', 'undist_pix_calibrated',
#                          'undist_pix_mat_x', 'undist_pix_mat_y'])

dvs_parameters = {'sensor_height': 128, 'sensor_width': 128, 'contrast_threshold': 0.45}


## ___Main algorithmic parameters___

# Size of the output reconstructed image mosaic (panorama)
output_height = 1024
output_width = 2 * output_height

# Methods used:
# 1) Select measurement function used for brightness gradient estimation
#    via Extended Kalman Filter (EKF). Options are:
#    'contrast'    : Constrast criterion (Gallego et al. arXiv 2015)
#    'event rate'  : Event rate criterion (Kim et al. BMVC 2014)
measurement_criterion = 'contrast'

# Set the expected noise level (variance of the measurements).
if measurement_criterion == 'contrast':
    var_R = 0.17**2 # units[C_th] ^ 2
if measurement_criterion == 'event_rate':
    var_R = 1e2**2  # units[1 / second] ^ 2

# 2) Select the gradient integration method. Options are:
#    'poisson_dirichlet'   : Requires the MATLAB PDE toolbox
#    'poisson_neumann'     : Requires the MATLAB Image Processing toolbox
#    'frankotchellappa'    : Does not require a MATLAB toolbox
integration_method = 'frankotchellappa'


# Provide function handles to convert from rotation matrix to axis-angle and vice-versa
f_r2a = coordinate_transforms.r2aa
f_a2r = coordinate_transforms.aa2r
# TODO: What about q2R?

## Loading Events
print("Loading Events")
filename_events = os.path.join(data_dir, 'events.txt')
events = pd.read_csv(filename_events, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'pol'])
# print("Head: \n", events.head(10))
num_events = events.size
print("Number of events in file: ", num_events)

# Remove time of offset
first_event_sec = events.loc[0, 'sec']
first_event_nsec = events.loc[0, 'nsec']
events['t'] = events['sec']-first_event_sec + 1e-9*(events['nsec']-first_event_nsec)
events = events[['t', 'x', 'y', 'pol']]
print("Head: \n", events.head(10))
print("Tail: \n", events.tail(10))


##Loading Camera poses
print("Loading Camera Orientations")
filename_events = os.path.join(data_dir, 'poses.txt')
poses = pd.read_csv(filename_events, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
num_poses = poses.size
print("Number of poses in file: ", num_poses)

poses['t'] = poses['sec']-first_event_sec + 1e-9*(poses['nsec']-first_event_nsec)
poses = poses[['t', 'qw', 'qx', 'qy', 'qz']] # Quaternions
print("Head: \n", poses.head(10))
print("Tail: \n", poses.tail(10))

# Convert quaternions to rotation matrices
