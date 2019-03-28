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
import time
import math

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
# Events have time in whole sec, time in ns, x in ]0, 127[, y in ]0, 127[
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

poses['t'] = poses['sec']-first_event_sec + 1e-9*(poses['nsec']-first_event_nsec) #time_ctrl in MATLAB
poses = poses[['t', 'qw', 'qx', 'qy', 'qz']] # Quaternions
print("Head: \n", poses.head(10))
print("Tail: \n", poses.tail(10))

# Convert quaternions to rotation matrices and save in a dictionary TODO: UGLY AS HELL!!
rotmats_dict = coordinate_transforms.q2R_dict(poses)
# print(rotmats_dict)
# print("Head: \n", poses.head(10))
# print("Tail: \n", poses.tail(10))

# rotmats_ctrl = np.zeros((num_poses, 3, 3))
# for k in len(num_poses):
#     rotmats_ctrl[k,:,:] =

## Image reconstruction using pixel-wise EKF
# Input: events, contrast threshold and camera orientation (discrete poses + interpolation)
# Output: reconstructed intensity image
# profile('on','-detail','builtin','-timer','performance')
starttime = time.time()

num_events_batch = 300
num_events_display = 100000
num_batches_display = math.floor(num_events_display / num_events_batch);

# Variables related to the reconstructed image mosaic. EKF initialization
# Gradient map
grad_map = {}
grad_map['x'] = np.zeros((output_height, output_width))
grad_map['y'] = np.zeros((output_height, output_width))
# % Covariance matrix of each gradient pixel
grad_initial_variance = 10
grad_map_covar = {}
grad_map_covar['xx'] =  np.ones((output_height, output_width)) * grad_initial_variance
grad_map_covar['xy'] = np.zeros((output_height, output_width))
grad_map_covar['yx'] = np.zeros((output_height, output_width))
grad_map_covar['yy'] =  np.ones((output_height, output_width)) * grad_initial_variance

# % For efficiency, a structure, called event map, contains for every pixel
# % the time of the last event and its rotation at that time.
s = {}
s['sae'] = -1e-6
s['rotation'] = np.zeros((3,3))
np.fill_diagonal(s['rotation'], np.NaN)
event_map = np.matlib.repmat(s, dvs_parameters['sensor_height'], dvs_parameters['sensor_width'])
print(event_map.shape)


rotmats_1stkey = list(rotmats_dict.keys())[0]
rot0 = rotmats_dict[rotmats_1stkey]  # to center the map around the first pose
one_vec = np.ones((num_events_batch, 1)) # ??
Id = np.eye(2) # ??

## Processing events
print("Processing events")
plt.figure()
fig_show_evol = plt.figure(facecolor='w')
#units = normalized should be true
first_plot = True # for efficient plotting

iEv = 0 # event counter
iBatch = 1 # packet-of-events counter

while True:
    if (iEv + num_events_batch > num_events):
        break #% There are no more events

    print("Showing events")
    #% Get batch of events
    events_batch = events[iEv:num_events_batch]
    iEv = iEv + num_events_batch
    print(events_batch)

    t_events_batch = events_batch['t']
    x_events_batch = events_batch['x']
    y_events_batch = events_batch['y']
    pol_events_batch = 2 * (events_batch['pol'] - 0.5)

    ## Get the two map points correspondig to each event and update the event map (time and rotation of last event)

    # Get time of previous event at same DVS pixel
    print(x_events_batch[0])
    print(y_events_batch[0])
    idx_to_mat = x_events_batch * dvs_parameters['sensor_height'] + y_events_batch

    print(event_map[13, 125]['sae'])
    t_prev_batch = np.array([event_map[x, y]['sae'] for x, y in zip(x_events_batch, y_events_batch)]).T
    print("Hello")
    print(t_prev_batch)

    #Get (interpolated) rotation of current event
    t_ev_mean = (t_events_batch.iloc[0] + t_events_batch.iloc[-1]) * 0.5
    print("Hi", poses['t'].iloc[-1])
    if t_ev_mean > poses['t'].iloc[-1]:
        break # event later than last known pose



    print(rotmats_dict)
    Rot = coordinate_transforms.rotation_interpolation(
        poses['t'], rotmats_dict, t_ev_mean)



    exit()

