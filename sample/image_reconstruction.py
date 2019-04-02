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
s['rotation'].fill(np.NaN)
# np.fill_diagonal(s['rotation'], np.NaN)
event_map = np.zeros((dvs_parameters['sensor_height'], dvs_parameters['sensor_width']))
event_map = event_map.tolist()
for i in range(dvs_parameters['sensor_height']):
    for j in range(dvs_parameters['sensor_width']):
        event_map[i][j] = s.copy()
# event_map = np.matlib.repmat(s.copy(), dvs_parameters['sensor_height'], dvs_parameters['sensor_width']) # ATTENTION: all s.copy are the same!!
event_map = np.array(event_map)
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

i=0
while True:
    if (iEv + num_events_batch > num_events):
        break #% There are no more events

    print("Showing events")
    #% Get batch of events
    events_batch = events[iEv:num_events_batch]
    iEv = iEv + num_events_batch
    # print(events_batch)

    t_events_batch = events_batch['t']
    x_events_batch = events_batch['x']
    y_events_batch = events_batch['y']
    pol_events_batch = 2 * (events_batch['pol'] - 0.5)

    ## Get the two map points correspondig to each event and update the event map (time and rotation of last event)

    # Get time of previous event at same DVS pixel
    # print(x_events_batch[0])
    # print(y_events_batch[0])
    idx_to_mat = x_events_batch * dvs_parameters['sensor_height'] + y_events_batch

    # print(event_map[13, 125]['sae'])
    t_prev_batch = np.array([event_map[x, y]['sae'] for x, y in zip(x_events_batch, y_events_batch)]).T
    # print("Hello")
    # print(t_prev_batch)

    #Get (interpolated) rotation of current event
    t_ev_mean = (t_events_batch.iloc[0] + t_events_batch.iloc[-1]) * 0.5
    # print("Hi", poses['t'].iloc[-1])
    if t_ev_mean > poses['t'].iloc[-1]:
        break # event later than last known pose


    # print(rotmats_dict)
    Rot = coordinate_transforms.rotation_interpolation(
        poses['t'], rotmats_dict, t_ev_mean)


    # print(np.concatenate((dvs_calibration['undist_pix_calibrated'][idx_to_mat, :], one_vec)))
    bearing_vec = np.vstack((dvs_calibration['undist_pix_calibrated'][idx_to_mat, :].T[0],
                             dvs_calibration['undist_pix_calibrated'][idx_to_mat, :].T[1],
                             one_vec[:,0])) # 3xN
    # print(bearing_vec)

    # Get map point corresponding to current event
    # print(rot0.shape)
    # print(Rot.shape)
    rotated_vec = rot0.T.dot(Rot).dot(bearing_vec)
    # print(rotated_vec)
    pm = coordinate_transforms.project_equirectangular_projection(rotated_vec, output_width, output_height)

    #  Get map point corresponding to previous event at same pixel
    rotated_vec_prev = np.zeros(rotated_vec.shape)
    # print(rotated_vec_prev.shape)
    # print(event_map.shape)
    # print(event_map[109,109].rotation)
    # exit()
    # for ii in range(num_events_batch):
    # print("BEARING: ", bearing_vec)
    # print("Rot0: ", rot0)
    # print("ROTaTED VEC PREV: ")
    # print(rotated_vec_prev)


    # event_map = event_map.tolist() #make list to change single entries
    for ii in range(num_events_batch):

        Rot_prev = event_map[x_events_batch[ii]][y_events_batch[ii]]['rotation'].copy()

        rotated_vec_prev[:, ii] = rot0.T.dot(Rot_prev).dot(bearing_vec[:,ii])
        #print(rotated_vec_prev[:, ii])
        # Update last rotation and time of event(SAE)
        # print(event_map[x_events_batch[ii], y_events_batch[ii]])
        # print(event_map[x_events_batch[2]][y_events_batch[2]])
        event_map[x_events_batch[ii]][y_events_batch[ii]]['sae'] = t_events_batch[ii]
        event_map[x_events_batch[ii]][y_events_batch[ii]]['rotation'] = Rot
        # print(event_map[x_events_batch[2]][y_events_batch[2]])

    pm_prev = coordinate_transforms.project_equirectangular_projection(rotated_vec_prev, output_width, output_height)
    # print("PM_PREV: ", pm_prev) ## Tested: Looks just as matlab output (still with discrepancies on ~5th digit)
    # print("ISNAN?")

    # print(pm - pm_prev) ##ATTENTION, actually differs from MATLAB output. Only a number of 1e13, so shouldn't matter
    # print(sum(np.isnan(pm_prev[0,:]) | np.isnan(pm_prev[1,:])))
    # print(sum(np.isnan(pm_prev[1,:])))
    # exit()

    ##TODO: Include again in code in the end!!
    if (t_prev_batch[-1] < 0) or (t_prev_batch[-1] < poses['t'][0]):
        continue # initialization phase. Fill in event_map

    #  Discard nan values
    mask_uninitialized = np.isnan(pm_prev[0,:]) | np.isnan(pm_prev[1,:])
    num_uninitialized = sum(mask_uninitialized)
    # print(num_uninitialized)

    # exit()
    if (num_uninitialized > 0):
        # % Delete uninitialized events
        print('Deleting {} points'.format(str(num_uninitialized)))
        t_events_batch = np.array(t_events_batch.iloc[~mask_uninitialized].tolist())
        t_prev_batch = np.array(t_prev_batch[~mask_uninitialized].tolist())

        pol_events_batch = np.array(pol_events_batch[~mask_uninitialized].tolist())
        # pm = pm[:][~mask_uninitialized].tolist()
        # pm_prev = pm[:][~mask_uninitialized].tolist()
        pm_ = pm.copy()
        pm_prev_ = pm_prev.copy()
        pm = []
        pm_prev = []
        for row in pm_:
            pm.append(row[~mask_uninitialized].tolist())
        for row in pm_prev_:
            pm_prev.append(row[~mask_uninitialized].tolist())
        pm = np.array(pm)
        pm_prev = np.array(pm_prev)
    # print(len(pm[0]))

    # Get time since previous event at same pixel
    tc = t_events_batch - t_prev_batch
    event_rate = 1./(tc + 1e-12) #% measurement or observation(z)

    # Get velocity vector
    # print(pm - pm_prev)
    vel = (pm - pm_prev) * event_rate  #TODO: Check for a point after the first to evaluate
    # exit()

    ## Extended Kalman Filter (EKF) for the intensity gradient map.
    # Get gradient and covariance at current map points pm
    ir = np.floor(pm[1,:]).astype(int) # row is y coordinate
    ic = np.floor(pm[0,:]).astype(int) # col is x coordinate

    gm = np.array([[grad_map['x'][i][j], grad_map['y'][i][j]] for i, j in zip(ir, ic)])
    Pg = np.array([[grad_map_covar['xx'][i][j],
           grad_map_covar['xy'][i][j],
           grad_map_covar['yx'][i][j],
           grad_map_covar['yy'][i][j]] for i, j in zip(ir, ic)])


    # EKF update
    if measurement_criterion == 'contrast':
        # Use contrast as measurement function
        # print("NOW")
        # print(gm.shape)
        dhdg = vel.T * np.array([tc * pol_events_batch, tc * pol_events_batch]).T # derivative of measurement function
        nu_innovation = dvs_parameters['contrast_threshold'] - np.sum(dhdg * gm, axis=1)
    else:
        # Use the event rate as measurement function
        dhdg = vel.T / np.array([dvs_parameters['contrast_threshold'] * pol_events_batch,
                                 dvs_parameters['contrast_threshold'] * pol_events_batch]).T #deriv. of measurement function
        nu_innovation = event_rate - np.sum(dhdg * gm, axis=1)

    Pg_dhdg = np.array([Pg[:, 0]*dhdg[:, 0] + Pg[:, 1] * dhdg[:, 1],
                        Pg[:, 2]*dhdg[:, 0] + Pg[:, 3] * dhdg[:, 1]]).T

    S_covar_innovation = dhdg[:, 0] * Pg_dhdg[:, 0] + dhdg[:, 1] * Pg_dhdg[:, 1] + var_R
    Kalman_gain = Pg_dhdg / np.array([S_covar_innovation, S_covar_innovation]).T
    # Update gradient and covariance
    gm = gm + Kalman_gain * np.array([nu_innovation, nu_innovation]).T
    Pg = Pg - np.array([Pg_dhdg[:, 0] * Kalman_gain[:, 0], Pg_dhdg[:, 0] * Kalman_gain[:, 1],
               Pg_dhdg[:, 1] * Kalman_gain[:, 0], Pg_dhdg[:, 1] * Kalman_gain[:, 1]]).T
    # print(gm)
    # print(Pg) #TODO: Test with points after the first one...

    # Store updated values
    gm = np.array([[grad_map['x'][i][j], grad_map['y'][i][j]] for i, j in zip(ir, ic)])

    k = 0
    for i, j in zip(ir, ic):
        grad_map['x'][i][j] = gm[k, 0]
        grad_map['y'][i][j] = gm[k, 1]
        grad_map_covar['xx'][i][j] = Pg[k, 0]
        grad_map_covar['xy'][i][j] = Pg[k, 1]
        grad_map_covar['yx'][i][j] = Pg[k, 2]
        grad_map_covar['yy'][i][j] = Pg[k, 3]
        k += 1
    exit()