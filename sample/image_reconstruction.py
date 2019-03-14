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
