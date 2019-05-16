import numpy as np
import scipy.linalg as sp
import pandas as pd
import matplotlib.pyplot as plt

import Tracking_Particle_Filter.tracking as tracking
import sample.visualisation as visualilation





def rotate(particles, sigma1, sigma2=0, sigma3=0):
    """
    Randomly (normal) perturbs particles.
    :param particles: DataFrame with particles
    :param tau: timestep
    :return: DataFrame with updated particles
    """
    # particles = particles.copy()

    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # rotation around x
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])  # rotation around y
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # rotation around z

    R_c = sp.expm(np.dot(sigma1, G1) + np.dot(sigma2, G2) + np.dot(sigma3, G3))

    # print(R_c)

    particles['Rotation'] = particles['Rotation'].apply(lambda x: np.dot(x, R_c))
    return particles['Rotation'].loc[0]


# exit()


# startmatrix = tracking.generate_random_rotmat(seed=1)

# startmatrix = np.array([[ 0.99525582, -0.00103689,  0.09728704],
#                         [-0.09720775, -0.05234339,  0.99388673],
#                         [ 0.00406178, -0.99862861, -0.05219586]])

startmatrix = np.array([[ 9.99999500e-01, -9.99848031e-04,  1.91420999e-06],
                        [-9.14216424e-07,  1.00014506e-03,  9.99999500e-01],
                        [-9.99849446e-04, -9.99999000e-01,  1.00014365e-03]])

particles = tracking.init_particles(N=1, init_rotmat=startmatrix, bound1=0, bound2=0, bound3=0)
print(particles)

sigma = 3.366410184326084e-05
m = 10004 ** 4
step = m*sigma
step = 2*np.pi/360

sigma1 = 0.0001334552012881879
sigma2 = 0.00013345525464213625
sigma3 = 0.0005283667589204627

rotated = []
rotated.append(startmatrix)
for i in range(10000):
    rotated.append(rotate(particles,
                          sigma1=sigma1, sigma2=sigma2, sigma3=sigma3))

# particle = pd.DataFrame(columns=['Rotation'])
# particle[0] =)

# print(particle)


import os
import sample.helpers as helpers
import sample.coordinate_transforms as coordinate_transforms

print(startmatrix)


all_rotations = tracking.run()
# print(all_rotations['Rotation'].values)
# all_rotations_array = all_rotations['Rotation'].values
# exit()

visualilation.visualize_rotmats(rotated)


# visualilation.visualize_rotmats(all_rotations_array)
#
# quaternions = helpers.rot2quaternions(all_rotations)
# rotations_ours = coordinate_transforms.q2R_df(quaternions)
# rotations_ours_array = rotations_ours['Rotation'].values
#
#
# visualilation.visualize_rotmats(rotations_ours_array)
#
directory_poses = '../output/poses/'
# filename_ours = 'quaternions_07052019T150842.txt'
filename_theirs = 'poses.txt'
# poses_ours = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_ours))
poses_theirs = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_theirs),
                                  includes_translations=True)
# rotations_ours = coordinate_transforms.q2R_df(poses_ours)
# print(rotations_ours)

rotations_theirs = coordinate_transforms.q2R_df(poses_theirs)


# visualilation.compare_trajectories(rotations_ours, rotations_theirs)
visualilation.compare_trajectories(all_rotations, rotations_theirs)

# visualilation.compare_trajectories(rotations_ours, rotations_theirs)

