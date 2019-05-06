import os
import time
import sys
import math

import numpy as np
import pandas as pd
import scipy.linalg as sp
import math
import sys

from mpl_toolkits.mplot3d import Axes3D
from sys import platform as sys_pf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sample.coordinate_transforms as coordinate_transforms
import sample.helpers as helpers

def compare_trajectories(df_ours, df_theirs):
    """
    :return: function checks whether the rotation matrices are really randomly distributed. muoltiplies rot matrix with Z-unit-vector. returns plotly and matplotlib plot which shows the distribution

    Function checks whether the rotation matrices are really randomly distributed.
    multiplies rot matrix with Z-unit-vector.
    :return: plotly and matplotlib plot which shows the distribution
    """


    vec = np.array([1,0,0]).T
    vecM = df_ours['Rotation'].apply(lambda x: np.dot(x, vec))
    rotX = vecM.str.get(0)
    rotY = vecM.str.get(1)
    rotZ = vecM.str.get(2)

    vecM_theirs = df_theirs['Rotation'].apply(lambda x: np.dot(x, vec))
    rotX_theirs= vecM_theirs.str.get(0)
    rotY_theirs = vecM_theirs.str.get(1)
    rotZ_theirs = vecM_theirs.str.get(2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    q =ax.scatter(rotX_theirs, rotY_theirs, rotZ_theirs, s = 0.1, color = 'r')
    ax.scatter(rotX_theirs[0], rotY_theirs[0], rotZ_theirs[0], s=10, color = 'm', marker = 'D')
    p =ax.scatter(rotX, rotY, rotZ, c=range(len(rotZ)), s=0.5 )
    cbar = fig.colorbar(p, ax=ax)
    # cbar2 = fig.colorbar(q, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    cbar.set_label("Nr. of pose")

    plt.show()

def cut_df_wrt_time(rotations_ours, rotations_theirs):
    t_max = rotations_ours['t'].max()
    rotations_theirs_cut = rotations_theirs[rotations_theirs['t'] < t_max]

    return rotations_theirs_cut


def visualize_particles(rotation_matrices, mean_value = None):
    """
    :return: function checks whether the rotation matrices are really randomly distributed. muoltiplies rot matrix with Z-unit-vector. returns plotly and matplotlib plot which shows the distribution

    Function checks whether the rotation matrices are really randomly distributed.
    multiplies rot matrix with Z-unit-vector.
    :return: plotly and matplotlib plot which shows the distribution
    """

    vec = np.array([1,0,0]).T
    vecM = rotation_matrices.apply(lambda x: np.dot(x, vec))
    rotX = vecM.str.get(0)
    rotY = vecM.str.get(1)
    rotZ = vecM.str.get(2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    p = ax.scatter(rotX, rotY, rotZ, c=range(len(rotZ)))
    if mean_value is not None:
        mean_vec = np.dot(mean_value, vec)
        q = ax.scatter3D(mean_vec[0],mean_vec[1],mean_vec[2], 'b')
    cbar = fig.colorbar(p, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    cbar.set_label("Nr. of pose")

    plt.show()


def plot_unitsphere_matplot():
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
    plt.show()


if __name__ == '__main__':
    directory_poses = '../output/poses/'
    filename_ours = 'quaternions_05052019T203206.txt'
    filename_theirs = 'poses.txt'
    poses_ours = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_ours))
    poses_theirs = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_theirs),
                                      includes_translations=True)
    rotations_ours = coordinate_transforms.q2R_df(poses_ours)
    rotations_theirs = coordinate_transforms.q2R_df(poses_theirs)
    rotations_theirs_cut = cut_df_wrt_time(rotations_ours, rotations_theirs)

    compare_trajectories(rotations_ours,rotations_theirs_cut)
