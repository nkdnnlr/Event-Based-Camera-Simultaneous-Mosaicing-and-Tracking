import numpy as np
import pandas as pd


def r2aa(R):
    """
    Convert Rotation Matrix (R) to Axis Angle (AA)
     Written by Garrick Orchard July 2017
     Based on:
     https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis.E2.80.93angle
    :param R: a 3x3 rotation matrix (np.array)
    :return: AA: a 1x4 axis angle. The axis is not normalized
    """

    # Check that input is a rotation matrix
    assert (R.shape == (3,3)) & (R.size == 9), "Input must be a 3x3 matrix"
    assert np.linalg.norm(R.transpose @ R - np.identity(3), 'fro') < 1e-7, 'Input must be a 3D rotation matrix'
    assert np.linalg.det(R) > 0, "Input must be a 3D rotation matrix"

    # Get rotation angle
    theta = np.arccos((np.trace(R) - 1) / 2)
    theta = np.real(theta)  # in case cosine is slightly out of the range[-1, 1]

    # Get rotation axis
    if abs(theta - np.pi) < 1e-3:
        # Rotations with an angle close to pi

        # Obtain the axis from the quadratic term in[u]_x in Rodrigues formula
        # Get the vector that generates the rank-1 matrix
        U = np.linalg.svd(0.5 * (R + np.identity(3)))[0]
        ax = U[:, 0].transpose()

        # Adjust the sign of the axis
        if np.linalg.norm(aa2r([ax, theta]) - R, 'fro') > np.linalg.norm(aa2r([-ax, theta]) - R, 'fro'):
            ax = - ax

    else:
        # Most rotations obtain the axis from the linear term in [u]_x in Rodrigues formula
        ax = [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]

        norm_ax = np.linalg.norm(ax)
        if norm_ax > 1e-8:
            ax = ax / norm_ax  # Ensure a unit length axis direction
        else:
            # Rotation close to zero degrees. Axis is undetermined
            ax = [0, 0, 1]  # This is what rotm2axang outputs

    # Output 4-vector: [axis, angle]
    AA = ax.copy()
    AA.append(theta)
    return AA


def aa2r(AA):
    """
    Convert Axis Angle (AA) to Rotation Matrix (R)
     Written by Garrick Orchard July 2017
     Based on:
     http://mathworld.wolfram.com/RodriguesRotationFormula.html
    :param AA: 1x4 axis angle rotation.
    :return: R: a 3x3 rotation matrix
    """

    assert AA.shape == 4, 'Input must be 1x4 or 4x1'  ##TODO: this might cause errors, check MATLAB

    # Axis
    norm_ax = np.linalg.norm(AA[0:3])
    if norm_ax < 1e-6:
        R = np.identity(3)
        return R
    ax = AA[0:3] / norm_ax  # Unit norm, avoid division by zero

    # Cross - product matrix
    omega = [[0, -ax(2), ax(1),
              ax(2), 0, -ax(0),
              -ax(1), ax(0), 0]]

    # Rotation angle
    theta = AA(3)

    # Rotation matrix, using Rodrigues formula
    R = np.identity(3) + omega * np.sin(theta) + omega * omega * (1 - np.cos(theta))
    return R


def q2R(q):
    """
    Converts quaternion(q) to 3x3 rotation matrix (R)
    MATLAB Code written by Garrick Orchard July 2017, based on
    http: // www.euclideanspace.com / maths / geometry / rotations / conversions / quaternionToMatrix /
    :param q: [qw, qx, qy, qz] * [1 i j k]
    :return: R
    """
    R = np.zeros((3, 3))
    R[0, 0] = 1. - 2. * (q[2]**2. + q[3]**2.)
    R[0, 1] = 2. * (q[1] * q[2] - q[3] * q[0])
    R[0, 2] = 2. * (q[1] * q[3] + q[2] * q[0])

    R[1, 0] = 2. * (q[1] * q[2] + q[3] * q[0])
    R[1, 1] = 1. - 2. * (q[1]**2. + q[3]**2.)
    R[1, 2] = 2. * (q[2] * q[3] - q[1] * q[0])

    R[2, 0] = 2. * (q[1] * q[3] - q[2] * q[0])
    R[2, 1] = 2. * (q[2] * q[3] + q[1] * q[0])
    R[2, 2] = 1. - 2. * (q[1]**2. + q[2]**2.)

    # print("q: ", q)
    # print("R: \n", R)
    # print("R: \n", R[1,0])


    # Numerically improve result by projecting on the space of rotation matrices
    # print(np.linalg.svd(R))

    u, s, v_T = np.linalg.svd(R, full_matrices=True)
    v = v_T.T
    R = u.dot(np.diag(np.array([1., 1., np.linalg.det(u.dot(v_T))]))).dot(v_T)
    return R

def q2R_dict(df):
    """

    :param df:
    :return:
    """
    # df['rotmats_ctrl'] = np.zeros((3,3))
    rotmats = {}
    for idx, row in df.copy().iterrows():
        rotmats[df.loc[idx, 't']] = q2R((df.loc[idx, 'qw'], df.loc[idx, 'qx'],
                                          df.loc[idx, 'qy'], df.loc[idx, 'qz']))
    return rotmats




# # Testing... TODO: Write proper test functions
# qw = 0.70746
# qx = -0.706753
# qy = 0.000354
# qz = 0.000353
#
# print(q2R([qw, qx, qy, qz]))

# should give something as (from matlab code with same quaternions)
# 0.999999500150000 -0.000999848031280 0.000001914209993
# -0.000000914216424 0.001000145059139 0.999999499854388
# -0.000999849445698 -0.999999000006388 0.001000143645140
# num_poses = 4
# rotmats_ctrl = np.zeros((num_poses, 3, 3))
# print(rotmats_ctrl)
# for k in range(num_poses):
#     rotmats_ctrl[k,:,:] = q2R([qw, qx, qy, qz])