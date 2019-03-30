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
    # assert np.linalg.norm(R.transpose @ R - np.identity(3), 'fro') < 1e-7, 'Input must be a 3D rotation matrix' TODO: removed warning!
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
    AA = list(ax.copy())
    AA.append(theta)
    return AA


def aa2r(AA):
    """
    Tested
    Convert Axis Angle (AA) to Rotation Matrix (R)
     Written by Garrick Orchard July 2017
     Based on:
     http://mathworld.wolfram.com/RodriguesRotationFormula.html
    :param AA: 1x4 axis angle rotation.
    :return: R: a 3x3 rotation matrix
    """
    assert AA.shape == (4,), 'Input must be 1x4 or 4x1'  ##TODO: this might cause errors, check MATLAB

    # Axis
    norm_ax = np.linalg.norm(AA[0:3])
    if norm_ax < 1e-6:
        R = np.identity(3)
        return R
    ax = AA[0:3] / norm_ax  # Unit norm, avoid division by zero

    # Cross - product matrix
    omega = np.array([[0, -ax[2], ax[1]],
              [ax[2], 0, -ax[0]],
              [-ax[1], ax[0], 0]])

    # Rotation angle
    theta = AA[3]

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
    Transforms quaternions vector q to rotation matrix R
    :param df:
    :return:
    """
    # df['rotmats_ctrl'] = np.zeros((3,3))
    rotmats = {}
    for idx, row in df.copy().iterrows():
        rotmats[df.loc[idx, 't']] = q2R((df.loc[idx, 'qw'], df.loc[idx, 'qx'],
                                          df.loc[idx, 'qy'], df.loc[idx, 'qz']))
    return rotmats


def rotation_interpolation(t, rotmats_dict, t_query):
    """
    Function for linear interpolation of rotations.
    Interpolate the orientation (rotation) at a given time.
    TODO: Tested. Seems a bit inaccurate compared to MATLAB output.
    :param t: timestamps of discrete set of orientations ("control poses")
    :param rotmats: discrete set of rotation matrices
    :param t_query: time of the requested rotation matrix
    :return: interpolated rotation matrix
    """
    rotmats_1stkey = list(rotmats_dict.keys())[0]
    rotmats_lastkey = list(rotmats_dict.keys())[-1]

    # Rotation interpolation
    if (t_query < t.iloc[0]  or t_query > t.iloc[-1]):
        rot_interp = np.NaN * rotmats_dict[rotmats_1stkey]

    elif t_query == t.iloc[-1]:
        rot_interp = rotmats_dict[rotmats_lastkey]

    else:
        idx_0 = t.loc[t <= t_query].index.max() #Take maximal index where t <= t_query
        print(idx_0)

        # Two rotations and their times
        t_0 = t[idx_0]
        t_1 = t[idx_0 + 1]

        rot_0 = rotmats_dict[t_0]
        rot_1 = rotmats_dict[t_1]

        print(rot_0)
        print(rot_1)

        # Interpolation parameter in [0, 1]
        d_t = (t_query - t_0) / (t_1 - t_0)

        # Linear interpolation, Lie group formulation
        axang_increm =np.array(r2aa((rot_0.T).dot(rot_1)))  #TODO: is a little bit different than MATLABS, but only on the 4th digit
        axang_increm[3] *= d_t
        rot_interp = rot_0.dot(aa2r(np.array(axang_increm)))
        return rot_interp


def project_equirectangular_projection(point_3d, output_width, output_height):
    """
    Tested
    Project a 3D point according to equirectangular model
    Used for 360 degrees panoramic cameras that output a full panorama frame
    :param point_3d: a 3D point
    :param output_width: width of the panorama
    :param output_height: height of the panorama
    :return: point_2d: projected point (coordinates in the panorama image)
    """
    rho = np.sqrt(sum(np.square(point_3d), 1)) # norm of each 3D point
    print(rho.shape)

    fx = output_width / (2.0 * np.pi);
    fy = output_height / np.pi
    principal_point = 0.5 * np.array([output_width, output_height])

    # np.arctan2
    phi = np.arctan2(point_3d[0,:], point_3d[2,:])

    theta = np.arcsin(-point_3d[2,:] / rho)
    point_2d = np.array([phi * fx, -theta * fy])
    point_2d[0, :] = point_2d[0, :] + principal_point[0]
    point_2d[1, :] = point_2d[1, :] + principal_point[1]
    return point_2d



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