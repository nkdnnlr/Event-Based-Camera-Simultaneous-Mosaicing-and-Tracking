import os
import numpy as np
import pandas as pd
import scipy.linalg as sp
import sample.helpers.coordinate_transforms as coordinate_transforms
import datetime


def get_first_matrix(filename_poses):
    """
    gets first matrix from poses file
    :param filename_poses: filename of poses
    :return: return first matrix as np array
    """
    poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    num_poses = poses.size

    poses['t'] = poses['sec'] + 1e-9*(poses['nsec']) #time_ctrl in MATLAB
    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']] # Quaternions


    first_matrix = coordinate_transforms.q2R((poses.loc[0, 'qw'], poses.loc[0, 'qx'],
                                              poses.loc[0, 'qy'], poses.loc[0, 'qz']))
    return first_matrix

def load_poses(filename_poses, includes_translations=False):
    """
    gets poses from poses file
    :param filename_poses: filename of poses
    :return: data frame with poses
    """

    if includes_translations:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        poses['t'] = poses['sec'] + 1e-9 * (poses['nsec'])  # time_ctrl in MATLAB
    else:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['t', 'qx', 'qy', 'qz', 'qw'])

    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']]
    num_poses = poses.size

    return poses


def load_poses_sec(filename_poses, includes_translations=False):
    """
    gets poses from poses file
    :param filename_poses: filename of poses
    :return:
    """

    if includes_translations:
        print(pd.read_csv)
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    else:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['t', 'qx', 'qy', 'qz', 'qw'])

    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']]
    num_poses = poses.size

    return poses

def load_poses_angvel(filename_poses, includes_translations=True, t_first_event=None):
    """
    gets poses from poses file, includes angular velocities in all directions
    :param filename_poses: filename of poses
    :return: data frame with poses
    """

    if includes_translations:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['time', 'x', 'y', 'z', 'wx', 'wy', 'wz'])
    else:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['time', 'wx', 'wy', 'wz'])

    if t_first_event is None:
        poses['t'] = poses['time'] - poses['time'].loc[0]  # time_ctrl in MATLAB
    else:
        poses['t'] = poses['time'] - t_first_event  # time_ctrl in MATLAB

    poses = poses[['t', 'wx', 'wy', 'wz']]
    num_poses = poses.size

    return poses


def load_events(filename, davis=False, head=None, return_number=False):
    """
    Loads events in file specified by filename (txt file)
    :param davis:
    :param filename: filename to events.txt
    :return: events
    """
    print("Loading Events")

    if not davis:

    # Events have time in whole sec, time in ns, x in ]0, 127[, y in ]0, 127[
        events = pd.read_csv(filename, delimiter=' ',
                             header=None,
                             names=['sec', 'nsec', 'x', 'y', 'pol'])
        # print("Head: \n", events.head(10))
        num_events = events.count()
        print("Number of events in file: ", num_events)

        # Remove time of offset
        first_event_sec = events.loc[0, 'sec']
        first_event_nsec = events.loc[0, 'nsec']
        events['t'] = events['sec'] - first_event_sec + 1e-9 * (events['nsec'] - first_event_nsec)
        events = events[['t', 'x', 'y', 'pol']]

    else:
        print("DAVIS")
        events = pd.read_csv(filename, delimiter=' ',
                             header=None,
                             names=['time', 'x', 'y', 'pol'])
        # print("Head: \n", events.head(10))
        num_events = events.count()
        print("Number of events in file: ", num_events)
        first_event = events.loc[0, 'time']
        events['t'] = events['time'] - first_event
        events = events[['t', 'x', 'y', 'pol']]


    if return_number:
        if head is None:
            return events, num_events
        else:
            return events.head(head), len(events.head(head))
    else:
        if head is None:
            return events
        else:
            return events.head(head)


def generate_event(t=0, x=128/2, y=128/2, pol=1, corner=None):
    if corner is None:
        event = pd.Series({'t': t, 'x': x, 'y': y, 'pol': pol})
    elif corner == 0:
        event = pd.Series({'t': t, 'x': 0, 'y': 0, 'pol': pol})
    elif corner == 1:
        event = pd.Series({'t': t, 'x': 127, 'y': 0, 'pol': pol})
    elif corner == 2:
        event = pd.Series({'t': t, 'x': 127, 'y': 127, 'pol': pol})
    elif corner == 3:
        event = pd.Series({'t': t, 'x': 0, 'y': 127, 'pol': pol})
    elif corner == 4:
        event = pd.Series({'t': t, 'x': 64, 'y': 64, 'pol': pol})
    return event


def generate_events():
    '''

    :return: generate data frame with generated events
    '''
    cols = ['t', 'x', 'y', 'pol']
    list_of_series = [generate_event(corner=i) for i in range(5)]
    events = pd.DataFrame(list_of_series, columns=cols)
    return events


def rotmat2quaternion(rotmat):
    """
    Converts rotation matrix to quaternions in form (qx,qy,qz,qw)
    :param rotmat: 3x3 Rotation matrix
    :return: quaternion in form: (qx,qy,qz,qw)
    """
    qw = np.sqrt(1. + rotmat[0][0] + rotmat[1][1] + rotmat[2][2]) / 2.
    qx = (rotmat[2][1] - rotmat[1][2]) / (4. * qw)
    qy = (rotmat[0][2] - rotmat[2][0]) / (4. * qw)
    qz = (rotmat[1][0] - rotmat[0][1]) / (4. * qw)
    return qx, qy, qz, qw

def rotmat2eulerangles(R):
    """
    Calculates Euler angles from rotation matrix
    :param rotmat: rotation matrix
    :return: Euler angles theta_x, theta_y, theta_z
    """
    theta_x = np.arctan2(R[2][1], R[2][2])
    theta_y = np.arctan2(-R[2][0], np.sqrt(R[2][1]**2 + R[2][2]**2))
    theta_z = np.arctan2(R[1][0], R[0][0])

    return theta_x, theta_y, theta_z

def rotmat2eulerangles_df(df):
    """
    TODO: not tested. test! (for example by recreating rotation matrix R)
    From rotation matrices DataFrame, creates DataFrame with Euler angles theta_x, theta_y, theta_z
    :param df: DataFrame with rotation matrices
    :return: DataFrame with Euler angles theta_x, theta_y, theta_z
    """
    eulerangles = pd.DataFrame(columns = ['all', 'th_x', 'th_y', 'th_z'])
    eulerangles['all'] = df['Rotation'].apply(lambda R: rotmat2eulerangles(R))
    eulerangles['th_x'] = eulerangles['all'].apply(lambda row: row[0])
    eulerangles['th_y'] = eulerangles['all'].apply(lambda row: row[1])
    eulerangles['th_z'] = eulerangles['all'].apply(lambda row: row[2])
    eulerangles = eulerangles.drop(columns=['all'])
    return eulerangles

def get_sigmas(eulerangles, all_events=3564657, batch_size=300, factor=1):
    """
    calculates the standard deviation of the motion update
    :param eulerangles:
    :return: sigma1,sigma2,sigma3
    """
    print(eulerangles.diff().head(10))
    print(eulerangles.diff().abs().head(10))
    print(eulerangles.diff().tail(10))
    print()
    diffabs = eulerangles.diff().abs()


    print(eulerangles.diff().sum())
    dx = eulerangles.diff().abs().sum()['th_x']
    dy = eulerangles.diff().abs().sum()['th_y']
    dz = eulerangles.diff().abs().sum()['th_z'] - 2*np.pi

    dx = eulerangles.diff().sum()['th_x']
    dy = eulerangles.diff().sum()['th_y']
    dz = eulerangles.diff().sum()['th_z'] - 2*np.pi

    # print(eulerangles.diff().describe())
    # print(eulerangles.diff().abs().describe())
    #
    # dx = eulerangles['th_x'].max() - eulerangles['th_x'].min()
    # dy = eulerangles['th_y'].max() - eulerangles['th_y'].min()
    # dz = eulerangles['th_z'].max() - eulerangles['th_z'].min()

    # print(dx)
    # print(dy)
    # print(dz)

    num_batches = all_events/batch_size
    sigma_1 = factor * dx / num_batches
    sigma_2 = factor * dy / num_batches
    sigma_3 = factor * dz / num_batches

    return sigma_1, sigma_2, sigma_3


def rot2quaternions(allrotations):
    """
    Converts rotations to quaternions and saves in quaternions_[datestring].csv
    :param allrotations: all rotations
    :return: datestring
    """

    # Makes DataFrame with quaternions
    quaternions = pd.DataFrame(columns = ['t','qx','qy','qz','qw'])
    quaternion = allrotations['Rotation'].apply(lambda x: rotmat2quaternion(x))
    quaternions['t'] = allrotations['t']
    quaternions['qx'] = quaternion.str.get(0)
    quaternions['qy'] = quaternion.str.get(1)
    quaternions['qz'] = quaternion.str.get(2)
    quaternions['qw'] = quaternion.str.get(3)
    return quaternions

def quaternions2file(quaternions, directory):
    """

    :param quaternions: data frame with quaternion
    :param directory: path
    :return: datestring
    """
    # Saves quaternions as csv
    # Gets datestring
    now = datetime.datetime.now()
    datestring = now.strftime("%d%m%YT%H%M%S")
    filename = 'quaternions_' + datestring + '.txt'
    filename = os.path.join(directory, filename)
    quaternions.to_csv(filename, index=None, header=None, sep=' ', mode='a')
    return datestring


def write_logfile(datestring, directory, **kwargs):
    """
    Writes logfile from metadata
    :param datestring:
    :param kwargs: dictionary with metadata, such as num_events, num_batches, etc.
    :return: logfile
    """
    filename = 'quaternions_' + datestring + '.log'
    filename = os.path.join(directory, filename)


    with open(filename, 'a') as the_file:
        the_file.write("{0}: {1}\n".format('Datestring', datestring))
        for key, value in kwargs.items():
            print(key, ":", value)
            the_file.write("{0}: {1}\n".format(key, value))

def generate_random_rotmat(unit=False, seed=None):
    """
    Initializes random rotation matrix
    :param unit: returns unit matrix if True
    :param seed: Fixing the random seed to test function. None per default.
    :return: 3x3 np.array
    """
    if unit:
        M = np.eye(3)

    else:
        if seed is not None:
            np.random.seed(seed)

        G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        n1 = np.random.uniform(-np.pi, np.pi)
        n2 = np.random.uniform(-np.pi, np.pi)
        n3 = np.random.uniform(-np.pi, np.pi)

        M = sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))

    return M

if __name__ == '__main__':
    data_dir = '../data/Datasets/RedRoom/second'
    filename_poses = os.path.join(data_dir, 'poses_orb.txt')

    poses = load_poses_sec(filename_poses, includes_translations=True)
    poses['roll'] = poses.apply(lambda x: coordinate_transforms.q2roll(qw=x['qw'],
                                                                       qx=x['qx'],
                                                                       qy=x['qy'],
                                                                       qz=x['qz']),
                                axis=1)
    poses['pitch'] = poses.apply(lambda x: coordinate_transforms.q2pitch(qw=x['qw'],
                                                                         qx=x['qx'],
                                                                         qy=x['qy'],
                                                                         qz=x['qz']),
                                axis=1)
    poses['yaw'] = poses.apply(lambda x: coordinate_transforms.q2yaw(qw=x['qw'],
                                                                     qx=x['qx'],
                                                                     qy=x['qy'],
                                                                     qz=x['qz']),
                                axis=1)
    poses = poses[['t', 'roll', 'pitch', 'yaw']]
    print(poses.describe())
    print(poses.head())
    poses.to_csv('poses_orb_euler.txt', sep=' ', header=None)

    # all_events = load_events(filename_events, False, head=3624650, return_number=True)
    # # print(type(all_events))
    # print(all_events)
    # exit()
    # # first_matrix = get_first_matrix(filename_poses)
    # # print(first_matrix)
    # # print(all_events)
    #
    # # write_logfile('abcdefg',  a=23, b='hello', aa='oops')
    # poses = load_poses(filename_poses, includes_translations=True)
    # rotmats = coordinate_transforms.q2R_df(poses)
    # print(rotmats.loc[0]['Rotation'])
    #
    #
    # eulerangles = rotmat2eulerangles_df(rotmats)
    #
    # # print(eulerangles.head(10))
    # print(eulerangles.describe())
    # sigma_1, sigma_2, sigma_3 = get_sigmas(eulerangles, factor=1)
    # print("sigmas")
    # print(sigma_1)
    # print(sigma_2)
    # print(sigma_3)

    # data_dir = '../data/synth1'
    # filename_poses = os.path.join(data_dir, 'poses.txt')
    # filename_events = os.path.join(data_dir, 'events.txt')
    #
    # all_events = load_events(filename_events, False, head=3624650, return_number=True)
    # # print(type(all_events))
    # print(all_events)
    # exit()
    # # first_matrix = get_first_matrix(filename_poses)
    # # print(first_matrix)
    # # print(all_events)
    #
    # # write_logfile('abcdefg',  a=23, b='hello', aa='oops')
    # poses = load_poses(filename_poses, includes_translations=True)
    # rotmats = coordinate_transforms.q2R_df(poses)
    # print(rotmats.loc[0]['Rotation'])
    #
    #
    # eulerangles = rotmat2eulerangles_df(rotmats)
    #
    # # print(eulerangles.head(10))
    # print(eulerangles.describe())
    # sigma_1, sigma_2, sigma_3 = get_sigmas(eulerangles, factor=1)
    # print("sigmas")
    # print(sigma_1)
    # print(sigma_2)
    # print(sigma_3)






    #Test: TODO: Good opportunity to practice testing with a testing module. Should be unit matrix (or close to it)
    # print(np.dot(first_matrix.T, first_matrix))