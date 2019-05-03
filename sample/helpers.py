import os
import numpy as np
import pandas as pd
import sample.coordinate_transforms as coordinate_transforms
import datetime


def get_first_matrix(filename_poses):
    """
    gets first matrix from poses file
    :param filename_poses: filename of poses
    :return:
    """
    poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    num_poses = poses.size
    # print("Number of poses in file: ", num_poses)

    poses['t'] = poses['sec'] + 1e-9*(poses['nsec']) #time_ctrl in MATLAB
    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']] # Quaternions
    # print("Head: \n", poses.head(10))
    # print("Tail: \n", poses.tail(10))

    first_matrix = coordinate_transforms.q2R((poses.loc[0, 'qw'], poses.loc[0, 'qx'],
         poses.loc[0, 'qy'], poses.loc[0, 'qz']))
    return first_matrix

def load_poses(filename_poses, includes_translations=False):
    """
    gets first matrix from poses file
    :param filename_poses: filename of poses
    :return:
    """

    if includes_translations:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        poses['t'] = poses['sec'] + 1e-9 * (poses['nsec'])  # time_ctrl in MATLAB
    else:
        poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['t', 'qx', 'qy', 'qz', 'qw'])

    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']]
    num_poses = poses.size

    return poses

def load_events(filename, head=None, return_number=False):
    """
    Loads events in file specified by filename (txt file)
    :param filename: filename to events.txt
    :return: events
    """
    print("Loading Events")
    # Events have time in whole sec, time in ns, x in ]0, 127[, y in ]0, 127[
    events = pd.read_csv(filename, delimiter=' ', header=head, names=['sec', 'nsec', 'x', 'y', 'pol'])
    # print("Head: \n", events.head(10))
    num_events = events.count()
    print("Number of events in file: ", num_events)

    # Remove time of offset
    first_event_sec = events.loc[0, 'sec']
    first_event_nsec = events.loc[0, 'nsec']
    events['t'] = events['sec'] - first_event_sec + 1e-9 * (events['nsec'] - first_event_nsec)
    events = events[['t', 'x', 'y', 'pol']]
    # print("Head: \n", events.head(10))
    # print("Tail: \n", events.tail(10))
    # print(events['0])
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


def rotmat2quaternion(rotmat):
    """
    Converts rotation matrix to quaternions in form (qx,qy,qz,qw)
    :param rotmat: 3x3 Rotation matrix
    :return: quaternion in form: (qx,qy,qz,qw)
    """
    qw = np.sqrt(1 + rotmat[0][0] + rotmat[1][1] + rotmat[2][2]) / 2
    qx = (rotmat[2][1] - rotmat[1][2]) / (4 * qw)
    qy = (rotmat[0][2] - rotmat[2][0]) / (4 * qw)
    qz = (rotmat[1][0] - rotmat[0][1]) / (4 * qw)
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



def write_quaternions2file(allrotations):
    """
    Converts rotations to quaternions and saves in quaternions_[datestring].csv
    :param allrotations: all rotations
    :return: datestring
    """
    # Gets datestring
    now = datetime.datetime.now()
    datestring = now.strftime("%d%m%YT%H%M%S")
    filename = 'quaternions_' + datestring + '.txt'

    # Makes DataFrame with quaternions
    quaternions = pd.DataFrame(columns = ['t','qx','qy','qz','qw'])
    quaternion = allrotations['Rotation'].apply(lambda x: rotmat2quaternion(x))
    quaternions['t'] = allrotations['t']
    quaternions['qx'] = quaternion.str.get(0)
    quaternions['qy'] = quaternion.str.get(1)
    quaternions['qz'] = quaternion.str.get(2)
    quaternions['qw'] = quaternion.str.get(3)

    # Saves quaternions as csv
    quaternions.to_csv(filename, index=None, header=None, sep=' ', mode='a')
    return datestring

def write_logfile(datestring, **kwargs):
    """
    Writes logfile from metadata
    :param datestring:
    :param kwargs: dictionary with metadata, such as num_events, num_batches, etc.
    :return:
    """
    filename = 'quaternions_' + datestring + '.log'

    with open(filename, 'a') as the_file:
        the_file.write("{0}: {1}\n".format('Datestring', datestring))
        for key, value in kwargs.items():
            print(key, ":", value)
            the_file.write("{0}: {1}\n".format(key, value))



if __name__ == '__main__':
    data_dir = '../data/synth1'
    filename_poses = os.path.join(data_dir, 'poses.txt')
    filename_events = os.path.join(data_dir, 'events.txt')

    # first_matrix = get_first_matrix(filename_poses)
    # print(first_matrix)
    # all_events = load_events(filename_events, head=None, return_number=True)
    # print(all_events)

    # write_logfile('abcdefg',  a=23, b='hello', aa='oops')
    poses = load_poses(filename_poses, includes_translations=True)
    rotmats = coordinate_transforms.q2R_df(poses)
    print(rotmats.loc[0]['Rotation'])


    eulerangles = rotmat2eulerangles_df(rotmats)
    # print(eulerangles.head(10))
    print(eulerangles.describe())






    #Test: TODO: Good opportunity to practice testing with a testing module. Should be unit matrix (or close to it)
    # print(np.dot(first_matrix.T, first_matrix))