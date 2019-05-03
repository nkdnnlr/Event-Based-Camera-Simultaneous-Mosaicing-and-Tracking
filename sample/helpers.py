import os
import numpy as np
import pandas as pd
import sample.coordinate_transforms as coordinate_transforms


def get_first_matrix(filename_poses):
    """
    gets first matrix from poses file
    :param filename_poses: filename of poses
    :return:
    """
    poses = pd.read_csv(filename_poses, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    num_poses = poses.size
    print("Number of poses in file: ", num_poses)

    poses['t'] = poses['sec'] + 1e-9*(poses['nsec']) #time_ctrl in MATLAB
    poses = poses[['t', 'qw', 'qx', 'qy', 'qz']] # Quaternions
    print("Head: \n", poses.head(10))
    print("Tail: \n", poses.tail(10))

    first_matrix = coordinate_transforms.q2R((poses.loc[0, 'qw'], poses.loc[0, 'qx'],
         poses.loc[0, 'qy'], poses.loc[0, 'qz']))
    return first_matrix

if __name__ == '__main__':
    data_dir = '../data/synth1'
    filename_poses = os.path.join(data_dir, 'poses.txt')
    first_matrix = get_first_matrix(filename_poses)
    # print(first_matrix)


    #Test: TODO: Good opportunity to practice testing with a testing module. Should be unit matrix (or close to it)
    # print(np.dot(first_matrix.T, first_matrix))