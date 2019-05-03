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


if __name__ == '__main__':
    data_dir = '../data/synth1'
    filename_poses = os.path.join(data_dir, 'poses.txt')
    filename_events = os.path.join(data_dir, 'events.txt')

    # first_matrix = get_first_matrix(filename_poses)
    # all_events = load_events(filename_events, head=None, return_number=True)
    # print(all_events)




    #Test: TODO: Good opportunity to practice testing with a testing module. Should be unit matrix (or close to it)
    # print(np.dot(first_matrix.T, first_matrix))