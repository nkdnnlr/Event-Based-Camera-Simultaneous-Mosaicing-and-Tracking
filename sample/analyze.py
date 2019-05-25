import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sample.helpers as helpers


def score_quaternions(output_file, poses_gt, poses_tracker):
    """
    Get score from difference of groundtruth and tracked position
    :param poses_gt:
    :param poses_tracker:
    :return:
    """
    # print(poses_gt.describe())
    # print(poses_gt.head())
    # print(poses_tracker.describe())
    # print(poses_tracker.head())

    # Intrapolate. Assumption: Equal time difference between poses.
    for idx, row in poses_tracker.iterrows():
        poses_gt = poses_gt.append(pd.DataFrame({'t': [np.round(row['t'], 8)]}), ignore_index=True)
    poses_gt_intrapolated = poses_gt.sort_values(by='t').reset_index(drop=True)
    poses_gt_intrapolated = poses_gt_intrapolated.interpolate(method='linear')

    # Calculate RMSE between quaternions of tracker and groundtruth
    scores = pd.DataFrame(columns=['t', 'qw', 'qx', 'qy', 'qz', 'RMSE'])
    for idx, row in poses_tracker.iterrows():
        t = np.round(row['t'], 8)
        qw, qx, qy, qz, t = poses_gt_intrapolated[poses_gt_intrapolated['t'] == t].values[0]
        scores.loc[idx, 't'] = t

        scores.loc[idx, 'qw'] = np.sqrt((qw - row['qw'])**2)
        scores.loc[idx, 'qx'] = np.sqrt((qx - row['qx'])**2)
        scores.loc[idx, 'qy'] = np.sqrt((qy - row['qy'])**2)
        scores.loc[idx, 'qz'] = np.sqrt((qz - row['qz'])**2)

        scores.loc[idx, 'RMSE'] = np.sqrt((qw - row['qw'])**2 +
                                          (qx - row['qx'])**2 +
                                          (qy - row['qy'])**2 +
                                          (qz - row['qz'])**2)


    plt.figure()
    scores.plot('t', ['qw', 'qx', 'qy', 'qz', 'RMSE'])
    plt.xlabel('time (s)')
    plt.ylabel('absolute error')
    plt.legend()
    plt.title('Error on tracked poses compared to ground truth')
    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':
    directory_poses = '../output/poses/'
    filename_groundtruth = 'poses.txt'
    directory_output = '../output/evaluation'
    filename_onlymotionupdate = 'quaternions_11052019T150554_onlymotionupdate.txt'
    filename_ours = 'quaternions_22052019T153630_verynice.txt'
    filename_output = os.path.join(directory_output, filename_ours[:-4] + '.png')

    poses_groundtruth = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_groundtruth),
                                           includes_translations=True)
    poses_groundtruth['t'] -= poses_groundtruth['t'].loc[0]
    poses_ours = helpers.load_poses(filename_poses=os.path.join(directory_poses, filename_ours))


    score_quaternions(filename_output, poses_groundtruth, poses_ours)