import time
import sys
import math

import numpy as np
import pandas as pd
import scipy.linalg as sp
import math
import sys

from mpl_toolkits.mplot3d import Axes3D
# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# plotly.tools.set_credentials_file(username='huetufemchopf', api_key='iZv1LWlHLTCKuwM1HS4t')
from sys import platform as sys_pf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import sample.coordinate_transforms as coordinate_transforms


def load_file(filename, names = None):

    dataframe = pd.read_csv(filename, columns=names, delimiter = ' ')

    return dataframe


poses_theirs = pd.read_csv('poses.txt', names = ['time', 'x','y','z','qx','qy','qz','qw'])
poses_ours = pd.read_csv('quaternions.txt', names = ['time','qx','qy','qz','qw'])


# print(poses_ours)

