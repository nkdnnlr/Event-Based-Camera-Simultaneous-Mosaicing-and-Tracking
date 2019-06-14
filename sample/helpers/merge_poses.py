import pandas as pd
import os
import glob

path = '/home/nik/UZH/NSC/3D Vision/Project/Event-Based-Camera-Simultaneous-Mosaicing-and-Tracking/data/Datasets/RedRoom/second/poses'
# path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(path + "/*.txt")

li = []
columns = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
# filename = os.path.join(path, 'KeyFrameTrajectory1.txt')
# df = pd.read_csv(filename)
# print(df)
# exit()

for filename in all_files:
    df = pd.read_csv(filename, names=columns, delim_whitespace=True)#, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.sort_values('t')

print(frame)
