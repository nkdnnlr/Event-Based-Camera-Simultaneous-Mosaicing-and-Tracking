import os
import time
import pandas as pd
from sample.helpers import load_events

starttime = time.time()


# Todo: Finish
size_x = 128
size_y = 128

center = True

directory = '../data/Datasets/BigRoom/2019-04-29-17-20-59'
events, num_events = load_events(os.path.join(directory, 'events.txt'), True, head=None, return_number=True)

x_min = events['x'].min()
x_max = events['x'].max()
y_min = events['y'].min()
y_max = events['y'].max()

x_min_cropped = (x_max - x_min + 1) / 2. + x_min - size_x / 2 - 1
x_max_cropped = (x_max - x_min + 1) / 2. + x_min + size_x / 2 - 1
y_min_cropped = (y_max - y_min + 1) / 2. + y_min - size_y / 2 - 1
y_max_cropped = (y_max - y_min + 1) / 2. + y_min + size_y / 2 - 1

print(events.describe())

print("Cropping...")

events = (events[(events.x >= x_min_cropped) & (events.x < x_max_cropped) &
                 (events.y >= y_min_cropped) & (events.y < y_max_cropped)].reset_index(drop=True))

events['x'] = events['x'] - x_min_cropped
events['y'] = events['y'] - y_min_cropped

print(events.describe())

print("Writing...")

events.to_csv(os.path.join(directory, 'events_cropped.txt'), header=None, index=None, sep=' ', mode='a')

print("\n Done after {} seconds.".format(time.time() - starttime))