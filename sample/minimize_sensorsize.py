import os
import pandas as pd
from helpers import load_events

# Todo: Finish
size=64

events, num_events = load_events('events.txt',head=None , return_number=True)

events_small_sensor = pd.DataFrame(columns=['t', 'x', 'y', 'pol'])

'''
def check_and_write(row):
    if row['x'].values <= size:
        df = events_small_sensor.append(row)
        return

print(check_and_write(events.iloc[[2]]))

events.iloc[[1]].apply(lambda r: check_and_write(r))
'''

for i in range(3624659):
    print(i)
    if events.iloc[[i]]['x'].values <= size and events.iloc[[i]]['y'].values <= size:
        events_small_sensor = events_small_sensor.append(events.iloc[[i]])
    else:
        None

print(events_small_sensor)

events_small_sensor.to_csv('/Users/JoelBachmann/Desktop/FS19/3D Vision/Project/Event-Based-Camera-Simultaneous-Mosaicing-and-Tracking/Tracking_Particle_Filter/events_small_sensor.txt', header=None, index=None, sep=' ', mode='a')
