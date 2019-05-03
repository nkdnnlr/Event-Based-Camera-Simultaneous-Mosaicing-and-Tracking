import os
import pandas as pd
from Tracking_Particle_Filter.tracking import load_events

# Todo: Finish
size=64

events, num_events = load_events('events.txt',head=None , return_number=True)

events_small_sensor = pd.DataFrame(columns=['sec', 'nsec', 'x', 'y', 'pol'])

def check_and_write(row):
    if row['x'] <= size:
        events_small_sensor.append(row)
        print('okok')
    return

check_and_write(events.iloc[2])

events_small_sensor.append(df2)
print(events_small_sensor)


# events.apply(lambda row check_and_write(row))
