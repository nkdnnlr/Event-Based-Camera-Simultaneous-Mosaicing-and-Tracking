import time as time

import numpy as np
import pandas as pd
import scipy.linalg as sp
import math
import sys


intensity_map = np.load("../output/intensity_map.npy")

firstevents=np.array([[0, 1249173, 108, 112, 1],
                     [0, 1259493, 109, 109, 1]])

num_particles = 5
num_events_batch = 300


def camera_intrinsics():

    '''
    in: -
    out: Camera intrinsic Matrix K
    '''
    f_x=115.534  # x-focal length
    s=0 # Skewness
    x_0=79.262
    f_y=115.565  # y-focal length
    y_0=65.531
    
    K=np.array([[f_x,s,x_0],[0,f_y,y_0],[0,0,1]])
    return K

def event2angles(u, v, df_rotationmatrices, camera_intrinsicsK):
    '''
    in: event in camera frame (u,v), dataframe p generates a dataframe for all particles for 1 event
    out: event in rotational frame (theta,phi)
    '''
    df = pd.DataFrame(columns=['theta', 'phi'])


    df_coordinates = pd.DataFrame(columns = ['r_w1', 'r_w2', 'r_w3'])
    event_times_K = np.dot(np.linalg.inv(camera_intrinsicsK),np.array([[u],[v],[1]])) #from camera frame (u,v) to world reference frame
    p_w_temp = np.dot([df_rotationmatrices], event_times_K)

    df_rotationmatrices
    print(p_w_temp[:][0])
    exit()
    df_coordinates['r_w1'] = p_w_temp[:, 0]
    df_coordinates['r_w2'] = p_w_temp[:, 1]
    df_coordinates['r_w3'] = p_w_temp[:, 2]
    p_m = np.array([np.arctan(np.divide(df_coordinates['r_w1'], df_coordinates['r_w3'])),
                  np.arctan(np.divide(df_coordinates['r_w2'], np.sqrt(np.square(df_coordinates['r_w2']) + np.square(df_coordinates['r_w3']))))
                  ])          #from world reference frame to rotational frame (theta, phi)

    return p_m

def angles2map(theta, phi, height=1024, width=2048):
    """
    Converts angles (theta in [-pi, pi], phi in [-pi/2, pi/2])
    to integer map points (pixel coordinates)
    :param theta:
    :param phi:
    :param height: height of image in pixels
    :param width: width of image in pixels
    :return:
    """
    y = np.floor((-1*phi+np.pi/2)/np.pi*height)
    x = np.floor((theta + np.pi)/(2*np.pi)*width)
    return y, x

### PARTICLE FILTER ###

# define global variables:
         #amount of particles

def generate_random():
    G1 = np.matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G2 = np.matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    n1 = np.random.uniform(0.0, 2*np.pi)
    n2 = np.random.uniform(0.0, 2*np.pi)
    n3 = np.random.uniform(0.0, 2*np.pi)

    M = sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))

    return M

#initialize num_particles particles

def init_particles(N):
    '''
    in: # particles num_particles
    out: data frame with Index, Rotation matrix and weight
    '''
    # p0 = np.eye(3)      #initial rotation matrix of particles
    df = pd.DataFrame(columns=['Index', 'Rotation', 'Weight'])
    df['Rotation'] = df['Rotation'].astype(object)
    w0=1/N
    for i in range(N):
        df.at[i, ['Index']] = int(i)
        df.at[i, ['Rotation']] = [generate_random()]
        df.at[i, ['Weight']] = float(w0)

    return df


tau=firstevents[1][1]-firstevents[0][1]         #time between events
tau_c=2000                                      #time between events in same pixel

def update_step(p):
    '''
    in: particles
    out: particles, updated
    '''
    G1 = np.matrix([[0, 0, 0],[0, 0,-1],[0,1,0]])
    G2 = np.matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    sigma=0.0002
    p_u=[]
    for i in range(num_particles):
        n1=np.random.normal(0.0,sigma**2 * tau)
        n2=np.random.normal(0.0,sigma**2 * tau)
        n3=np.random.normal(0.0,sigma**2 * tau)
        p_u.append(np.dot( p[i] , sp.expm(np.dot(n1,G1) + np.dot(n2,G2) + np.dot(n3,G3))))
    return(p_u)



def measurement_update(event, particle_rm_t, particle_rm_t_minus_tc,  weigths):
    """
    :param event: 0, time, x ,y , a bunch of things
    :param particle_rm: particle rotation matrix
    :param weigths: integer weight
    :return: updated particle weights
    """

    p_m = []


    for i in range(len(particle_rm_t)):
        pw1t, pw2t, pw3t = particle_rm_t.dot(camera_intrinsics(event[2],event[3],1))
        pw1_tmintc, pw2_tmintc, pw3_tmintc = particle_rm_t_minus_tc.dot(camera_intrinsics(event[2], event[3], 1))

        pm1t = math.atan(pw1/pw3)
        pm2t = math.atan(pw2 / math.sqrt(pw1**2 + pw3**2))
        pm1_tmintc = math.atan(pw1_tmintc / pw3_tmintc)
        pm2_tmintc = math.atan(pw2_tmintc / math.sqrt(pw1_tmintc ** 2 + pw3_tmintc ** 2))

        # pm12 = [pm1t, pm2t]


def load_events(filename):
    print("Loading Events")
    # Events have time in whole sec, time in ns, x in ]0, 127[, y in ]0, 127[
    events = pd.read_csv(filename, delimiter=' ', header=None, names=['sec', 'nsec', 'x', 'y', 'pol'])
    # print("Head: \n", events.head(10))
    num_events = events.size
    print("Number of events in file: ", num_events)

    # Remove time of offset
    first_event_sec = events.loc[0, 'sec']
    first_event_nsec = events.loc[0, 'nsec']
    events['t'] = events['sec'] - first_event_sec + 1e-9 * (events['nsec'] - first_event_nsec)
    events = events[['t', 'x', 'y', 'pol']]
    print("Head: \n", events.head(10))
    print("Tail: \n", events.tail(10))
    # print(events['0])
    return events


def initialize_pixelmap(sensor_height, sensor_width):
    """
    Initializes pixelmap, which is a
    tensor of size sensorwidth*sensorheight*2,
    with tuple (t,pol) as entries
    :param sensor_height:
    :param sensor_width:
    :return:
    """
    pixelmap_t = np.zeros((sensor_height, sensor_width),
                          dtype=[('time', 'f8'), ('polarity', 'i4')])
    pixelmap_tc = np.zeros((sensor_height, sensor_width),
                           dtype=[('time', 'f8'), ('polarity', 'i4')])
    pixelmap = np.array([pixelmap_t, pixelmap_tc])
    return pixelmap


def update_pixelmap(pixelmap, event):
    """
    Updates pixelmap for each event. Saves event at t and t-t_c
    Runtime: ~200seconds for all events
    :param pixelmap: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
    :param event: Pandas Series with ['t', 'x', 'y', 'pol']
    :return:
    """
    x = int(event['x'])
    y = int(event['y'])
    pixelmap[1][y, x] = pixelmap[0][y, x]
    pixelmap[0][y, x] = (event['t'], event['pol'])
    return

def update_pixelmap_from_batch(pixelmap, batch_event):
    """
    Updates pixelmap for each event. Saves event at t and t-t_c
    :param pixelmap: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
    :param event: Pandas DataFrame with ['t', 'x', 'y', 'pol']
    :return:
    """
    for event in batch_event:
        x = int(event['x'])
        y = int(event['y'])
        pixelmap[1][y, x] = pixelmap[0][y, x]
        pixelmap[0][y, x] = (event['t'], event['pol'])
    return


if __name__ == '__main__':

    events = load_events('../data/synth1/events.txt')

    pixelmap = initialize_pixelmap(128, 128)
    # starttime = time.time()
    # i = 0
    # for idx, event in events.iterrows():
    #     # print(event)
    #     update_pixelmap(pixelmap=pixelmap, event=event)
    #     # i += 1
    #     # if i >= 10:
    #     #     break
    # endtime = time.time() - starttime
    # print("Endtime: ", endtime)

    ## Testing pixelmap
    # event = events.loc[557]
    # print(event)
    # pixelmap_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # pixelmap_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # pixelmap_ttc = np.array([pixelmap_t, pixelmap_tc])
    # print(pixelmap_ttc)
    # update_pixelmap(pixelmap_ttc, event)
    # print(pixelmap_ttc)
    #
    # ## Testing pixelmap
    # # event = events.loc[557]
    # # print(event)
    # # pixelmap_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # pixelmap_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # pixelmap_ttc = np.array([pixelmap_t, pixelmap_tc])
    # # print(pixelmap_ttc)
    # # update_pixelmap(pixelmap_ttc, event)
    # # print(pixelmap_ttc)
    # #
    # # # print(pixelmap[1])
    # # print(pixelmap_ttc[1][0,0])
    # #
    # #
    #
    #
    # # state update step
    camera_intrinsicsK = camera_intrinsics()
    particles= init_particles(num_particles)
    test = event2angles(events['x'][0], events['y'][0], particles['Rotation'], camera_intrinsicsK)
    print(test)
