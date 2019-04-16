import time
import sys
import math

import numpy as np
import pandas as pd
import scipy.linalg as sp
import matplotlib.pyplot as plt


intensity_map = np.load("../output/intensity_map.npy")

# TODO: Change!
firstevents=np.array([[0, 1249173, 108, 112, 1],
                     [0, 1259493, 109, 109, 1]])

num_particles = 500
num_events_batch = 300
tau=firstevents[1][1]-firstevents[0][1]         #time between events
# tau_c=2000                                      #time between events in same pixel
mu = 0.22
sigma = 8.0*10**(-2)
minimum_constant = 1e-3

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

def event2angles(event, df_rotationmatrices, calibration):
    """
    For a given event, generates dataframe
    with particles as rows and angles as columns.
    :param event: Event in camera frame
    :param df_rotationmatrices:
    :param calibration:
    :return:
    """
    event_times_K = np.dot(np.linalg.inv(calibration), np.array([[event['x']], [event['y']], [1]])) #from camera frame (u,v) to world reference frame
    coordinates = ['r_w1', 'r_w2', 'r_w3']
    df_coordinates = pd.DataFrame.from_records(df_rotationmatrices.apply(lambda x: np.dot(x, event_times_K)),
                                               columns=coordinates)
    df_coordinates['r_w1'] = df_coordinates['r_w1'].str.get(0)
    df_coordinates['r_w2'] = df_coordinates['r_w2'].str.get(0)
    df_coordinates['r_w3'] = df_coordinates['r_w3'].str.get(0)

    # from world reference frame to rotational frame (theta, phi)
    df_angles = pd.DataFrame(columns=['theta', 'phi'])
    df_angles['theta'] = np.arctan(df_coordinates['r_w1'] / df_coordinates['r_w3'])
    df_angles['phi'] = np.arctan(df_coordinates['r_w2'] / np.sqrt(df_coordinates['r_w1']**2 + df_coordinates['r_w3']**2))

    return df_angles


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


def particles_per_event2map(event, particles, calibration):
    """
    For each event, gets map angles and coordinates (for on panoramic image)
    :param event:
    :param particles:
    :param calibration:
    :return:
    """
    particles_per_event = event2angles(event, particles['Rotation'], calibration)
    particles_per_event['v'], particles_per_event['u'] = zip(*particles_per_event.apply(
        lambda row: angles2map(row['theta'], row['phi']), axis=1))
    return particles_per_event


### PARTICLE FILTER ###

# define global variables:
         #amount of particles

def generate_random_rotmat(seed = None):
    """
    Initializes random rotation matrix
    :param seed: Fixing the random seed to test function. None per default.
    :return: 3x3 np.array
    """
    if seed is not None:
        np.random.seed(seed)

    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

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
    TODO: Add time to particle. For example, make particles every 1/1000 s. We will need to save particles somewhere.
    '''
    # p0 = np.eye(3)      #initial rotation matrix of particles
    df = pd.DataFrame(columns=['Rotation', 'Weight'])
    df['Rotation'] = df['Rotation'].astype(object)
    w0 = 1/N
    for i in range(N):
        # TODO: random seed is fixed now. Change again!
        df.at[i, ['Rotation']] = [generate_random_rotmat(seed=None)]
        df.at[i, ['Weight']] = float(w0)
    return df


def update_step(particle):
    '''
    in: particles
    out: particles, updated
    '''
    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    sigma1 = 2.3e-8
    sigma2 = 5.0e-6
    sigma3 = 7e-5
    p_u=[]
    for i in range(num_particles):
        n1 = np.random.normal(0.0, sigma1**2 * tau)
        n2 = np.random.normal(0.0, sigma2**2 * tau)
        n3 = np.random.normal(0.0, sigma3**2 * tau)
        p_u.append(np.dot(particle[i], sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))))
    return p_u


def get_latest_particles(t_asked, particles_all_time):
    """
    From list of particles over all times,
    get set of particles that was just before asked t_asked
    :param t_asked: t_asked of interest (e.g. t_asked of current event)
    :param particles_all_time:
    :return:
    """
    dt_pos = 1e-4 #TODO: Write as class variable (also dt_pos_inv)
    dt_pos_inv = 1. / dt_pos
    t_particles = math.floor(t_asked * dt_pos_inv) / dt_pos
    return particles_all_time[particles_all_time['t'] == t_particles]


def get_pixelmap_for_particles(event, sensormap, particles_all_time):
    """
    Working on...
    :param event:
    :param sensormap:
    :param particles_all_time:
    :return:
    """
    t = event['t']
    x = event['x']
    y = event['y']
    ttc = sensormap[1][x,y][0]

    particles = get_latest_particles(t_asked=t, particles_all_time=particles_all_time)

    particles_ttc = get_latest_particles(t_asked=ttc, particles_all_time=particles_all_time)

    particles_per_event2map(event, particles, camera_intrinsicsK)

    pass




def measurement_update_temp(event, particle, pixelmap):
    """
    1. Rotate event with rotation matrix from particle.
    :param event:
    :param particle:
    :param pixelmap:
    :return:
    """
    pass


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
    # print("Head: \n", events.head(10))
    # print("Tail: \n", events.tail(10))
    # print(events['0])
    return events


def mexhat(t, sigma=8.0*10e-2, k_e = 1.0*10e-3, Ce = 0.22):

    c = 2. / math.sqrt(3 * sigma) * (math.pi ** 0.25)
    return c * (1 - t ** 2 / sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))


def initialize_sensormap(sensor_height, sensor_width):
    """
    Initializes sensormap, which is a
    tensor of size sensorwidth*sensorheight*2,
    with tuple (t,pol) as entries
    :param sensor_height:
    :param sensor_width:
    :return:
    """
    sensormap_t = np.zeros((sensor_height, sensor_width),
                          dtype=[('time', 'f8'), ('polarity', 'i4')])
    sensormap_tc = np.zeros((sensor_height, sensor_width),
                           dtype=[('time', 'f8'), ('polarity', 'i4')])
    sensormap = np.array([sensormap_t, sensormap_tc])
    return sensormap


def update_sensormap(sensormap, event):
    """
    Updates sensormap for each event. Saves event at t and t-t_c
    Runtime: ~200seconds for all events
    :param sensormap: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
    :param event: Pandas Series with ['t', 'x', 'y', 'pol']
    :return:
    """
    x = int(event['x'])
    y = int(event['y'])
    sensormap[1][y, x] = sensormap[0][y, x]
    sensormap[0][y, x] = (event['t'], event['pol'])
    return


def update_sensormap_from_batch(sensormap, batch_event):
    """
    Updates sensormap for each event. Saves event at t and t-t_c
    :param sensormap: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
    :param event: Pandas DataFrame with ['t', 'x', 'y', 'pol']
    :return:
    """
    for event in batch_event:
        x = int(event['x'])
        y = int(event['y'])
        sensormap[1][y, x] = sensormap[0][y, x]
        sensormap[0][y, x] = (event['t'], event['pol'])
    return


def event_likelihood(z, mu, sigma, k_e):
    """
    For a given absolute log intensity difference z,
    returns the likelihood of an event.
    likelihood = normalize(gaussian distribution + noise)
    :param z: log intensity difference
    :param mu: mean
    :param sigma: standard deviation
    :param k_e: minimum constant / noise
    :return:
    """
    y = k_e + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(z - mu) ** 2 / (2 * sigma) ** 2)
    return y/np.max(y)


if __name__ == '__main__':

    events = load_events('../data/synth1/events.txt')

    sensormap = initialize_sensormap(128, 128)
    starttime = time.time()
    i = 0
    for idx, event in events.head(50000).iterrows():
        # print(event)
        update_sensormap(sensormap=sensormap, event=event)
        # i += 1
        # if i >= 10:
        #     break
    endtime = time.time() - starttime
    print("Endtime: ", endtime)
    print(sensormap[0][0,0][0])
    exit()

    ## Testing sensormap
    # event = events.loc[557]
    # print(event)
    # sensormap_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # sensormap_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # sensormap_ttc = np.array([sensormap_t, sensormap_tc])
    # print(sensormap_ttc)
    # update_sensormap(sensormap_ttc, event)
    # print(sensormap_ttc)
    #
    # ## Testing sensormap
    # # event = events.loc[557]
    # # print(event)
    # # sensormap_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # sensormap_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # sensormap_ttc = np.array([sensormap_t, sensormap_tc])
    # # print(sensormap_ttc)
    # # update_sensormap(sensormap_ttc, event)
    # # print(sensormap_ttc)
    # #
    # # # print(sensormap[1])
    # # print(sensormap_ttc[1][0,0])
    # #
    # #
    #
    #
    #
    # # state update step
    camera_intrinsicsK = camera_intrinsics()
    particles= init_particles(num_particles)
    print(particles)
    # particles_per_event = event2angles(events.loc[0], particles['Rotation'], camera_intrinsicsK)
    # particles_per_event['v'], particles_per_event['u'] = zip(*particles_per_event.apply(
    #     lambda row: angles2map(row['theta'], row['phi']), axis=1))
    particles_per_event = particles_per_event2map(events.loc[0], particles, camera_intrinsicsK)
    print(particles_per_event)
    plt.figure(1)
    plt.scatter(particles_per_event['theta'], particles_per_event['phi'])
    plt.show()
    plt.figure(2)
    plt.scatter(particles_per_event['u'], particles_per_event['v'])
    plt.show()

    #
    # # particles= init_particles(N)
    #
    # t = []
    # tt = range(1, 100)
    # for i in tt:
    #     t.append(mexhat(i))
    #
    # plt.plot(tt, t)
    # plt.show()

