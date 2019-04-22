import time
import sys
import math

import numpy as np
import pandas as pd
import scipy.linalg as sp
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
# plotly.tools.set_credentials_file(username='huetufemchopf', api_key='iZv1LWlHLTCKuwM1HS4t')
import matplotlib.pyplot as plt

event_file = '../data/synth1/events.txt'
intensity_map = np.load('../output/intensity_map.npy')


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
    f_x = 115.534  # x-focal length
    s = 0  # Skewness
    x_0 = 79.262
    f_y = 115.565  # y-focal length
    y_0 = 65.531

    K = np.array([[f_x, s, x_0], [0, f_y, y_0], [0, 0, 1]])
    return K


def load_events(filename, head = None):
    """
    Loads events in file specified by filename (txt file)
    :param filename:
    :return: events
    """
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
    if head is None:
        return events
    else:
        return events.head(head)

def event_and_particles_to_angles(event, df_rotationmatrices, calibration):
    """
    For a given event, generates dataframe
    with particles as rows and angles as columns.
    :param event: Event in camera frame
    :param df_rotationmatrices: DataFrame with rotation matrices
    :param calibration: camera calibration
    :return: DataFrame with particles as rows and angles as columns.
    """
    event_times_K = np.dot(np.linalg.inv(calibration), np.array([[event['x']], [event['y']], [1]])) #from camera frame (u,v) to world reference frame
    coordinates = ['r_w1', 'r_w2', 'r_w3']
    df_coordinates = pd.DataFrame.from_records(df_rotationmatrices.apply(lambda x: np.dot(x, event_times_K)),
                                               columns=coordinates)
    # print(df_coordinates['r_w1'])
    # print(df_coordinates['r_w1'].str.get(0))
    # print(df_coordinates['r_w1'].str.get(1))

    df_coordinates['r_w1'] = df_coordinates['r_w1'].str.get(0) # ATTENTION: This is tested and correct. str.get(0) just removes brackets. See output above.
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
    :return: tuple with integer map points (pixel coordinates)
    """
    y = np.floor((-1*phi+np.pi/2)/np.pi*height)
    x = np.floor((theta + np.pi)/(2*np.pi)*width)
    return y, x


def particles_per_event2map(event, particles, calibration):
    """
    For each event, gets map angles and coordinates (for on panoramic image)
    :param event: one event
    :param particles: dataframe with particles
    :param calibration:
    :return:  DataFrame with particles as rows and as columns theta, phi, v, u (coordinates)
    """
    particles_per_event = event_and_particles_to_angles(event, particles['Rotation'], calibration)
    particles_per_event['v'], particles_per_event['u'] = zip(*particles_per_event.apply(
        lambda row: angles2map(row['theta'], row['phi']), axis=1))
    particles_per_event['pol'] = event['pol']
    return particles_per_event


def generate_random_rotmat(unit=False, seed = None):
    """
    Initializes random rotation matrix
    :param unit: returns unit matrix if True
    :param seed: Fixing the random seed to test function. None per default.
    :return: 3x3 np.array
    """
    if unit:
        M = np.eye(3)

    else:
        if seed is not None:
            np.random.seed(seed)

        G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        n1 = np.random.uniform(-np.pi, np.pi)
        n2 = np.random.uniform(-np.pi, np.pi)
        n3 = np.random.uniform(-np.pi, np.pi)

        M = sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))

    return M

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
        df.at[i, ['Rotation']] = [generate_random_rotmat(unit=False, seed=None)]
        df.at[i, ['Weight']] = float(w0)
    return df

    # print(events)


### PARTICLE FILTER ###

# define global variables:
         #amount of particles


##initialize num_particles particles



def motion_update(particles):
    '''
    in: particles as data frame with Rotation and Weight
    out: particles, updated
    '''
    updated_particles = particles.copy()  # type: object
    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    # TODO: update sigma and tau!!
    sigma1 = 2.3e-8
    sigma2 = 5.0e-6
    sigma3 = 7e-5
    # p_u=[]
    # for i in range(len(particles)):
        # n1 = np.random.normal(0.0, sigma1**2 * tau)
        # n2 = np.random.normal(0.0, sigma2**2 * tau)
        # n3 = np.random.normal(0.0, sigma3**2 * tau)
        # p_u.append(np.dot(particle[i], sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))))
    updated_particles['Rotation'] = updated_particles['Rotation'].apply(
        lambda x: np.dot(x, sp.expm(np.dot(np.random.normal(0.0, sigma1**2 * tau), G1)
                                    + np.dot(np.random.normal(0.0, sigma2**2 *tau), G2)
                                    + np.dot(np.random.normal(0.0,sigma3**2 * tau), G3))))
    return updated_particles


def initialize_sensortensor(sensor_height, sensor_width):
    """
    Initializes sensortensor, which is a
    tensor of size  2 x sensorwidth x sensorheight,
    with tuple (t,pol) as entries
    :param sensor_height:
    :param sensor_width:
    :return: initial sensortensor np.array([.. ])->128*128 pixels
    """
    sensortensor_t = np.zeros((sensor_height, sensor_width),
                          dtype=[('time', 'f8'), ('polarity', 'i4')])
    sensortensor_tc = np.zeros((sensor_height, sensor_width),
                           dtype=[('time', 'f8'), ('polarity', 'i4')])
    sensortensor = np.array([sensortensor_t, sensortensor_tc])
    return sensortensor


def update_sensortensor(sensortensor, event):
    """
    Updates sensortensor for each event. Saves event at t and t-t_c
    Runtime: ~200seconds for all events
    :param sensortensor: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
    :param event: Pandas Series with ['t', 'x', 'y', 'pol']
    :return: void
    """
    x = int(event['x'])
    y = int(event['y'])
    sensortensor[1][y, x] = sensortensor[0][y, x]
    sensortensor[0][y, x] = (event['t'], event['pol'])
    return


def get_latest_particles(t_asked, particles_all_time):
    """
    From list of particles over all times
    (one per timestep/batch, already resampled),
    get set of particles that was just before asked t_asked
    :param t_asked: t_asked of interest (e.g. t_asked of current event)
    :param particles_all_time:
    :return: the particle that came before a time-of-interest.
    """
    dt_pos = 1e-4 #TODO: Write as class variable (also dt_pos_inv)
    dt_pos_inv = 1. / dt_pos
    t_particles = math.floor(t_asked * dt_pos_inv) / dt_pos
    return particles_all_time[particles_all_time['t'] == t_particles]


def get_intensity_from_gradientmap(gradientmap, u, v):
    """
    Gets
    :param gradientmap:
    :param x:
    :param y:
    :return:
    """
    return gradientmap[v, u]

def event_likelihood(z, mu=0.22, sigma=8.0*1e-2, k_e=1.0*1e-3):
    """
    For a given absolute log intensity difference z,
    returns the likelihood of an event.
    likelihood = normalize(gaussian distribution + noise)
    :param z: log intensity difference
    :param mu: mean
    :param sigma: standard deviation
    :param k_e: minimum constant / noise
    :return: event-likelihood (scalar)
    """
    y = k_e + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(z - mu) ** 2 / (2 * sigma) ** 2)
    return y/np.max(y)



def measurement_update(event_batch,
                            particles,
                            all_rotations,
                            sensortensor):
    """
    Working on... not tested yet
    :param event_batch:
    :param particles: the 50 or so particles with attributes ['rotation']
    :param sensortensor:
    :param pixelmap:
    :return:
    """
    particles['z'] = 0
    particles['weight'] = []
    for event in event_batch:
        update_sensortensor(sensortensor, event)
        x = event['x']
        y = event['y']
        t = sensortensor[0][y, x][0]
        tminustc = sensortensor[1][y, x][0]
        particle_ttc = get_latest_particles(tminustc, particles_all_time=all_rotations) # single rotationmatrix before ttc
        pm_t = particles_per_event2map(event, particles, calibration)[['u', 'v']]
        pm_ttc = particles_per_event2map(event, particle_ttc, calibration)[['u', 'v']]
        print("PM_t", pm_t)
        print("PM_ttc", pm_ttc)
        u_ttc = pm_ttc.at[0, 'u']
        v_ttc = pm_ttc.at[0, 'v']
        print(u_ttc)
        print(v_ttc)
        pm_t['logintensity_ttc'] = intensity_map[v_ttc, u_ttc]
        pm_t['logintensity_t'] = pm_t.apply(lambda row: intensity_map[row['u'], row['v']])
        particles['z'] = pm_t['logintensity_t'] - pm_t['logintensity_ttc']
        particles['weight'].append(particles['z'].apply(lambda z: event_likelihood(z)))
    particles['weight'] = particles['weight'].mean(axis=1) ##not tested, probably wrong
    ### Delete ['z'] column
    return particles





def normalize_particle_weights(particles):
    '''
    normalizes particle weights
    :param particles: particles
    :return: particles with normalized weight (sum of weights = 1)
    '''
    s=0
    for i in range(len(particles)):
        s += particles.loc[i, 'Weight']

    for i in range(len(particles)):
        particles.loc[i, 'Weight']=particles.loc[i, 'Weight']/s

    return particles


def resampling(particles):
    '''
    resamples particles
    :param particles: tuple of N particles: (rotmat, normalized weight)
    :return: resampled particles, weighted average
    '''
    sum_of_weights=particles['Weight'].cumsum(axis=0)

    resampled_particles = pd.DataFrame(columns=['Rotation', 'Weight'])
    resampled_particles['Rotation'] = resampled_particles['Rotation'].astype(object)
    '''
    for i in range(len(particles)):     # i: resampling for each particle
        r = np.random.uniform(0, 1)
        for n in range(len(particles)):
            if sum_of_weights[n] >= r and n==0:
                n_tilde=n
            if sum_of_weights[n] >= r and r > sum_of_weights[n - 1]:
                n_tilde=n

        resampled_particles.at[i, ['Rotation']] = [particles.loc[n_tilde, 'Rotation']]
        resampled_particles.at[i, ['Weight']]=float(1/len(particles))
    '''
    i,j=0,0

    for i in range(len(particles)):
        r = np.random.uniform(0,1)
        

    return resampled_particles

particles=init_particles(5)
print(resampling(particles))

def test_distributions_rotmat(rotation_matrices):
    """
    :return: function checks whether the rotation matrices are really randomly distributed. muoltiplies rot matrix with Z-unit-vector. returns plotly and matplotlib plot which shows the distribution

    Function checks whether the rotation matrices are really randomly distributed.
    multiplies rot matrix with Z-unit-vector.
    :return: plotly and matplotlib plot which shows the distribution
    """

    vec = np.array([1,0,0]).T
    vecM = rotation_matrices.apply(lambda x: np.dot(x, vec))
    rotX = vecM.str.get(0)
    rotY = vecM.str.get(1)
    rotZ = vecM.str.get(2)

    trace1 = go.Scatter3d(
        x=rotX,
        y=rotY,
        z=rotZ,
        mode='markers',
        marker=dict(
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.1
            ),
            opacity=0.8
        )
    )


    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')

    ax = plt.axes(projection='3d')
    ax.scatter3D(rotX, rotY, rotZ, c=rotZ, cmap='Greens')
    ax.scatter3D([1], [0], [0], 'b')
    plt.show()


'''
if __name__ == '__main__':
    calibration = camera_intrinsics()
    event_batch = load_events(event_file, 300)
    particles = init_particles(1)
    print(particles)
    sensortensor = initialize_sensortensor(128, 128)

    #
    # measurement_update_temp(event_batch, particles,
    #                         all_rotations, sensortensor)

    particles1 = init_particles(1)
    particles5 = init_particles(5)
    print(particles1.at[0, 'Weight'])
    # print(particles1['Weight'].tolist())
    exit()
    particles5['Difference'] = particles5['Weight'] - particles1['Weight'].tolist()
    print(particles5)
    exit()

    ### Testing the event stream and pixelmap. TODO: Something is flipped. Else looks alright.
    fig_sensor = plt.figure(1)
    plt.scatter(event_batch['x'], event_batch['y'], c=event_batch['pol'])
    plt.xlim([0, 128])
    plt.ylim([0, 128])
    plt.title("Sensor")
    plt.show()

    # rotmat0 = generate_random_rotmat(0)
    # print(rotmat0)

    mappoints = []
    u = []
    v = []
    pol = []
    # exit()
    for idx, event in event_batch.iterrows():
        # df_angles = event_and_particles_to_angles(event, particles['Rotation'], calibration)
        df_uvp = particles_per_event2map(event, particles, calibration)[['v', 'u', 'pol']]
        u_ = int(df_uvp['u'].tolist()[0])
        u.append(u_)
        v_ = int(df_uvp['v'].tolist()[0])
        v.append(v_)
        pol_ = df_uvp['pol'].tolist()[0]
        pol.append(pol_)
        intensity = get_intensity_from_gradientmap(gradientmap=intensity_map,
                                       u=u_, v=v_)
        print(intensity)


    fig_mappoints = plt.figure(2)
    plt.scatter(u, v, c=pol)
    plt.title("Mappoints")
    # plt.xlim([0, 2048])
    # plt.ylim([0, 1024])
    plt.show()
'''


# def get_pixelmap_for_particles(event, sensortensor, particles_all_time):
#     """
#     Working on...
#     :param event:
#     :param sensortensor:
#     :param particles_all_time:
#     :return:
#     """
#     t = event['t']
#     x = event['x']
#     y = event['y']
#     ttc = sensortensor[1][x,y][0]
#
#     particles = get_latest_particles(t_asked=t, particles_all_time=particles_all_time)
#
#     particles_ttc = get_latest_particles(t_asked=ttc, particles_all_time=particles_all_time)
#
#     particles_per_event2map(event, particles, camera_intrinsicsK)
#
#     pass

# def update_sensortensor_from_batch(sensortensor, batch_event):
#     """
#     Updates sensortensor for each event. Saves event at t and t-t_c
#     :param sensortensor: tensor sensorwidth*sensorheight*2, with tuple (t,pol) as entries
#     :param event: Pandas DataFrame with ['t', 'x', 'y', 'pol']
#     :return: void
#     """
#     for event in batch_event:
#         x = int(event['x'])
#         y = int(event['y'])
#         sensortensor[1][y, x] = sensortensor[0][y, x]
#         sensortensor[0][y, x] = (event['t'], event['pol'])
#     return


'''
if __name__ == '__main__':

    events = load_events('../data/synth1/events.txt')

    # sensortensor = initialize_sensortensor(128, 128)
    # starttime = time.time()
    # i = 0
    # for idx, event in events.head(50000).iterrows():
    #     # print(event)
    #     update_sensortensor(sensortensor=sensortensor, event=event)
    #     # i += 1
    #     # if i >= 10:
    #     #     break
    # endtime = time.time() - starttime
    # print("Endtime: ", endtime)
    # print(sensortensor[0][0,0][0])
    # exit()

    ## Testing sensortensor
    # event = events.loc[557]
    # print(event)
    # sensortensor_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # sensortensor_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # sensortensor_ttc = np.array([sensortensor_t, sensortensor_tc])
    # print(sensortensor_ttc)
    # update_sensortensor(sensortensor_ttc, event)
    # print(sensortensor_ttc)
    #
    # ## Testing sensortensor
    # # event = events.loc[557]
    # # print(event)
    # # sensortensor_t = np.zeros((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # sensortensor_tc = np.ones((28, 28), dtype=[('time', 'f8'), ('polarity', 'i4')])
    # # sensortensor_ttc = np.array([sensortensor_t, sensortensor_tc])
    # # print(sensortensor_ttc)
    # # update_sensortensor(sensortensor_ttc, event)
    # # print(sensortensor_ttc)
    # #
    # # # print(sensortensor[1])
    # # print(sensortensor_ttc[1][0,0])
    # #
    # #
    #
    #
    #
    # # state update step

    # camera_intrinsicsK = camera_intrinsics()
    particles= init_particles(num_particles)
    # # print(particles)
    # particles_per_event = event_and_particles_to_angles(events.loc[0], particles['Rotation'], camera_intrinsicsK)
    # # particles_per_event['v'], particles_per_event['u'] = zip(*particles_per_event.apply(
    # #     lambda row: angles2map(row['theta'], row['phi']), axis=1))
    # particles_per_event = particles_per_event2map(events.loc[0], particles, camera_intrinsicsK)
    # # print(particles_per_event)
    updated_particles = motion_update(particles)
    # print(updated_particles)

    test_distributions_rotmat(updated_particles['Rotation'])
    # test_distributions_rotmat(updated_particles['Rotation'])
    # plt.figure(1)
    # plt.scatter(particles_per_event['theta'], particles_per_event['phi'])
    # plt.show()
    # plt.figure(2)
    # plt.scatter(particles_per_event['u'], particles_per_event['v'])
    # plt.show()
'''
