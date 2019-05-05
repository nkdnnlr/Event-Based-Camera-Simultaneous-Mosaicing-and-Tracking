import time
import sys
import math

import os
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
import sample.helpers as helpers
import sample.visualisation as visualisation


from numpy import outer
import math

data_dir = '../data/synth1'
intensity_map = np.load('../output/intensity_map.npy')
event_file = os.path.join(data_dir, 'events.txt')
filename_poses = os.path.join(data_dir, 'poses.txt')


# Constants
num_particles = 200
num_events_batch = 300
sigma_init1=0.01
sigma_init2=0.01
sigma_init3=0.01
sigma_1 = 3.3663987633184266e-05*100000000# sigma3 for motion update
sigma_2 = 3.366410184326084e-05*100000000# sigma3 for motion update
sigma_3 = 0.0005285784750737629*100000000 # sigma3 for motion update
total_nr_events_considered = 600001  #TODO: Only works if not dividable by events by batch
first_matrix = helpers.get_first_matrix(filename_poses)


# tau=7000
# tau_c=2000                                      #time between events in same pixel
mu = 0.22
# sigma_3 = 8.0*10**(-2)
minimum_constant = 1e-3
sensor_height = 128
sensor_width = 128
image_height = 1024
image_width = 2*image_height
randomseed = None

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

    # K = np.array([[f_x, s, x_0], [0, f_y, y_0], [0, 0, 1]])
    K = np.array([[91.4014729896821, 0.0, 64.0], [0.0, 91.4014729896821, 64.0], [0, 0, 1]]) #as per guillermo's suggestion

    return K

def event_and_particles_to_angles(event, particles, calibration):
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
    # df_coordinates = pd.DataFrame()
    # df_coordinates[coordinates] = pd.DataFrame.from_records(df_rotationmatrices.apply(lambda x: np.dot(x, event_times_K)))
    particles[coordinates] = pd.DataFrame.from_records(particles['Rotation'].apply(lambda x: np.dot(x, event_times_K)))
    # df_coordinates = pd.DataFrame.from_records(df_rotationmatrices.apply(lambda x: np.dot(x, event_times_K)),
    #                                            columns=coordinates)

    particles['r_w1'] = particles['r_w1'].str.get(0)  # ATTENTION: This is tested and correct. str.get(0) just removes brackets. See output above.
    particles['r_w2'] = particles['r_w2'].str.get(0)
    particles['r_w3'] = particles['r_w3'].str.get(0)

    # from world reference frame to rotational frame (theta, phi)
    # df_angles = pd.DataFrame(columns=['theta', 'phi'])
    # df_angles['theta'] = np.arctan(df_coordinates['r_w1'] / df_coordinates['r_w3'])
    # df_angles['phi'] = np.arctan(df_coordinates['r_w2'] / np.sqrt(df_coordinates['r_w1']**2 + df_coordinates['r_w3']**2))

    particles['theta'] = np.arctan(particles['r_w1'] / particles['r_w3'])
    particles['phi'] = np.arctan(particles['r_w2'] / np.sqrt(particles['r_w1'] ** 2 + particles['r_w3'] ** 2))

    return particles
    # return df_angles

def event_and_oneparticle_to_angles(event, particle, calibration):
    """
    For a given event, generates dataframe
    with particles as rows and angles as columns.
    :param event: Event in camera frame
    :param df_rotationmatrices: DataFrame with rotation matrices
    :param calibration: camera calibration
    :return: DataFrame with particles as rows and angles as columns.
    """
    event_times_K = np.dot(np.linalg.inv(calibration), np.array([[event['x']], [event['y']], [1]])) #from camera frame (u,v) to world reference frame
    r_w1, r_w2, r_w3 = np.dot(particle['Rotation'], event_times_K)
    r_w1 = r_w1[0]
    r_w2 = r_w2[0]
    r_w3 = r_w3[0]

    theta = np.arctan(r_w1 / r_w3)
    phi = np.arctan(r_w2 / np.sqrt(r_w1 ** 2 + r_w3 ** 2))

    return theta, phi

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
    v = -1*(np.floor((-1*phi+np.pi/2)/np.pi*height))+height
    # v = np.floor((np.pi / 2 - phi) / np.pi * height) # jb's version
    u = np.floor((theta + np.pi)/(2*np.pi)*width)
    return v, u

def angles2map_df(particles, height=1024, width=2048):
    """
    USED FOR COLLECTION OF PARTICLES
    For DataFrame particles, converts angles (theta in [-pi, pi], phi in [-pi/2, pi/2])
    to integer map points (pixel coordinates)
    :param particles: DataFrame
    :param height: height of image in pixels
    :param width: width of image in pixels
    :return: particles
    """
    particles['v'] = particles['phi'].apply(lambda angle: -1*(np.floor((-1*angle+np.pi/2)/np.pi*height))+height)
    particles['u'] = particles['theta'].apply(lambda angle: np.floor((angle + np.pi)/(2*np.pi)*width))
    return particles

def angles2map_series(particle, height=1024, width=2048):
    """
    For DataFrame particles, converts angles (theta in [-pi, pi], phi in [-pi/2, pi/2])
    to integer map points (pixel coordinates)
    :param particles: DataFrame
    :param height: height of image in pixels
    :param width: width of image in pixels
    :return: particles
    """
    particle['v'] = -1*(np.floor((-1 * particle['phi']+np.pi/2)/np.pi*height))+height
    particle['u'] = np.floor((particle['theta'] + np.pi)/(2*np.pi)*width)
    return particle

def particles_per_event2map(event, particles, calibration):
    """
    For each event, gets map angles and coordinates (for on panoramic image)
    :param event: one event
    :param particles: dataframe with particles
    :param calibration:
    :return:  DataFrame with particles as rows and as columns theta, phi, v, u (coordinates)
    """
    particles = event_and_particles_to_angles(event, particles, calibration)
    particles = angles2map_df(particles)
    particles['pol'] = event['pol']
    return particles

def oneparticle_per_event2map(event, particle, calibration):
    """
    For each event, gets map angles and coordinates (for on panoramic image)
    :param event: one event
    :param particles: dataframe with particles
    :param calibration:
    :return:  DataFrame with particles as rows and as columns theta, phi, v, u (coordinates)
    """
    theta, phi = event_and_oneparticle_to_angles(event, particle, calibration)
    v, u = angles2map(theta, phi)
    particle['v'] = v
    particle['u'] = u
    particle['pol'] = event['pol']
    return particle

def generate_random_rotmat(unit=False, seed=None):
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

def init_particles(N, init_rotmat, sigma1, sigma2, sigma3, seed=None):
    '''
    in: N = # particles num_particles
        init_rotmat:  Rotation Matrix of initial pose
    out: data frame with Index, Rotation matrix and weight
    '''
    df = pd.DataFrame(columns=['Rotation', 'Weight', 'theta', 'phi', 'v', 'u', 'pol', 'r_w1', 'r_w2', 'r_w3', 'z'])
    df['Rotation'] = df['Rotation'].astype(object)
    w0 = 1/N
    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    for i in range(N):
        n1 = np.random.uniform(-sigma1, sigma1)
        n2 = np.random.uniform(-sigma2, sigma2)
        n3 = np.random.uniform(-sigma3, sigma3)
        df.at[i,['Rotation']] = [ np.dot( init_rotmat , sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))) ]
        df.at[i, ['Weight']] = float(w0)
    return df

    # print(events)

### PARTICLE FILTER ###

# define global variables:
         #amount of particles


##initialize num_particles particles

def motion_update(particles, tau, seed=None):
    """

    :param particles: DataFrame
    :param tau: timestep #TODO: is this tau in seconds?
    :return: updated particles
    """
    if seed is not None:
        np.random.seed(seed)

    updated_particles = particles.copy()  # type: object
    G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # rotation around x
    G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])  # rotation around y
    G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # rotation around z
    # TODO: update sigma_3 and tau!!
    sigma1 = sigma_1
    sigma2 = sigma_2
    sigma3 = sigma_3
    # p_u=[]
    # for i in range(len(particles)):
        # n1 = np.random.normal(0.0, sigma1**2 * tau)
        # n2 = np.random.normal(0.0, sigma2**2 * tau)
        # n3 = np.random.normal(0.0, sigma3**2 * tau)
        # p_u.append(np.dot(particle[i], sp.expm(np.dot(n1, G1) + np.dot(n2, G2) + np.dot(n3, G3))))
    updated_particles['Rotation'] = updated_particles['Rotation'].apply(
        lambda x: np.dot(x, sp.expm(np.dot(np.random.normal(0.0, sigma3**2 * tau), G1) +
                                    np.dot(np.random.normal(0.0, sigma3**2 * tau), G2) +
                                    np.dot(np.random.normal(0.0, sigma3**2 * tau), G3))))

    #print(updated_particles['Rotation'])

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
    #TODO: Only tested for first.
    From list of particles over all times
    (one per timestep/batch, already resampled),
    get set of particles that was just before asked t_asked
    :param t_asked: t_asked of interest (e.g. t_asked of current event)
    :param particles_all_time:
    :return: the particle that came before a time-of-interest.
    """
    # dt_pos = 1e-4 #TODO: Write as class variable (also dt_pos_inv)
    # dt_pos_inv = 1. / dt_pos
    # t_particles = math.floor(t_asked * dt_pos_inv) / dt_pos
    return particles_all_time[particles_all_time['t'] <= t_asked].iloc[-1]

def get_intensity_from_gradientmap(gradientmap, u, v):
    """
    Gets
    :param gradientmap:
    :param x:
    :param y:
    :return:
    """
    return gradientmap[v, u]

def event_likelihood(z, event, mu=0.22, sigma=8.0*1e-2, k_e=1.0*1e-3):
    """
    For a given absolute log intensity difference z,
    returns the likelihood of an event.
    likelihood = gaussian distribution + noise
    TODO: What about negative values? -> Done. test!!
    TODO: If z negative but event positive, cancel! -> Done, test!!
    :param z: log intensity difference
    :param mu: mean
    :param sigma: standard deviation
    :param k_e: minimum constant / noise
    :return: event-likelihood (scalar)
    """
    if np.sign(z) == np.sign(event['pol']):
        return k_e + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(np.abs(z) - mu) ** 2 / (2 * sigma) ** 2)
    else:
        return k_e

def measurement_update(events_batch,
                       particles,
                       all_rotations,
                       sensortensor,
                       calibration):
    """

    :param events_batch: events of a batch
    :param particles: particles
    :param all_rotations: DataFrame containing one time and one rotation per batch.
    :param sensortensor:
    :param calibration:
    :return: particles
    """
    particles['Weight'] = np.empty((len(particles), 0)).tolist()
    for idx, event in events_batch.iterrows():
        update_sensortensor(sensortensor, event)
        particles = particles_per_event2map(event, particles, calibration)
        tminustc = sensortensor[1][int(event['y']), int(event['x'])][0]
        particle_ttc = get_latest_particles(tminustc,
                                            particles_all_time=all_rotations)  # single rotationmatrix before ttc
        particle_ttc = oneparticle_per_event2map(event, particle_ttc, calibration)
        u_ttc = particle_ttc['u']
        v_ttc = particle_ttc['v']
        particles['logintensity_ttc'] = intensity_map[int(v_ttc-1), int(u_ttc-1)]
        particles['logintensity_t'] = particles.apply(lambda row: intensity_map[int(row.v-1), int(row.u-1)], axis=1)
        particles['z'] = particles['logintensity_t'] - particles['logintensity_ttc']
        particles['Weight'] = particles.apply(lambda x: x.Weight + [event_likelihood(x.z, event)], axis=1)
    particles['Weight'] = particles['Weight'].apply(lambda x: np.mean(x)) #Tested
    return particles

def normalize_particle_weights(particles):
    '''
    normalizes particle weights
    :param particles: particles
    :return: particles with normalized weight (sum of weights = 1)
    '''
    particles['Weight'] = particles['Weight']/particles['Weight'].sum()
    return particles

def resampling(particles):
    #TODO: Check if it really does what it should. Looks really scary with the if-conditions.
    '''
    resamples particles
    :param particles:
    :return: resampled particles, weighted average
    '''

    sum_of_weights=particles['Weight'].cumsum(axis=0)

    resampled_particles = pd.DataFrame(columns=['Rotation', 'Weight'])
    # resampled_particles['Rotation'] = resampled_particles['Rotation'].astype(object)
    # resampled_particles['Weight'] = resampled_particles['Weight'].astype(object)
    #
    # resampled_particles['Rotation'] = particles['Rotation'].sample(n=num_particles, replace=True,
    #                                                            weights=particles['Weight'], random_state=1)
    # resampled_particles['Weight'] = float(1 / num_particles)
    # resampled_particles = resampled_particles.reset_index(drop=True)
    # #
    for i in range(len(particles)):     # i: resampling for each particle
        r = np.random.uniform(0, 1)
        for n in range(len(particles)):
            if sum_of_weights[n] >= r and n==0:
                n_tilde=n
                continue
            if sum_of_weights[n] >= r and r > sum_of_weights[n - 1]:
                n_tilde=n
                continue

        resampled_particles.at[i, ['Rotation']] = [particles.loc[n_tilde, 'Rotation']]
        resampled_particles.at[i, ['Weight']] = float(1/len(particles))
        resampled_particles['Weight'] = resampled_particles['Weight'].astype('float64')

    return resampled_particles

def mean_of_resampled_particles(particles):
    '''
    TODO: Does too much computations: gets matrix for every particle for every particle ;)
    :param particles: pandas df of resampled particles (all with the same weight)
    :return: mean of rotation matrix
    '''
    rotmats=np.zeros((len(particles),3,3))
    for i in range(len(particles)):
        rotmats[i] = sp.logm(particles['Rotation'].as_matrix()[i])
    liemean = sum(rotmats)/len(particles)
    mean = sp.expm(liemean)


    '''
    random_x = np.random.randn(400)
    random_y = np.random.randn(400)

    trace = go.Scatter(
        x=random_x,
        y=random_y,
        mode='markers'
    )
    data = [trace]
    # Plot and embed in ipython notebook!
    plot_url = py.plot(data, filename='basic-line')
    '''

    return mean



def run():
    # num_particles = 20

    # plot_unitsphere_matplot()
    print("Events per batch: ", num_events_batch)
    print("Initialized particles: ", num_particles)
    calibration = camera_intrinsics()
    events, num_events = helpers.load_events(event_file, head=total_nr_events_considered, return_number=True)
    events = events.astype({'x': int, 'y': int})
    print(events.head()['x'])
    print("Events total: ", num_events)
    num_batches = int(np.floor(num_events/num_events_batch))
    print("Batches total: ", num_batches)

    particles = init_particles(num_particles, first_matrix, sigma_init1 , sigma_init2, sigma_init3,  seed=None)

    sensortensor = initialize_sensortensor(128, 128)
    # print(particles)

    batch_nr = 0
    event_nr = 0
    t_batch = 0
    all_rotations = pd.DataFrame(columns=['t', 'Rotation'])
    unit_matrix = np.array([[1,0,0], [0,1,0], [0,0,1]])
    all_rotations.loc[batch_nr] = {'t': t_batch,
                                   'Rotation': unit_matrix}
    # print(all_rotations)
    # exit()
    print("Start tracker!")
    starttime = time.time()
    mean_of_rotations = pd.DataFrame(columns = ['Rotation'])
    mean_of_rotations['Rotation'].astype(object)
    while batch_nr < num_batches:
        events_batch = events[event_nr:event_nr + num_events_batch]
        t_batch = events.loc[event_nr]['t']
        dt_batch = (events_batch['t'].max() - events_batch['t'].min())/num_events_batch
        # print("t_batch: {} sec".format(t_batch))
        # print("dt_batch: {} sec".format(dt_batch))
        particles = measurement_update(events_batch, particles, all_rotations, sensortensor, calibration)
        particles = normalize_particle_weights(particles)
        particles = resampling(particles)

        event_nr += num_events_batch
        batch_nr += 1
        t_batch = events.loc[event_nr]['t']

        new_rotation = mean_of_resampled_particles(particles)

        # visualize_particles(particles['Rotation'],  mean_value=new_rotation)

        all_rotations.loc[batch_nr] = {'t': t_batch,
                                       'Rotation': new_rotation}
        # print("time: ", t_batch, "Rotations: ", helpers.rotmat2quaternion(new_rotation))
        print("batch: {} time: {}".format(batch_nr, t_batch))
        particles = motion_update(particles, tau=dt_batch, seed=None)

        mean_of_rotations.loc[batch_nr] = [new_rotation]


    print(batch_nr)
    print(event_nr)
    visualisation.visualize_particles(mean_of_rotations['Rotation'], mean_value = None)
    datestring = helpers.write_quaternions2file(all_rotations)
    #Include all wished
    time_passed = round(time.time() - starttime)
    helpers.write_logfile(datestring,
                          experiment='find_optimal_parameters',
                          num_particles=num_particles,
                          num_events=total_nr_events_considered,
                          num_events_per_batch=num_events_batch,
                          sigma1=sigma_1,
                          sigma2=sigma_2,
                          sigma3=sigma_3,
                          seconds_passed=time_passed)


    print("Time passed: {} sec".format(time_passed))
    print("Done")

if __name__ == '__main__':
    run()
