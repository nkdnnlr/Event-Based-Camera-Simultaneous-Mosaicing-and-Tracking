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



data_dir = '../data/synth1'
intensity_map = np.load('../output/intensity_map.npy')
event_file = os.path.join(data_dir, 'events.txt')
filename_poses = os.path.join(data_dir, 'poses.txt')
outputdir_poses = '../output/poses/'


# Constants
eventlikelihood_comparison_flipped = False
num_particles = 1000
num_events_batch = 300
sigma_init1 = 0
sigma_init2 = 0
sigma_init3 = 0
factor = 1
sigma_likelihood = 8.0*1e-2
sigma_1 = factor * 0.0004# sigma1 for motion update
sigma_2 = factor * 0.0004# sigma2 for motion update
sigma_3 = factor * -0.0005287901912270614 # sigma3 for motion update
total_nr_events_considered = int(3564657/360*40)  #TODO: Only works if not dividable by events by batch
first_matrix = helpers.get_first_matrix(filename_poses)

all_rotations_test = []


# tau=7000
# tau_c=2000                                      #time between events in same pixel
# mu = 0.22
mu = 0.45
# sigma_3 = 8.0*10**(-2)
minimum_constant = 1e-3
sensor_height = 128
sensor_width = 128
image_height = 1024
image_width = 2*image_height
randomseed = None


class Tracker():
    def __init__(self):
        self.calibration = self.camera_intrinsics()
        pass

    def camera_intrinsics(self):
        """
        Define camera intrinsic matrix and return
        :return: Camera intrinsic Matrix K
        """

        # f_x = 115.534  # x-focal length
        # s = 0  # Skewness
        # x_0 = 79.262
        # f_y = 115.565  # y-focal length
        # y_0 = 65.531
        # K = np.array([[f_x, s, x_0], [0, f_y, y_0], [0, 0, 1]])

        #from Guillermo:
        K = np.array([[91.4014729896821, 0.0, 64.0],
                      [0.0, 91.4014729896821, 64.0],
                      [0, 0, 1]])
        return K


    def init_particles(self, N, init_rotmat, bound1, bound2, bound3, seed=None):
        """
        Initialize all particles based on various parameters
        :param N: # particles num_particles
        :param init_rotmat: Rotation Matrix of initial pose
        :param bound1:
        :param bound2:
        :param bound3:
        :param seed:
        :return: DataFrame with ['Rotation', 'Weight', 'theta', 'phi', 'v', 'u', 'pol', 'r_w1', 'r_w2', 'r_w3', 'z', 'pol']
        """
        df = pd.DataFrame(columns=
                          ['Rotation', 'Weight', 'theta',
                           'phi', 'v', 'u', 'pol',
                           'p_w1', 'p_w2', 'p_w3',
                           'z', 'logintensity_ttc',
                           'logintensity_t']
                          )
        df['Rotation'] = df['Rotation'].astype(object)
        w0 = 1/N
        G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        for i in range(N):
            n1 = np.random.uniform(-bound1, bound1)
            n2 = np.random.uniform(-bound2, bound2)
            n3 = np.random.uniform(-bound3, bound3)
            df.at[i, ['Rotation']] = [np.dot(init_rotmat, sp.expm(np.dot(n1, G1) +
                                                                  np.dot(n2, G2) +
                                                                  np.dot(n3, G3))
                                            )
                                     ]
            df.at[i, ['Weight']] = float(w0)
        return df


    def initialize_sensortensor(self, sensor_height=128, sensor_width=128):
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


    def update_sensortensor(self, sensortensor, event):
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


    def event_and_particles_to_angles(self, event, particles, calibration_inv):
        """
        For a given event, generates dataframe
        with particles as rows and angles as columns.
        :param event: Event in camera frame
        :param df_rotationmatrices: DataFrame with rotation matrices
        :param calibration_inv: inverted camera calibration
        :return: DataFrame with particles as rows and angles as columns.
        """
        k_inv_times_event = np.dot(calibration_inv,
                                   np.array([[event['x']], [event['y']], [1]])
                                   )  # from camera frame (u,v) to world reference frame

        coordinates = ['p_w1', 'p_w2', 'p_w3']
        particles[coordinates] = pd.DataFrame.from_records(particles['Rotation'].apply(lambda x: np.dot(x, k_inv_times_event)))

        particles['p_w1'] = particles['p_w1'].str.get(0)  # This is tested and correct. str.get(0) just removes brackets
        particles['p_w2'] = particles['p_w2'].str.get(0)
        particles['p_w3'] = particles['p_w3'].str.get(0)

        # from world reference frame to rotational frame (theta, phi)
        particles['theta'] = np.arctan2(particles['p_w1'], particles['p_w3'])
        particles['phi'] = np.arctan2(particles['p_w2'], np.sqrt(particles['p_w1'] ** 2 + particles['p_w3'] ** 2))

        return

    def event_and_oneparticle_to_angles(self, event, particle, calibration_inv):
        """
        For a given event, generates dataframe
        with particles as rows and angles as columns.
        :param event: Event in camera frame
        :param df_rotationmatrices: DataFrame with rotation matrices
        :param calibration: camera calibration
        :return: DataFrame with particles as rows and angles as columns.
        """
        k_inv_times_event = np.dot(calibration_inv, np.array([[event['x']], [event['y']], [1]])) #from camera frame (u,v) to world reference frame
        r_w1, r_w2, r_w3 = np.dot(particle['Rotation'], k_inv_times_event)
        r_w1 = r_w1[0]
        r_w2 = r_w2[0]
        r_w3 = r_w3[0]

        theta = np.arctan2(r_w1, r_w3)
        phi = np.arctan2(r_w2, np.sqrt(r_w1 ** 2 + r_w3 ** 2))

        return theta, phi


    def angles2map(self, theta, phi, height=1024, width=2048):
        """
        Converts angles (theta in [-pi, pi], phi in [-pi/2, pi/2])
        to integer map points (pixel coordinates)
        :param theta:
        :param phi:
        :param height: height of image in pixels
        :param width: width of image in pixels
        :return: tuple with integer map points (pixel coordinates)
        """
        # v = np.floor((-1*phi+np.pi/2)/np.pi*height)
        v = np.floor((np.pi / 2 - phi) / np.pi * height) # jb's version
        u = np.floor((theta + np.pi)/(2*np.pi)*width)
        return v, u


    def angles2map_df(self, particles, height=1024, width=2048):
        """
        USED FOR COLLECTION OF PARTICLES
        For DataFrame particles, converts angles (theta in [-pi, pi], phi in [-pi/2, pi/2])
        to integer map points (pixel coordinates)
        :param particles: DataFrame
        :param height: height of image in pixels
        :param width: width of image in pixels
        :return: particles
        """
        particles['v'] = particles['phi'].apply(lambda angle: np.floor((np.pi/2 - angle) / np.pi * height))
        particles['u'] = particles['theta'].apply(lambda angle: np.floor((angle + np.pi) / (2 * np.pi) * width))
        return


    def particles_per_event2map(self, event, particles, calibration_inv):
        """
        For each event, gets map angles and coordinates (for on panoramic image)
        :param event: one event
        :param particles: dataframe with particles
        :param calibration:
        :return:  DataFrame with particles as rows and as columns theta, phi, v, u (coordinates)
        """
        self.event_and_particles_to_angles(event, particles, calibration_inv)
        self.angles2map_df(particles)
        particles['pol'] = event['pol']
        return


    def oneparticle_per_event2map(self, event, particle, calibration_inv):
        """
        TODO: Takes a bit long...
        For each event, gets map angles and coordinates (for on panoramic image)
        :param event: one event
        :param particle: dataframe with particle
        :param calibration_inv:
        :return:  DataFrame with particles as rows and as columns theta, phi, v, u (coordinates)
        """
        theta, phi = self.event_and_oneparticle_to_angles(event, particle, calibration_inv)
        v, u = self.angles2map(theta, phi)
        particle['v'] = v
        particle['u'] = u
        particle['pol'] = event['pol']
        return


    def motion_update(self, particles, velocity=1.):
        """
        Randomly (normal) perturbs particles.
        :param particles: DataFrame with particles
        :param velocity: timestep
        :return: DataFrame with updated particles
        """
        G1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # rotation around x
        G2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])  # rotation around y
        G3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # rotation around z

        particles['Rotation'] = particles['Rotation'].apply(lambda x: np.dot(x, sp.expm(np.random.normal(0.0, abs(velocity*sigma_1)) * G3 +
                                                                                        np.random.normal(0.0, abs(velocity*sigma_2)) * G1 +
                                                                                        np.random.normal(sigma_3, abs(velocity*sigma_3)) * G2)))
        np.random.normal()

        return particles


    def get_latest_particles(self, t_asked, particles_all_time):
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


    def event_likelihood(self, z, event, mu=0.45, sigma=sigma_likelihood, k_e=1.0*1e-3):
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
        #TODO: Test if == or != works better. -> Seems as != looks better, see slack!
        if eventlikelihood_comparison_flipped:
            if np.sign(z) != np.sign(event['pol']):
                return k_e + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(np.abs(z) - mu) ** 2 / (2 * sigma) ** 2)
            else:
                return k_e
        else:
            if np.sign(z) == np.sign(event['pol']):
                return k_e + 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(np.abs(z) - mu) ** 2 / (2 * sigma) ** 2)
            else:
                return k_e


    def measurement_update(self,
                           events_batch,
                           particles,
                           all_rotations,
                           sensortensor,
                           calibration_inv):
        """

        :param events_batch: events of a batch
        :param particles: particles
        :param all_rotations: DataFrame containing one time and one rotation per batch.
        :param sensortensor:
        :param calibration_inv:
        :return: particles
        """
        particles['Weight'] = np.empty((len(particles), 0)).tolist()

        for idx, event in events_batch.iterrows():
            self.update_sensortensor(sensortensor, event)
            self.particles_per_event2map(event, particles, calibration_inv)
            tminustc = sensortensor[1][int(event['y']), int(event['x'])][0]
            particle_ttc = self.get_latest_particles(tminustc,
                                                particles_all_time=all_rotations)  # single rotationmatrix before ttc
            self.oneparticle_per_event2map(event,
                                      particle_ttc,
                                      calibration_inv)

            particles['logintensity_ttc'] = intensity_map[int(particle_ttc['v']-1),
                                                          int(particle_ttc['u']-1)]
            particles['logintensity_t'] = particles.apply(lambda row: intensity_map[int(row.v-1), int(row.u-1)], axis=1)
            particles['z'] = particles['logintensity_t'] - particles['logintensity_ttc']
            particles['Weight'] = particles.apply(lambda x: x.Weight + [self.event_likelihood(x.z, event)], axis=1)
        particles['Weight'] = particles['Weight'].apply(lambda x: np.mean(x)) #Tested

        return


    def normalize_particle_weights(self, particles):
        '''
        normalizes particle weights
        :param particles: particles
        :return: particles with normalized weight (sum of weights = 1)
        '''
        particles['Weight'] = particles['Weight']/particles['Weight'].sum()
        return


    def resampling(self, particles):
        #TODO: Check if it really does what it should. Looks really scary with the if-conditions.
        '''
        resamples particles
        :param particles:
        :return: resampled particles, weighted average
        '''

        # sum_of_weights=particles['Weight'].cumsum(axis=0)

        resampled_particles = pd.DataFrame(columns=['Rotation', 'Weight'])

        resampled_particles['Rotation'] = particles['Rotation'].sample(n=len(particles), replace=True,
                                                                  weights=particles['Weight'], random_state=1)
        resampled_particles['Weight'] = float(1 / len(particles))
        resampled_particles = resampled_particles.reset_index(drop=True)

        # for i in range(len(particles)):     # i: resampling for each particle
        #     r = np.random.uniform(0, 1)
        #     for n in range(len(particles)):
        #         if sum_of_weights[n] >= r and n==0:
        #             n_tilde=n
        #             break
        #         if sum_of_weights[n] >= r and r > sum_of_weights[n - 1]:
        #             n_tilde=n
        #             break
        #
        #     resampled_particles.at[i, ['Rotation']] = [particles.loc[n_tilde, 'Rotation']]
        #     resampled_particles.at[i, ['Weight']] = float(1/len(particles))
        #     resampled_particles['Weight'] = resampled_particles['Weight'].astype('float64')

        return resampled_particles

    def mean_of_resampled_particles(self, particles):
        '''
        :param particles: pandas df of resampled particles (all with the same weight)
        :return: mean of rotation matrix
        '''
        rotmats=np.zeros((len(particles), 3, 3))
        for i in range(len(particles)):
            rotmats[i] = sp.logm(particles['Rotation'].as_matrix()[i])
        liemean = sum(rotmats)/len(particles)
        mean = sp.expm(liemean)

        return mean



    def run(self):
        """
        Runs the experiment, saves and plots output.
        :return:
        """
        print("Events per batch: ", num_events_batch)
        print("Initialized particles: ", num_particles)
        calibration = self.camera_intrinsics()
        calibration_inv = np.linalg.inv(calibration)
        events, num_events = helpers.load_events(event_file,
                                                 head=total_nr_events_considered,
                                                 return_number=True)
        events = events.astype({'x': int, 'y': int})
        print(events.head(5))
        print("Events total: ", num_events)
        num_batches = int(np.floor(num_events/num_events_batch))
        print("Batches total: ", num_batches)

        dt_mean = events['t'].diff().mean()
        particles = self.init_particles(num_particles, first_matrix,
                                   sigma_init1, sigma_init2, sigma_init3,
                                   seed=None)

        sensortensor = self.initialize_sensortensor(128, 128)

        batch_nr = 0
        event_nr = 0
        t_batch = 0
        all_rotations = pd.DataFrame(columns=['t', 'Rotation'])
        all_rotations.loc[batch_nr] = {'t': t_batch,
                                       'Rotation': first_matrix}

        print("Start tracker!")
        starttime = time.time()

        mean_of_rotations = pd.DataFrame(columns=['Rotation'])
        mean_of_rotations['Rotation'].astype(object)

        all_rotations_test.append(first_matrix)
        starttime = time.time()
        while batch_nr < num_batches:
            events_batch = events[event_nr:event_nr + num_events_batch]
            dt_batch = (events_batch['t'].max() - events_batch['t'].min())/num_events_batch
            velocity = dt_mean / dt_batch

            particles = self.motion_update(particles, velocity=velocity)

            self.measurement_update(events_batch,
                               particles, all_rotations,
                               sensortensor,
                               calibration_inv
                               )
            self.normalize_particle_weights(particles)
            particles = self.resampling(particles)

            event_nr += num_events_batch
            batch_nr += 1
            t_batch = events.loc[event_nr]['t']

            new_rotation = self.mean_of_resampled_particles(particles)
            all_rotations_test.append(new_rotation)


            # visualize_particles(particles['Rotation'],  mean_value=new_rotation)

            all_rotations.loc[batch_nr] = {'t': t_batch,
                                           'Rotation': new_rotation}
            # print("time: ", t_batch, "Rotations: ", helpers.rotmat2quaternion(new_rotation))
            dtime = time.time()-starttime
            print("batch: {}/{}\t time: {}s/{}s".format(batch_nr, num_batches, int(dtime), int(dtime/batch_nr*num_batches)))

            mean_of_rotations.loc[batch_nr] = [new_rotation]


        print(batch_nr)
        print(event_nr)
        quaternions = helpers.rot2quaternions(all_rotations)
        datestring = helpers.quaternions2file(quaternions, directory='../output/poses/')

        #Include all wished
        time_passed = round(time.time() - starttime)
        helpers.write_logfile(datestring, directory= '../output/poses/',
                              experiment='Finding optimal parameters',
                              eventlikelihood_comparison_flipped=eventlikelihood_comparison_flipped,
                              num_particles=num_particles,
                              num_events=total_nr_events_considered,
                              num_events_per_batch=num_events_batch,
                              sigma1=sigma_1,
                              sigma2=sigma_2,
                              sigma3=sigma_3,
                              sigma_init1=sigma_init1,
                              sigma_init2=sigma_init2,
                              sigma_init3=sigma_init3,
                              seconds_passed=time_passed)


        print("Time passed: {} sec".format(time_passed))
        # visualisation.visualize_particles(mean_of_rotations['Rotation'], mean_value = None)
        print("Done")
        return all_rotations


if __name__ == '__main__':
    tracker = Tracker()
    tracker.run()
    # plot_unitsphere()
    # plot_unitsphere_matplot()


#########TESTING

    # print("Events per batch: ", num_events_batch)
    # print("Initialized particles: ", num_particles)
    # calibration = camera_intrinsics()
    # events, num_events = helpers.load_events(event_file, head=1, return_number=True)
    # events = events.astype({'x': int, 'y': int})
    # print(events.head()['x'])
    # print("Events total: ", num_events)
    # num_batches = int(np.floor(num_events/num_events_batch))
    # print("Batches total: ", num_batches)
    # particles = init_particles(num_particles)
    # sensortensor = initialize_sensortensor(128, 128)
    # # print(particles)
    #
    # batch_nr = 0
    # event_nr = 0
    # t_batch = 0
    # all_rotations = pd.DataFrame(columns=['t', 'Rotation'])
    # unit_matrix = np.array([[1,0,0], [0,1,0], [0,0,1]])
    # all_rotations.loc[batch_nr] = {'t': t_batch,
    #                                'Rotation': unit_matrix}
    #
    #
    # particles = init_particles(num_particles)
    # particles = measurement_update(events, particles, all_rotations, sensortensor, calibration)
    #
    # particles = normalize_particle_weights(particles)
    # particles = resampling(particles)
    #
    # resampled_particles = pd.DataFrame(columns=['Rotation', 'Weight'])
    # resampled_particles['Rotation'] = particles['Rotation'].sample(n=num_particles, replace=True,
    #                                          weights=particles['Weight'], random_state=1)
    #
    # resampled_particles['Weight'] = float(1/num_particles)
    # resampled_particles['Weight'] = resampled_particles['Weight'].astype(object)
    # resampled_particles = resampled_particles.reset_index(drop=True)
    # print(resampled_particles)
