#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:44:59 2019

@author: celine
"""
import math

import pandas as pd
import numpy as np
import random


### Load events
df = pd.read_csv('events.txt', delimiter=' ')
events = np.array(df)



### Camera intrinsics for later

def camera_intrinsics(u, v, z=1):
    """

    :param u:
    :param v:
    :param z:
    :return:
    """
    
    fu = 115.534
    fv = 115.565
    u0 = 79.262
    v0 = 65.531
    
    p_c = np.array([u,v,z])

    K_invers = np.matrix([[1/fu, 0, -(u0/fu)],
                          [0, 1/fv, -(v0/fv)],
                          [0,0,1]])
    
    return K_invers.dot(p_c)


for i in events: 
    
    # print(Camera_Intrinsics(i[2], i[3]))
    pass
    

#### Particle filter

## Initialize particles
    
def initialize_particles(N=500):

    """

    :param N: nr of particles
    :return: list with N particles, all initialized with a unit matrix
    """
    R0 = np.matrix([[1, 0, 0],[0, 1,0],[0,0,1]])
    p = []
    w = []
    w0=1/N
    for i in range(N):
        p.append(R0)
        w.append(w0)

    return(p, w)

## Motion update of particles

def motion_update(p):

    """

    :param p: single particle
    :return: perturbed particles, gaussian movement in all directions
    """

    G1 = np.matrix([[0, 0, 0],[0, 0,-1],[0,1,0]])
    G2 = np.matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    G3 = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    tau = 137 ##calculated the mean between all differences of the particles assuming they are in order. has to be checked
    n1 = random.gauss(0.0, tau)
    n2 = random.gauss(0.0, tau)
    n3 = random.gauss(0.0, tau)

    R_i_t = p.dot(np.exp(n1*G1 + n2*G2 * n3*G3))

    return(R_i_t)

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














###############################

r1, w1 = initialize_particles()
r2 =[]
for i in range(len(p1)):
    r2.append(motion_update(r1[i]))

w2 = []



