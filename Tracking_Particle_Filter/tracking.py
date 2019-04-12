import numpy as np
import pandas as pd
import scipy.linalg as sp
import math

firstevents=np.array([[0, 1249173, 108, 112, 1],
                     [0, 1259493, 109, 109, 1]])
N=5

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

def event_to_3d(x, t, u, v, p):
    '''
    in: event in camera frame (u,v)
    out: event in rotational frame (theta,phi)
    '''   
    p_w=np.dot(np.dot(R_wc,np.linalg.inv(camera_intrinsics())),np.array([[u],[v],[1]]))             #from camera frame (u,v) to world reference frame
    p_m=np.array([np.arctan(np.divide(p_w[0],p_w[2])),
                  np.arctan( np.divide(p_w[1], np.sqrt( np.square(p_w[1])+np.square(p_w[2]) ) ) )
                  ])                                                                                #from world reference frame to rotational frame (theta, phi)
    return p_m

def angle2map(theta, phi, height=1024, width=2048):
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

#initialize N particles

def init_particles(N):
    '''
    in: # particles N
    out: location of N particles in rotational frame
    '''
    # p0 = np.eye(3)      #initial rotation matrix of particles
    p = []
    w = []
    w0=1/N
    for i in range(N):
        p.append(generate_random())

        w.append(w0)
    return(p,w)

#state update step
[p,w]=init_particles(N)

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
    for i in range(N):
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

