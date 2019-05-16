# #!/usr/bin/env python
#
# import numpy as np
# import time
# import matplotlib
# matplotlib.use('TKAgg')
# from matplotlib import pyplot as plt
#
#
# def randomwalk(dims=(256, 256), n=20, sigma=5, alpha=0.95, seed=1):
#     """ A simple random walk with memory """
#
#     r, c = dims
#     gen = np.random.RandomState(seed)
#     pos = gen.rand(2, n) * ((r,), (c,))
#     old_delta = gen.randn(2, n) * sigma
#
#     while True:
#         delta = (1. - alpha) * gen.randn(2, n) * sigma + alpha * old_delta
#         pos += delta
#         for ii in range(n):
#             if not (0. <= pos[0, ii] < r):
#                 pos[0, ii] = abs(pos[0, ii] % r)
#             if not (0. <= pos[1, ii] < c):
#                 pos[1, ii] = abs(pos[1, ii] % c)
#         old_delta = delta
#         yield pos
#
#
# def run(niter=1000, doblit=True):
#     """
#     Display the simulation using matplotlib, optionally using blit for speed
#     """
#
#     fig, ax = plt.subplots(1, 1)
#     ax.set_aspect('equal')
#     ax.set_xlim(0, 255)
#     ax.set_ylim(0, 255)
#     # ax.hold(True)
#     rw = randomwalk()
#     x, y = rw.__next__()
#
#     plt.show(False)
#     plt.draw()
#
#     if doblit:
#         # cache the background
#         background = fig.canvas.copy_from_bbox(ax.bbox)
#
#     points = ax.plot(x, y, 'o')[0]
#     tic = time.time()
#
#     for ii in range(niter):
#
#         # update the xy data
#         x, y = rw.__next__()
#         points.set_data(x, y)
#
#         if doblit:
#             # restore background
#             fig.canvas.restore_region(background)
#
#             # redraw just the points
#             ax.draw_artist(points)
#
#             # fill in the axes rectangle
#             fig.canvas.blit(ax.bbox)
#
#         else:
#             # redraw everything
#             fig.canvas.draw()
#
#     plt.close(fig)
#     print("Blit = %s, average FPS: %.2f" % (
#         str(doblit), niter / (time.time() - tic)))
#
# if __name__ == '__main__':
#     run(doblit=False)
#     run(doblit=True)

# import numpy as np
# import matplotlib.pyplot as plt
#
# A = np.array([[100,2,3], [10, 20, 30], [100, 200, 300]])
# plt.imshow(A)
# plt.show()

import numpy as np
import pandas as pd
# filename = "/home/nik/UZH/NSC/3D Vision/Project/Event-Based-Camera-Simultaneous-Mosaicing-and-Tracking/output/log_intensity_map.csv"
# intensity_map = pd.read_csv(filename, header=None)
# intensity_map = intensity_map.values
# np.save("intensity_map.npy", intensity_map)

def angle2map(theta, phi, height=1024, width=2048):
    """
    :param theta:
    :param phi:
    :param height:
    :param width:
    :return:
    """
    y = np.int(np.floor((-1*phi+np.pi/2)/np.pi*height))
    x = np.int(np.floor((theta + np.pi)/(2*np.pi)*width))
    return y, x

import pandas as pd
import Tracking_Particle_Filter.tracking as tracking


# d = {'Rotation': ['a', 'b', 'c' ,'d'], 'Weight': [0, 0,0,100]}
# particles = pd.DataFrame(data=d)
#
# tracking.resampling(particles)
#
# print(particles)
# print(tracking.resampling(particles))
#

def rot2d(theta):
    matrix = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return matrix

def abstandnorm(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

A = rot2d(np.pi/2)
B = rot2d(np.pi/4)


v = np.array([1, 0])
w = np.array([1/np.sqrt(2), 1/np.sqrt(2)])


# print(abstandnorm(v, w))
print(abstandnorm(np.dot(A, v), np.dot(B, v)))
print(abstandnorm(np.dot(A, v), np.dot(B, w)))

# for x in range(1000000):
#     y = x**2
#     print(x)

# print(A.T.dot(A))