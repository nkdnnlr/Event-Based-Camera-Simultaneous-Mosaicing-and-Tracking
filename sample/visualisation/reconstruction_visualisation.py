import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

import sample.helpers.integration_methods as integration_methods



images_dir = '../output/ourdataset/test'

pickle_in = open("grad_map.pickle", "rb")
grad_map = pickle.load(pickle_in)

pickle_in = open("trace_map.pickle", "rb")
trace_map = pickle.load(pickle_in)


grad_map_clip = {}
grad_map_clip['x'] = grad_map['x']
grad_map_clip['y'] = grad_map['y']
mask = trace_map > 0.01  # % reconstruct only gradients with small covariance


grad_map_clip['x'][mask] = 0
grad_map_clip['y'][mask] = 0




# grad_map_clip.to_csv('grad_map_clip.csv')
#
# grad_map_clip['x'] = grad_map_clip['x'].loc[1750:2000]
# grad_map_clip['y'] = grad_map_clip['y'].loc[400:700]






rec_image = integration_methods.frankotchellappa(grad_map_clip['x'], grad_map_clip['y']);
rec_image = rec_image - np.mean(rec_image)

rec_image_normalized = rec_image / np.max(np.abs(rec_image))
fig_normalized = plt.figure(1)
plt.imshow(rec_image_normalized, cmap=plt.cm.binary)
plt.title("Reconstructed image (log)")
plt.savefig(os.path.join(images_dir, "reconstructed_log.pdf"), dpi=350)
# plt.show()
#
rec_image_exp = np.exp(0.001 + rec_image)
fig_normalized_linear = plt.figure(2)
plt.imshow(rec_image_exp, cmap=plt.cm.binary)
plt.title("Reconstructed image (linear)")
plt.savefig(os.path.join(images_dir, "reconstructed_linear.pdf"), dpi=350)
# plt.show()

fig_gradientx = plt.figure(3)
h_gx = plt.imshow(grad_map['x'] / np.std(grad_map['x']), cmap=plt.cm.binary, vmin=-5, vmax=5)
plt.title("Gradient in X")
plt.savefig(os.path.join(images_dir, "gradient_x.pdf"), dpi=350)
# plt.show()

fig_gradienty = plt.figure(4)
h_gx = plt.imshow(grad_map['y'] / np.std(grad_map['y']), cmap=plt.cm.binary, vmin=-5, vmax=5)
plt.title("Gradient in Y")
plt.savefig(os.path.join(images_dir, "gradient_y.pdf"), dpi=350)
# plt.show()

fig_tracemap = plt.figure(5)
h_gx = plt.imshow(trace_map/np.max(trace_map), cmap=plt.cm.binary, vmin=0, vmax=1)
plt.title("Trace of Covariance")
plt.savefig(os.path.join(images_dir, "trace.pdf"), dpi=350)
plt.show()

# g_ang = -1*np.arctan2(grad_map['y'], grad_map['x'])
# g_grad = np.sqrt(np.power(grad_map['x'], 2) + np.power(grad_map['y'], 2))
# g_grad_unit = g_grad/1.
# g_grad_unit[g_grad_unit > 1.0] = 1.0
# g_ang_unit = g_ang/360. + 0.5

np.save("intensity_map.npy", rec_image)



