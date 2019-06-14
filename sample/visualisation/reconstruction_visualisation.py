import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

import sample.helpers.integration_methods as integration_methods


## Import image data
images_dir = "../../output/ourdataset/test"

pickle_in = open("grad_map.pickle", "rb")
grad_map = pickle.load(pickle_in)

pickle_in = open("trace_map.pickle", "rb")
trace_map = pickle.load(pickle_in)

## Process image data
grad_map_clip = {}
grad_map_clip["x"] = grad_map["x"]
grad_map_clip["y"] = grad_map["y"]
mask = trace_map > 0.01  # % reconstruct only gradients with small covariance

grad_map_clip["x"][mask] = 0
grad_map_clip["y"][mask] = 0

rec_image = integration_methods.frankotchellappa(grad_map_clip["x"], grad_map_clip["y"])
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
h_gx = plt.imshow(
    grad_map["x"] / np.std(grad_map["x"]), cmap=plt.cm.binary, vmin=-5, vmax=5
)
plt.title("Gradient in X")
plt.savefig(os.path.join(images_dir, "gradient_x.pdf"), dpi=350)
# plt.show()

fig_gradienty = plt.figure(4)
h_gx = plt.imshow(
    grad_map["y"] / np.std(grad_map["y"]), cmap=plt.cm.binary, vmin=-5, vmax=5
)
plt.title("Gradient in Y")
plt.savefig(os.path.join(images_dir, "gradient_y.pdf"), dpi=350)
# plt.show()

fig_tracemap = plt.figure(5)
h_gx = plt.imshow(trace_map / np.max(trace_map), cmap=plt.cm.binary, vmin=0, vmax=1)
plt.title("Trace of Covariance")
plt.savefig(os.path.join(images_dir, "trace.pdf"), dpi=350)
plt.show()

np.save("intensity_map.npy", rec_image)
