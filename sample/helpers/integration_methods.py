import numpy as np


def frankotchellappa(dzdx, dzdy):
    """
    % Frankt-Chellappa Algorithm
    % Input gx and gy
    % Output : reconstruction
    % Author: Amit Agrawal, 2005
    %
    % Obtained online from (July 2017):
    % http://www.cs.cmu.edu/~ILIM/projects/IM/aagrawal/software.html
    %
    % Permission to use, copy and modify this software and its documentation without fee for educational, research and non-profit purposes, is hereby granted, provided that the above copyright notice and the following three paragraphs appear in all copies.
    %
    % To request permission to incorporate this software into commercial products contact: Vice President of Marketing and Business Development; Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139
    %
    % In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
    %
    % MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. the software provided hereunder is on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.
    %

    % disp('=======================================================')
    % disp('Solving Using Frankot Chellappa Algorithm');
    :param dzdx:
    :param dzdy:
    :return: z
    """
    # print(dzdx[0,0])

    rows, cols = np.array(dzdx).shape

    wx, wy = np.meshgrid(
        np.arange(-np.pi / 2, np.pi / 2, np.pi / (cols)),
        np.arange(-np.pi / 2, np.pi / 2, np.pi / (rows)),
    )

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = np.fft.ifftshift(wx)
    wy = np.fft.ifftshift(wy)

    DZDX = np.fft.fft2(dzdx)  # Fourier transforms of gradients
    DZDY = np.fft.fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    #  weighting the Fourier coefficients by their frequencies in x and y and
    #  then dividing by the squared frequency.  eps is added to the
    #  denominator to avoid division by 0.
    eps = np.finfo(float).eps
    j = 1j

    # % dd = wx.^2 + wy.^2;
    Z = (-j * wx * DZDX - j * wy * DZDY) / (np.power(wx, 2) + np.power(wy, 2) + eps)

    z = np.real(np.fft.ifft2(Z))  # Reconstruction
    z = z - np.min(z)
    z = z / 2
    # print(z[0,0])
    # print(z[100, 100])
    # print(z[554, 100])
    # print(z.shape)
    # exit()
    return z


if __name__ == "__main__":
    import os
    import pickle

    import pandas as pd

    import matplotlib.pyplot as plt

    output_dir = "../output/ourdataset/test"

    pickle_in = open("grad_map_clip.pickle", "rb")
    grad_map_clip = pickle.load(pickle_in)

    # print(grad_map_clip['x'][:, 1750:2000].shape)
    # exit()

    rec_image = frankotchellappa(
        grad_map_clip["x"][400:700, 1750:2000], grad_map_clip["y"][400:700, 1750:2000]
    )
    rec_image = frankotchellappa(grad_map_clip["x"], grad_map_clip["y"])
    rec_image = rec_image - np.mean(rec_image)

    rec_image_normalized = rec_image / np.max(np.abs(rec_image))
    fig_normalized = plt.figure(1)
    plt.imshow(rec_image_normalized, cmap=plt.cm.binary)
    plt.title("Reconstructed image (log)")
    plt.savefig(os.path.join(output_dir, "reconstructed_log.png"))
    plt.show()
    #
    rec_image_exp = np.exp(0.001 + rec_image)
    fig_normalized_linear = plt.figure(2)
    plt.imshow(rec_image_exp, cmap=plt.cm.binary)
    plt.title("Reconstructed image (linear)")
    plt.savefig(os.path.join(output_dir, "reconstructed_linear.png"))
    plt.show()

    fig_gradientx = plt.figure(3)
    h_gx = plt.imshow(
        grad_map_clip["x"] / np.std(grad_map_clip["x"]),
        cmap=plt.cm.binary,
        vmin=-5,
        vmax=5,
    )
    plt.title("Gradient in X")
    plt.savefig(os.path.join(output_dir, "gradient_x.png"))
    plt.show()

    fig_gradienty = plt.figure(4)
    h_gx = plt.imshow(
        grad_map_clip["y"] / np.std(grad_map_clip["y"]),
        cmap=plt.cm.binary,
        vmin=-5,
        vmax=5,
    )
    plt.title("Gradient in Y")
    plt.savefig(os.path.join(output_dir, "gradient_y.png"))
    plt.show()
