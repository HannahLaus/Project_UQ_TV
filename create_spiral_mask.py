import numpy as np
import matplotlib.pyplot as plt


def spiral_cart_w_edges_parameter_lut(undersampling_factor):
    # Load manually optimized parameters for spiral trajectory design.
    # Number of full 360° turns is kept constant. Undersampling_factor is
    # mostly determined by the number of spokes. Fine-tuning by pixel-wise
    # correction of the maximum radius in k-space.

    n_turns = 0.75
    k_space_corr_px = -180
    use_golden_angle = True

    if undersampling_factor == 2:
        n_spokes = 146
    elif undersampling_factor == 3:
        n_spokes = 94
    elif undersampling_factor == 4:
        n_spokes = 70
    elif undersampling_factor == 8:
        n_spokes = 34
    elif undersampling_factor == 16:
        n_spokes = 17
    else:
        raise ValueError('Unknown undersampling factor for spiral trajectory')

    return n_turns, n_spokes, k_space_corr_px, use_golden_angle



def create_spiral_kspace_trajectory(matrix, n_turns, n_spokes, k_space_corr_px, use_golden_angle):
    npoints = matrix ** 2
    npoints = int(round(npoints / 10 ** np.floor(np.log10(npoints))) * 10 ** np.floor(np.log10(npoints)))
    scaling = 5 * matrix / 128 * 2 / n_turns

    if use_golden_angle:
        theta_offsets = 0.5 + 137.507 * 2 * np.pi / 360 * np.arange(1, n_spokes + 1)
    else:
        theta_offsets = 0.5 + np.linspace(0, 2 * np.pi, n_spokes + 1)[:-1]

    theta = np.linspace(0, n_turns * 2 * np.pi, npoints)

    r = scaling * theta
    G = np.zeros((2, len(r), n_spokes))

    for j_spoke in range(n_spokes):
        G[:, :, j_spoke] = [r * np.cos(theta_offsets[j_spoke] + theta),
                            r * np.sin(theta_offsets[j_spoke] + theta)]

    k = np.cumsum(G, axis=1)

    k = k / np.max(np.abs(k)) * (matrix - k_space_corr_px) / 2
    k = k + matrix / 2 + 1
    k[0, :, :] = k[0, :, :] - 2

    return k


#Choose undersampling factors and image size
undersampling_factors = [2,3]  # [2, 3, 4, 8, 16]
save_mask = True
matrix = 256


for j_undersampling_factor in range(len(undersampling_factors)):
    undersampling_factor = undersampling_factors[j_undersampling_factor]


    n_turns, n_spokes, k_space_corr_px, use_golden_angle = spiral_cart_w_edges_parameter_lut(undersampling_factor)

    k = create_spiral_kspace_trajectory(matrix, n_turns, n_spokes, k_space_corr_px, use_golden_angle)

    # Regridding to Cartesian and transform to logical mask
    k_space_mask = np.zeros((matrix,matrix))
    for jn in range(k.shape[1]):
        for j_spoke in range(n_spokes):
            if (k_space_mask.shape[0] - round(k[0, jn, j_spoke]) <= k_space_mask.shape[0]) and \
                    (round(k[1, jn, j_spoke]) <= k_space_mask.shape[1]) and \
                    (k_space_mask.shape[0] - round(k[0, jn, j_spoke]) > 0) and \
                    (round(k[1, jn, j_spoke]) > 0):
                k_space_mask[-(round(k[0, jn, j_spoke])-1), round(k[1, jn, j_spoke])-1] = 1

    # Actual undersampling factor plus/minus 0.05
    undersampling_factor = round((matrix**2) / np.sum(k_space_mask) * 20) / 20

    #Crop Mask to correct size
    k_space_mask = k_space_mask[50:206,50:206]

    # Visualization of mask
    plt.figure()
    plt.title('k-space mask')
    plt.imshow(k_space_mask, cmap='gray')
    plt.axis('equal')
    plt.xlim([1, 156])
    plt.ylim([1, 156])


    # Save mask
    if save_mask:
            np.save(f'k_space_mask_spiral_undersampling_factor_{undersampling_factor}.npy', k_space_mask)


plt.show()
