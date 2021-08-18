import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def draw_zoom(img):
    """Adds a "zoomed" window at the bottom."""
    sub_size = 16
    start = (len(img) - sub_size) // 2
    end = start + sub_size
    sliced = img[start:end, start:end]
    factor = 6
    start = len(img) - factor * sub_size
    end = len(img)
    img[start:end, start:end] = sliced.repeat(factor, axis=0).repeat(factor, axis=1)
    img[start:end, (start - factor):start] = 0
    img[(start - factor):start, (start - factor):end] = 0


def resource_path(path):
    """Relative resource path."""
    directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(directory, path)

def main():
    """Plots a comparison matrix (currently BM3D and DnCNN)."""
    plt.rcParams["figure.figsize"] = (10, 6)

    n_images = 7  # Number of different images
    mode_names = ["Original", "Noisy", "BM3D", "DnCNN"]

    n_rows = len(mode_names)
    fig, axs = plt.subplots(n_rows, n_images)

    plt.suptitle("Comparison of denoisers (BM3D, DnCNN) - CCID")

    for column in range(n_images):
        name = '{:02d}'.format(column + 1)

        img_original = mpimg.imread(resource_path('../library/dataset/clean_data/Test/Set12/%s.png' % name))
        img_noisy = mpimg.imread(resource_path('../library/dataset/noisy_data/Test/Set12/%s_sigma25.png' % name))
        img_bm3d = mpimg.imread(resource_path('../library/BM3D/denoised_images/Test/Set12/%s_sigma25.png' % name))
        img_dncnn = mpimg.imread(resource_path('../library/DnCNN/TrainingCodes/dncnn_pytorch/results/Set12/%s_dncnn.png' % name))

        for i, img in enumerate([img_original, img_noisy, img_bm3d, img_dncnn]):
            ax = axs[i, column]

            draw_zoom(img)

            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(mode_names[i])
            if i == n_rows - 1:
                ax.set_xlabel('%s.png' % name)

            ax.set_aspect('equal')

            ax.imshow(img, cmap='Greys_r', interpolation='nearest')

    # plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()


if __name__ == '__main__':
    # Make sure you generate all the images before running the script.
    main()
