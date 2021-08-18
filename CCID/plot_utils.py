import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import numpy as np


def show_image(image):
    """Displays a bw image in matplotlib, without margin."""
    dpi = matplotlib.rcParams['figure.dpi']
    figsize = image.shape[0] / float(dpi), image.shape[1] / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, cmap='gray')
    plt.show()


def show_image_matrix(images, titles=None, suptitle=None):
    """Displays a matrix of images in matplotlib."""
    rows = len(images)
    columns = len(images[0])

    fig = plt.figure(figsize=(columns + 1, rows + 1))  # Avoid large blank margins
    fig.set_dpi(images[0][0].shape[0])  # Preserve original image size

    if suptitle:
        fig.suptitle(suptitle)

    for i, row in enumerate(images):
        for j, image in enumerate(row):
            ax = fig.add_subplot(rows, columns, i * columns + j + 1)
            ax.axis('off')
            if image is None:
                image = np.ones(images[0][0].shape)  # Placeholder
            ax.imshow(image, cmap='gray', interpolation='none', vmin=0, vmax=1)
            if titles:
                ax.set_title(titles[i][j], fontsize=6)

    plt.margins(0, 0)
    plt.show()


def upscale_image_sample(image, level):
    scale = 2 ** level
    factor = image.shape[0] // scale
    mid = image.shape[0] // 2
    start = mid - factor // 2
    end = start + factor
    sliced = image[start:end, start:end]
    upscaled = sliced.repeat(scale, axis=0).repeat(scale, axis=1)
    return upscaled


def psd_plot(noisy_images, denoised_images, ground_truths, reliable_images, fused_images):
    # Compute the PSD of a single image:
    noisy_image_dcts = []
    denoised_image_dcts = []
    ground_truth_dcts = []
    reliable_image_dcts = []
    factor_of_fused_images = int(len(fused_images) / len(noisy_images))
    fused_image_dcts = [[] for i in range(factor_of_fused_images)]
    for i in range(len(noisy_images)):
        noisy_image_dcts.append(
            np.abs(dct(dct(noisy_images[i], n=256, axis=0, norm='ortho'), n=256, axis=1, norm='ortho')))
        denoised_image_dcts.append(
            np.abs(dct(dct(denoised_images[i], n=256, axis=0, norm='ortho'), n=256, axis=1, norm='ortho')))
        ground_truth_dcts.append(
            np.abs(dct(dct(ground_truths[i], n=256, axis=0, norm='ortho'), n=256, axis=1, norm='ortho')))
        reliable_image_dcts.append(
            np.abs(dct(dct(reliable_images[i], n=256, axis=0, norm='ortho'), n=256, axis=1, norm='ortho')))
        for j in range(factor_of_fused_images):
            fused_image_dcts[j].append(np.abs(
                dct(dct(fused_images[i * factor_of_fused_images + j], n=256, axis=0, norm='ortho'), n=256, axis=1,
                    norm='ortho')))
    noisy_image_dct = np.mean(noisy_image_dcts, axis=0)
    denoised_image_dct = np.mean(denoised_image_dcts, axis=0)
    ground_truth_dct = np.mean(ground_truth_dcts, axis=0)
    reliable_image_dct = np.mean(reliable_image_dcts, axis=0)
    fused_image_dct = []
    for j in range(factor_of_fused_images):
        fused_image_dct.append(np.mean(fused_image_dcts[j], axis=0))

    # Plot the image PSD:
    xaxs = np.linspace(0, 0.5, 256)
    ax = plt.figure().add_subplot(111)
    for j in range(factor_of_fused_images):
        label = 'fused image %.02f' % (j / (factor_of_fused_images - 1))
        ax.plot(xaxs, np.mean(fused_image_dct[j][:256, :256], 0), color=str(j / factor_of_fused_images), label=label)
    ax.plot(xaxs, np.mean(noisy_image_dct[:256, :256], 0), color='g', label='noisy image PSD')
    ax.plot(xaxs, np.mean(reliable_image_dct[:256, :256], 0), color='b', label='reliable image PSD')
    ax.plot(xaxs, np.mean(denoised_image_dct[:256, :256], 0), color='r', label='denoised image PSD')
    ax.plot(xaxs, np.mean(ground_truth_dct[:256, :256], 0), color='y', label='ground truth PSD')

    ax.set_yscale('log')
    ax.grid()
    plt.ylabel('Power spectral density')
    plt.xlabel('Spatial image frequencies (fraction of sampling rate)')
    plt.legend()
    plt.title("")  # Add a title here
    plt.show()
