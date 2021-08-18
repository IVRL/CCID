import os

import cv2
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch

from CCID.confidence.confidence import load_confidence_model, predict_confidence
from CCID.fusion.reliable_sr import bilinear_interpolation_filter, upscale, \
    bicubic_interpolation_filter
from CCID.plot_utils import show_image_matrix, upscale_image_sample, psd_plot
from CCID.denoiser.reliable_filters import gaussian_blur_filter, bilateral_filter, median_filter, \
    non_local_means_filter
from CCID.fusion.fusion_methods import simple_weighted_fusion, simple_dct_fusion, dwt_fusion, \
    exponential_dct_fusion, \
    threshold_fusion, adaptive_dct_fusion
from CCID.confidence.models.ConvNet_region import *
from CCID.confidence.models.DnCNN import *

set_12 = "12"
set_68 = "68"
set_microscopy = "microscopy"
set_14 = "14"
super_resolution_choice = 4  # Super-resolution setting, if enabled (see below)

# Choices
reliable_choice = "gaussian"
fusion_choice = "weighted_dwt"
set_choice = set_12
is_super_resolution_choice = False

reliable_strategies = dict(
    gaussian=gaussian_blur_filter(kernel_size=13, sigma=2),
    median=median_filter(kernel_size=5),
    bilateral=bilateral_filter(),
    nlmeans=non_local_means_filter(),
    # Super-resolution
    bilinear=bilinear_interpolation_filter(super_resolution_choice),
    bicubic=bicubic_interpolation_filter(super_resolution_choice)
)

fusion_strategies = dict(
    weighted_spatial=simple_weighted_fusion,
    weighted_dct=simple_dct_fusion,
    weighted_dwt=dwt_fusion,
    gaussian_weighted=exponential_dct_fusion,
    thresholded_dct=threshold_fusion,
    adaptive_dct=adaptive_dct_fusion
)


def fuse_image(noisy_image, denoised_image, fusion_weight, confidence_map=None, insight=None):
    """ create reliable inputs from the noisy images """
    reliable_image = reliable_strategies[reliable_choice](noisy_image)

    fused_image = fusion_strategies[fusion_choice](reliable_image, denoised_image, fusion_weight,
                                                   confidence_map=confidence_map,
                                                   insight=insight)
    return reliable_image, fused_image


def load_models():
    global denoise_model
    global confidence_model

    directory_path = os.path.dirname(os.path.realpath(__file__))
    denoise_model = torch.load(
        os.path.join(directory_path, "../library/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth"),
        map_location=torch.device("cpu"))
    denoise_model.eval()  # Important
    confidence_model = load_confidence_model()


def load_images():
    def load_image(image_path):
        directory_path = os.path.dirname(os.path.realpath(__file__))
        absolute_path = os.path.join(directory_path, image_path)
        img = imread(absolute_path)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.array(bw, dtype=np.float32) / 255.0

    image_triplets = []
    if is_super_resolution_choice:
        path = "../library/SAN/TestCode/model/BIX4_G20R10P48/results"
        for img in sorted(set([img.split("_")[0] for img in os.listdir(path)])):
            kind = lambda k: "%s_x%s_%s.png" % (img, super_resolution_choice, k)
            image_triplets.append([
                load_image(os.path.join(path, kind('HR'))),
                upscale(load_image(os.path.join(path, kind('LR'))), super_resolution_choice),
                load_image(os.path.join(path, kind('SR'))),
            ])
    elif set_choice != set_microscopy:
        for image_id in range(int(set_choice)):
            image_name_format = "{:02d}" if set_choice == set_12 else "{:03d}" if set_choice == set_68 else None
            image_name = image_name_format.format(image_id + 1)
            ground_truth_image = load_image(
                "../library/dataset/clean_data/Test/Set%s/%s.png" % (set_choice, image_name))
            noisy_image = ground_truth_image + np.random.normal(0, sigma / 255.0, ground_truth_image.shape)
            noisy_image = noisy_image.astype(np.float32)

            noisy_image_torch = torch.from_numpy(noisy_image).view(1, -1, noisy_image.shape[0], noisy_image.shape[1])
            denoised_image = denoise_model.forward(noisy_image_torch)
            denoised_image = denoised_image.view(noisy_image.shape[0], noisy_image.shape[1])
            denoised_image = denoised_image.detach().numpy().astype(np.float32)
            image_triplets.append([
                ground_truth_image, noisy_image, denoised_image
            ])
    else:
        path = "../library/microscopy/"
        folder = ["gt", "avg2", "dncnn_avg2"]
        for im in os.listdir(os.path.join(path, folder[0])):
            if im.endswith(".png"):
                name, ext = os.path.splitext(im)
                image_triplets.append([
                    load_image(os.path.join(path, folder[0], im)),
                    load_image(os.path.join(path, folder[1], im)),
                    load_image(os.path.join(path, folder[2], name + '_dncnn' + ext))
                ])
    return image_triplets


def fusion_eval(method="weighted_dwt"):
    global sigma
    sigma = 25
    load_models()
    image_triplets = load_images()
    ALL_PSNR = []
    ALL_SSIM = []
    for triplet in image_triplets[:1]:
        PSNR = []
        SSIM = []
        ground_truth_image, noisy_image, denoised_image = triplet
        confidence_map = predict_confidence(noisy_image, denoised_image, confidence_model)
        reliable_image = reliable_strategies[reliable_choice](noisy_image)
        n = 100
        for i in range(0, n + 1):
            coefficient = i / float(n - 1)
            if method == "weighted_dwt":
                fused_image_patch_confidence = fusion_strategies[method](reliable_image, denoised_image, coefficient,
                                                                         confidence_map=confidence_map,
                                                                         patch_wise=True)
                # fused_image_patch_no_confidence = fusion_strategies["weighted_dwt"](reliable_image, denoised_image,
                #                                                                    coefficient,
                #                                                                    confidence_map=None, patch_wise=True)
                fused_image_full = fusion_strategies[method](reliable_image, denoised_image, coefficient,
                                                             confidence_map=None, patch_wise=False)
            else:
                fused_image_patch_confidence = fusion_strategies[method](reliable_image, denoised_image, coefficient,
                                                                         confidence_map)
                fused_image_full = fusion_strategies["weighted_dct"](reliable_image, denoised_image, coefficient)
            psnr_patch_conf = compare_psnr(ground_truth_image, fused_image_patch_confidence)
            ssim_patch_conf = compare_ssim(ground_truth_image, fused_image_patch_confidence)
            # psnr_patch_no_conf = compare_psnr(ground_truth_image, fused_image_patch_no_confidence)
            # ssim_patch_no_conf = compare_ssim(ground_truth_image, fused_image_patch_no_confidence)
            psnr_full = compare_psnr(ground_truth_image, fused_image_full)
            ssim_full = compare_ssim(ground_truth_image, fused_image_full)
            PSNR.append([psnr_full, psnr_patch_conf])
            SSIM.append([ssim_full, ssim_patch_conf])
        ALL_PSNR.append(np.array(PSNR))
        ALL_SSIM.append(np.array(SSIM))
    all_psnr_array = np.array(ALL_PSNR)
    all_ssim_array = np.array(ALL_SSIM)
    avg_psnr = np.average(all_psnr_array, axis=0)
    avg_ssim = np.average(all_ssim_array, axis=0)
    return avg_psnr, avg_ssim


def plot_eval():
    avg_psnr_dwt, avg_ssim_dwt = fusion_eval("weighted_dwt")
    avg_psnr_dct, avg_ssim_dct = fusion_eval("adaptive_dct")
    plt.style.use(['seaborn-paper', 'seaborn-whitegrid'])
    plt.style.use(['seaborn'])
    # sns.set(palette='colorblind')
    sns.set(palette='muted')
    matplotlib.rc("font", family="Times New Roman", size=12)

    index = np.arange(-1, avg_psnr_dwt.shape[0])
    labels = index / (avg_psnr_dwt.shape[0] - 1) * 20

    fig, ax = plt.subplots(1, 2, figsize=(48, 8))
    # for i in range(2):
    #     if i == 0:
    #         ax[i, 0].plot(avg_psnr_dwt)
    #     else:
    #         ax[i, 0].plot(avg_psnr_dct)
    #     # ax[0].set_ylim([15, 33])
    #     # ax[0].set_ylim([14, 33])
    #     ax[i, 0].set_xlabel('Fusion weight')
    #     ax[i, 0].set_ylabel('PSNR')
    #     ax[i, 0].set_xticklabels(labels)
    #     ax[i, 0].legend(["Normal", "Confidence_guided"], loc="upper left")
    #
    #     if i == 0:
    #         ax[i, 1].plot(avg_ssim_dwt)
    #     else:
    #         ax[i, 1].plot(avg_ssim_dct)
    #     # ax[1].set_ylim([0.3, 1])
    #     # ax[1].set_ylim([0.2, 1])
    #     ax[i, 1].set_xlabel('Fusion weight')
    #     ax[i, 1].set_ylabel('SSIM')
    #     ax[i, 1].set_xticklabels(labels)
    #     ax[i, 1].legend(["Normal", "Confidence_Guided"], loc="upper left")
    ax[0].plot(avg_psnr_dwt)
    # ax[0].set_ylim([15, 33])
    # ax[0].set_ylim([14, 33])
    ax[0].set_xlabel('Fusion weight')
    ax[0].set_ylabel('PSNR')
    ax[0].set_xticklabels(labels)
    ax[0].legend(["Normal", "Confidence_guided"], loc="upper left")


    ax[1].plot(avg_ssim_dwt)
    # ax[1].set_ylim([0.3, 1])
    # ax[1].set_ylim([0.2, 1])
    ax[1].set_xlabel('Fusion weight')
    ax[1].set_ylabel('SSIM')
    ax[1].set_xticklabels(labels)
    ax[1].legend(["Normal", "Confidence_Guided"], loc="upper left")

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def main():
    global sigma
    sigma = 55
    noisy_images = []
    denoised_images = []
    ground_truths = []
    fused_images = []
    reliable_images = []
    confidence_maps = []
    load_models()
    image_triplets = load_images()

    for triplet in image_triplets:
        ground_truth_image, noisy_image, denoised_image = triplet
        # confidence_map = predict_confidence(noisy_image, denoised_image, confidence_model)

        ground_truths.append(ground_truth_image)
        noisy_images.append(noisy_image)
        denoised_images.append(denoised_image)
        # confidence_maps.append(confidence_map)

        n_insight = 2 if fusion_choice == "weighted_dct" else 0

        matrix = []
        titles = []

        for i in range(2 + n_insight):
            matrix.append([])
            titles.append([])

        matrix[0].append(ground_truth_image)
        titles[0].append("Ground truth\n\n")

        matrix[0].append(noisy_image)
        titles[0].append("Noisy input\nPSNR: %.2f\nSSIM: %.2f" % (
            compare_psnr(ground_truth_image, noisy_image), compare_ssim(ground_truth_image, noisy_image)))

        for i in range(n_insight):
            for k in range(2):
                matrix[2 + i].append(None)
                titles[2 + i].append(None)

        n = 5

        for i in range(0, n):
            insight = []
            coefficient = i / float(n - 1)
            (reliable_image, fused_image) = fuse_image(noisy_image, denoised_image, coefficient,
                                                       # confidence_map=confidence_map,
                                                       insight=insight)
            if i == 0:
                reliable_images.append(reliable_image)

            fused_images.append(fused_image)
            psnr = compare_psnr(ground_truth_image, fused_image)
            ssim = compare_ssim(ground_truth_image, fused_image)

            matrix[0].append(fused_image)
            titles[0].append("Fused %.2f\nPSNR: %.2f\nSSIM: %.2f" % (coefficient, psnr, ssim))

            for k, additional in enumerate(insight):
                if k + 2 >= len(matrix):
                    matrix.append([])
                    titles.append([])
                matrix[k + 2].append(additional)
                titles[k + 2].append(None)

        for i in range(len(matrix[0])):
            matrix[1].append(upscale_image_sample(matrix[0][i], 4))
            titles[1].append(None)

        titles[1][0] = "Magnified fragment"

        if fusion_choice == "weighted_dct":
            titles[2][2] = "Reliable image mask"
            titles[3][2] = "Denoised image mask"

        """ Show per image matrix """
        show_image_matrix(matrix, titles, "Fusion (%s, %s)" % (reliable_choice, fusion_choice))
    psd_plot(noisy_images, denoised_images, ground_truths, reliable_images, fused_images)


if __name__ == '__main__':
    # main()
    plot_eval()
