import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct


def simple_weighted_fusion(reliable_image, denoised_image, fusion_weight, confidence_map=None, **kwargs):
    """ Simple example to fuse reliable input image and denoised image """
    estimated_noise = reliable_image - denoised_image
    fused_image = reliable_image - fusion_weight * estimated_noise
    return fused_image


def simple_dct_fusion(reliable_image, denoised_image, fusion_weight, confidence_map=None, insight=None):
    """ Simple fusion strategy, in the frequency domain (DCT). """
    size = reliable_image.shape[0]

    distance_values = np.sqrt(np.fromfunction(lambda y, x: x * x + y * y, reliable_image.shape[:2], dtype=float)) / size

    epsilon = 0.001  # A value sufficiently small
    scale = 0.1  # To be optimized
    sigma = scale * (1 / (1 + epsilon - fusion_weight) - 1)
    mask_denoised = np.exp(-(distance_values * distance_values) / (2 * sigma))
    mask_reliable = 1.0 - mask_denoised

    reliable_frequencies = cv2.dct(reliable_image)
    denoised_frequencies = cv2.dct(denoised_image)
    combined_frequencies = np.multiply(reliable_frequencies, mask_reliable) \
                           + np.multiply(denoised_frequencies, mask_denoised)
    inversed = cv2.idct(combined_frequencies).astype('float32')
    if insight is not None:
        insight.append(mask_reliable)
        insight.append(mask_denoised)
    return inversed


def adaptive_dct_fusion(reliable_image, denoised_image, fusion_weight, confidence_map, **kwargs):
    """."""
    discretization_step = 0.01
    total_steps = round(1 / discretization_step) + 1
    discretized_fusions = []
    for i in range(total_steps):
        w = discretization_step * i
        discretized_fusions.append(reliable_image if i == 0 else
                                   denoised_image if i == total_steps - 1 else
                                   simple_dct_fusion(reliable_image, denoised_image, w))

    confidence_smooth = cv2.resize(confidence_map, reliable_image.shape, 0, 0, interpolation=cv2.INTER_CUBIC)

    output = np.zeros(reliable_image.shape, np.float32)

    for i in range(reliable_image.shape[0]):
        for j in range(reliable_image.shape[1]):
            w = (confidence_smooth[i][j] - (1 - fusion_weight)) / fusion_weight
            w = min(max(w, 0.0), 1.0)
            w_i = round(w / discretization_step)
            output[i][j] = discretized_fusions[w_i][i][j]

    return output


def exponential_dct_fusion(reliable_image, denoised_image, fusion_weight, confidence_map=None, **kwargs):
    heightSq = reliable_image.shape[0] ** 2
    widthSq = reliable_image.shape[1] ** 2
    a = fusion_weight
    denoised_weight = np.clip(np.fromfunction(lambda y, x: a ** (x ** 2 / widthSq + y ** 2 / heightSq),
                                              reliable_image.shape[:2], dtype=float), 0, 1)

    reliable_weight = 1 - denoised_weight

    reliable_dct = cv2.dct(reliable_image)
    denoised_dct = cv2.dct(denoised_image)
    combined_dct = reliable_dct * reliable_weight + denoised_dct * denoised_weight
    combined = cv2.idct(combined_dct).astype("float32")
    return combined


def primitive_radius_fusion(reliable_image, denoised_image, fusion_weight, confidence_map=None):
    num_points = 20
    max_radius = np.sqrt(reliable_image.shape[1] ** 2 + reliable_image.shape[0] ** 2)
    fusion_weights_v = np.linspace(0, max_radius, num_points)
    fusion_weights_r = np.ones(fusion_weights_v.shape)
    reliable_dct = cv2.dct(reliable_image)
    denoised_dct = cv2.dct(denoised_image)
    return cv2.idct(l2_weighed_dct_fusion(reliable_dct, denoised_dct, fusion_weights_r, fusion_weights_v))


def l2_weighed_dct_fusion(reliable_dct, denoised_dct, fusion_weights_r, fusion_weights_v, confidence_map=None):
    denoised_mask = np.interp(
        np.fromfunction(lambda y, x: np.sqrt(x ** 2 + y ** 2)), fusion_weights_r, fusion_weights_v)
    reliable_mask = 1 - denoised_mask
    return reliable_mask * reliable_dct + denoised_mask * denoised_dct


def threshold_fusion(reliable_image, denoised_image, fusion_weight, confidence_map, **kwargs):
    tpe = 2
    reliable_dct = dct(reliable_image, type=tpe)
    denoised_dct = dct(denoised_image, type=tpe)

    diff = np.abs(reliable_dct - denoised_dct)
    t_max = np.max(diff) * fusion_weight
    t_min = t_max / 2

    return idct(threshold_dct_fusion(reliable_dct, denoised_dct, t_min, t_max))


def threshold_dct_fusion(reliable_dct, denoised_dct, t_min, t_max, confidence_map):
    mask = np.zeros(reliable_dct.shape)
    diff = denoised_dct - reliable_dct
    mask = np.clip(((diff - t_min) / (t_max - t_min)), 0, 1)
    # Don't reduce frequencies that the reliable filter produced
    mask[diff < 0] = 1
    return reliable_dct * mask + (1 - mask) * denoised_dct


def dwt_fusion(reliable_image, denoised_image, fusion_weight, confidence_map=None, patch_wise=True, **kwargs):
    """ fusion strategy, using DWT. """
    confidence_threshold = 0.8

    def transform_each_patch(reliable_patch, denoised_patch, updated_fusion_weight):
        """ First: Do wavelet transform on each patch """
        wavelet = 'db1'
        reliable_coeff = pywt.wavedec2(reliable_patch, wavelet, level=3)
        denoised_coeff = pywt.wavedec2(denoised_patch, wavelet, level=3)

        """ Second: for each level in both patch do the fusion """
        fused_coeff = []
        for i in range(len(reliable_coeff)):
            if i == 0:
                """ LL
                    The first values in each decomposition is the approximation values of the top level """
                fused_coeff.append(
                    (1 - updated_fusion_weight) * reliable_coeff[0] + updated_fusion_weight * denoised_coeff[0])
            else:
                """ LH, HL, HH
                    For the rest of the levels we have tuples with 3 coefficients """
                coeff = []
                for j in range(3):
                    coeff.append(
                        (1 - updated_fusion_weight) * reliable_coeff[i][j] + updated_fusion_weight * denoised_coeff[i][
                            j])

                fused_coeff.append(tuple(coeff))

        """ Third: After we fused the coefficient we need to transfer back to get the patch """
        fused_patch = pywt.waverec2(fused_coeff, wavelet)
        return fused_patch

    if patch_wise:
        patch_size = 8
        h, w = denoised_image.shape
        if confidence_map is None:
            confidence_map = np.full((h, w), confidence_threshold)
        h_confidence, w_confidence = confidence_map.shape
        fused_image = np.zeros(shape=(h, w))
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                reliable_patch = reliable_image[i:i + patch_size, j:j + patch_size]
                denoised_patch = denoised_image[i:i + patch_size, j:j + patch_size]
                i_of_confidence = i // patch_size
                j_of_confidence = j // patch_size

                if (i_of_confidence >= h_confidence
                        or j_of_confidence >= w_confidence):
                    confidence_value = confidence_threshold
                else:
                    confidence_value = confidence_map[i_of_confidence, j_of_confidence]

                """ If confidence value is 0.5, the weight is unchanged
                    If the confidence is 1, then give more weight to the denoised image
                    Else, give more weight to the reliable image
                """
                updated_fusion_weight = fusion_weight * (1 + (confidence_value - confidence_threshold))
                if updated_fusion_weight > 1:
                    updated_fusion_weight = 1
                fused_patch = transform_each_patch(reliable_patch, denoised_patch, updated_fusion_weight)
                fused_image[i:i + patch_size, j:j + patch_size] = fused_patch
    else:
        fused_image = transform_each_patch(reliable_image, denoised_image, fusion_weight)
    return fused_image.astype('float32')
