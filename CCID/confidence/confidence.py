import os

import torch
import numpy as np
from CCID.confidence.models.ConvNet_region import ConvNet_region


def load_confidence_model(saved_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "saved_model/model.cpkt"), device="cpu"):
    """ Load the model """
    # device = "cuda:0"
    model = ConvNet_region(n_channels=3)
    checkpoint = torch.load(saved_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()  # Important
    return model


def predict_confidence(noisy_image, denoised_image, confidence_model=None):
    """ Given the original noisy image and denoised image,
        this function will returns how confident we are about
        the denoised image. """
    from CCID.fusion.fusion import reliable_strategies

    if confidence_model is None:
        confidence_model = load_confidence_model()

    """ Prepare the input for the network """
    residual_image = noisy_image - denoised_image
    reliable_image = reliable_strategies["gaussian"](noisy_image)

    """ The input to the network is the stack of residual_image, noisy image, together with reliable_image """
    stacked_images = np.stack([residual_image, noisy_image, reliable_image], axis=2)
    stacked_images = stacked_images.transpose((2, 0, 1))

    """ The shape of the input is NCHW """
    stacked_images_torch = torch.from_numpy(stacked_images) \
        .view(1, stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2])

    """ Perform the inference"""
    confidence_map = confidence_model(stacked_images_torch)
    confidence_map = confidence_map.view(confidence_map.shape[2], confidence_map.shape[3])
    confidence_map = confidence_map.detach().numpy().astype(np.float32)
    return confidence_map
