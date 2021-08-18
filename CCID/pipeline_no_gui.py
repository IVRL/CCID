import argparse
import os
from skimage.io import imread, imsave

import CCID
from CCID.denoiser.denoiser import *
from CCID.fusion.fusion import fuse_image
from CCID.confidence.confidence import predict_confidence

""" This .py file will control the full logic of the program """


def parse_args():
    module_root = os.path.dirname(CCID.__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", default=os.path.join(module_root, "library/dataset/clean_data/Test/Set12"),
                        type=str, help="Directory containing input images.")
    parser.add_argument("--output-directory", default=os.path.join(module_root, "report_data"), type=str,
                        help="Directory to which the output will be written to.")
    parser.add_argument("--dncnn-model",
                        default=os.path.join(module_root,
                                             "library/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth"),
                        type=str, help="Path to the DnCNN model to use.")
    parser.add_argument("--sigma", default=55, type=int, help="Noise level to be artificially added.")
    parser.add_argument("--weight", default=0.5, type=float, help="Fusion weight.")
    return parser.parse_args()


def main():
    """The main test code."""

    args = parse_args()

    sigma_normalized = args.sigma / 255.0

    """ For reproducibility """
    np.random.seed(seed=0)

    if not os.path.isdir(args.input_directory):
        raise Exception("Not a directory: %s" % args.input_directory)

    for image_name in os.listdir(args.input_directory):
        assert any(image_name.endswith(ext) for ext in ["png", "jpg", "jpeg", "bmp"])

        model = torch.load(args.dncnn_model,
                           map_location=torch.device("cpu"))
        model.eval()
        original_image = np.array(imread(os.path.join(args.input_directory, image_name)), dtype=np.float32) / 255.0
        noisy_image = original_image + np.random.normal(0, sigma_normalized, original_image.shape)
        noisy_image = noisy_image.astype(np.float32)
        noisy_image_torch = torch.from_numpy(noisy_image).view(1, -1, noisy_image.shape[0],
                                                               noisy_image.shape[1])

        """ Take noisy image, return the denoised one.
            The denoise is done using DnCNN model, trained on sigma=25
        """
        denoised_image = model(noisy_image_torch)
        denoised_image = denoised_image.view(noisy_image.shape[0], noisy_image.shape[1])
        denoised_image = denoised_image.cpu()
        denoised_image = denoised_image.detach().numpy().astype(np.float32)

        """ Take noisy image and denoised image, return the predicted confidence map.
            The confidence is per region based, each 8x8 region have a confidence value.
        """
        confidence_map = predict_confidence(noisy_image, denoised_image)
        residual_image = noisy_image - denoised_image

        """ Take noisy image, denoised image and confidence map, together with the user input
            fusion weight, return the fused image 
        """
        reliable_image, fused_image = fuse_image(noisy_image, denoised_image, args.weight,
                                                 confidence_map=confidence_map)
        # print("Here")
        # imsave(os.path.join(args.output_directory, "different_sigma_values/sigma=55/denoised_image/{}".format(image_name)),
        #        (np.clip(denoised_image, 0, 1) * 255).astype(np.uint8))
        # imsave(os.path.join(args.output_directory, "different_sigma_values/sigma=55/reliable_image/{}".format(image_name)),
        #        (np.clip(reliable_image, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
