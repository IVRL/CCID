import os
import numpy as np
from skimage.io import imread, imsave

gaussian_sigma = 25  # consistent with DnCNN
clean_img_directory = "clean_data"
noisy_img_directory = "noisy_data"
sub_folders = ["Test/Set12", "Test/Set68"]


def check_folder():
    for sub_folder in sub_folders:
        sub_path = os.path.join(noisy_img_directory, sub_folder)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)


def generate_noisy_image():
    for sub_folder in sub_folders:
        for im in os.listdir(os.path.join(clean_img_directory, sub_folder)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = np.array(imread(os.path.join(clean_img_directory, sub_folder, im)), dtype=np.float64) / 255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, gaussian_sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float64)
                name, ext = os.path.splitext(im)
                imsave(os.path.join(noisy_img_directory, sub_folder, name + "_sigma{}".format(gaussian_sigma) + ext),
                       (np.clip(y, 0, 1) * 255).astype(np.uint8))


if __name__ == '__main__':
    check_folder()
    generate_noisy_image()
