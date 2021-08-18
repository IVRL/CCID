# CCID/library/DnCNN/TrainingCodes/dncnn_pytorch/data/Train400/test_001.png

import glob

import cv2
import os
import json
import numpy as np
import torch
from CCID.confidence.models.DnCNN import DnCNN
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.measure
import CCID.fusion.fusion as fusion

patch_size, stride = 40, 20
scales = [1, 0.8]

training_dataset_path = "CCID/library/DnCNN/TrainingCodes/dncnn_pytorch/data/Train400"
testing_dataset_path = "CCID/library/DnCNN/TrainingCodes/dncnn_pytorch/data/Test/Set68"

saved_testing_data_path = "CCID/confidence_map_test_data"
saved_training_data_path = "CCID/confidence_map_train_data"

inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate_confidence_map(denoised_img, ground_truth_img):
    # if difference is larger than 100 in the range of 255, then confidence is 0
    # if difference is within 0 and 100, then linear decrease from 1 to 0
    # down-sampling, average difference in 8 by 8 region
    abs_difference_per_pixel = torch.absolute(denoised_img - ground_truth_img)
    # reduced_resolution = skimage.measure.block_reduce(abs_difference_per_pixel, (8, 8), np.average)
    reduced_resolution = F.avg_pool2d(abs_difference_per_pixel, 8)
    return torch.clip(1 - reduced_resolution / (100 / 255), 0, 1)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return torch.flipud(img)
    elif mode == 2:
        return torch.rot90(img)
    elif mode == 3:
        return torch.flipud(torch.rot90(img))
    elif mode == 4:
        return torch.rot90(img, k=2)
    elif mode == 5:
        return torch.flipud(torch.rot90(img, k=2))
    elif mode == 6:
        return torch.rot90(img, k=3)
    elif mode == 7:
        return torch.flipud(torch.rot90(img, k=3))

class ImagesDataset(Dataset):
    def __init__(self, dataset_path, saved_dataset_dir, aug_count, noise_count, verbose=False):
        self.dataset_dir = saved_dataset_dir
        self.aug_count = aug_count
        self.noise_count = noise_count
        self.verbose = verbose
        self.orig_dataset_dir = dataset_path
        self.num_elems = self.data_generator()

    def __len__(self):
        return self.num_elems

    def __getitem__(self, idx):
        input = np.load(self.dataset_dir + "/input_" + str(idx) + ".npy")
        output = np.load(self.dataset_dir + "/output_" + str(idx) + ".npy")
        input = input.transpose((2, 0, 1))
        # output = output.transpose((2, 0, 1))

        return input, output[np.newaxis, ...]

    def gen_patches(self, file_name, model):

        def to_2_dim(arg):
            # Calling squeeze() with no arguments might delete
            # some other dimensions if they are 1
            return arg.squeeze(0).squeeze(0).cpu().detach().numpy()

        # Get multiscale patches from a single image
        img = torch.Tensor(cv2.imread(file_name, 0)).to(inference_device) / 255.0  # gray scale
        h, w = img.shape
        img.unsqueeze_(0).unsqueeze_(0)
        patches = []
        target_patches = []
        for _ in range(0, self.aug_count):
            for s in scales:
                h_scaled, w_scaled = int(h * s), int(w * s)
                # img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC).transpose()
                img_scaled = F.interpolate(img, (h_scaled, w_scaled), mode='bicubic', align_corners=False)
                assert img.shape[0] * img.shape[1] == 1
                # extract patches
                for i in range(0, h_scaled - patch_size, stride):
                    for j in range(0, w_scaled - patch_size, stride):
                        for _ in range(self.noise_count):
                            noise_level = np.random.uniform(0, 100)
                            x = img_scaled[:, :, i:i + patch_size, j:j + patch_size]
                            assert x.shape[0] * x.shape[1] == 1

                            # Augment data and add noise
                            x_aug = data_aug(x, mode=np.random.randint(0, 8))

                            noise = x.new_empty(x.shape).normal_(mean=0, std=noise_level / 255)
                            x_aug_noise = x_aug + noise
                            x_aug_noise = torch.clip(x_aug_noise, 0, 1)
                            # Predict using DnCNN

                            # Compute all three `input` images
                            denoised_x = model(x_aug_noise)
                            residual_x = to_2_dim(x_aug_noise - denoised_x)
                            confidence_map_x = to_2_dim(generate_confidence_map(denoised_x, x_aug))

                            x_aug_noise = to_2_dim(x_aug_noise)
                            # reliable image is generated using gaussian, currently
                            reliable_image = fusion.reliable_strategies["gaussian"](x_aug_noise)
                            input = np.stack([residual_x, x_aug_noise, reliable_image], axis=2)
                            patches.append(input)
                            target_patches.append(confidence_map_x)
        return patches, target_patches



    def data_generator(self):
        model = torch.load(
            "CCID/library/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth")\
            .to(inference_device)
    
        saved_data_dir = self.dataset_dir
        data_dir = self.orig_dataset_dir

        model.eval()
        # generate clean patches from a dataset
        file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
        # initialize
        input = []
        output = []
        processed_images_path = saved_data_dir + "/processed_images.json"
        if not os.path.exists(saved_data_dir):
            os.makedirs(saved_data_dir)
        try:
            with open(saved_data_dir + "/processed_images.json", 'r') as f:
                savestate = json.load(f)
                processed_images = set(savestate['processed_input'])
                next_id = savestate['latest_id'] + 1
        except IOError:
            print("Processed image list not found. Generating from scratch")
            processed_images = set()
            next_id = 0
    
        # generate patches
        for i in range(len(file_list)):
            file = file_list[i]
            if file not in processed_images:
                patches, target_patches = self.gen_patches(file, model)
                for (patch, target_patch) in zip(patches, target_patches):
                    file_path = saved_data_dir + "/input_" + str(next_id)
                    np.save(file_path, patch)
                    file_path = saved_data_dir + "/output_" + str(next_id)
                    np.save(file_path, target_patch)
                    next_id += 1
    
                processed_images.add(file)
                with open(processed_images_path, 'w') as f:
                    savestate = {'latest_id': next_id - 1, 'processed_input': list(processed_images)}
                    json.dump(savestate, f)
    
            if self.verbose:
                print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    
        if self.verbose:
            print("^_^-training data finished-^_^")
    
        return next_id



if __name__ == '__main__':
    input, output = data_generator()
    sum_output = np.vstack(output.numpy()).flatten()
    plt.hist(sum_output, bins=100)
    plt.title("Overall, sigma from 5 to 145, step 10")
    plt.show()
