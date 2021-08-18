from abc import ABC, abstractmethod

import bm3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Denoiser(ABC):
    """Represents a trained denoiser."""

    @abstractmethod
    def forward(self, noisy_image):
        """Takes a noisy image as input, returns a denoised version."""
        pass


class DenoiserBM3D(Denoiser):
    """A BM3D denoiser."""

    def __init__(self, sigma_psd=25 / 255.0):
        self.sigma_psd = sigma_psd

    def forward(self, noisy_image):
        return bm3d.bm3d(noisy_image, sigma_psd=self.sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)


class DnCNN(nn.Module):
    """Helper class to hold the actual pytorch model. Author: not us."""

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class DenoiserDnCNN(Denoiser):
    """A DnCNN denoiser."""

    def __init__(self, model_path, enable_cuda=True):
        self.model = torch.load(model_path)
        # running on CPU
        # self.model = torch.load(model_path, map_location='cpu')
        if torch.cuda.is_available() and enable_cuda:
            self.model = self.model.cuda()

    def forward(self, noisy_image):
        y = noisy_image
        y = y.astype(np.float32)
        y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

        torch.cuda.synchronize()
        y_ = y_.cuda()
        x_ = self.model(y_)  # inference
        x_ = x_.view(y.shape[0], y.shape[1])
        x_ = x_.cpu()
        x_ = x_.detach().numpy().astype(np.float32)

        return x_
