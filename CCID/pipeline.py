# Important: you must have Tkinter installed
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider, RadioButtons
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error
from torch.serialization import SourceChangeWarning

import CCID
from CCID.confidence.confidence import predict_confidence, load_confidence_model
from CCID.fusion.fusion import reliable_strategies, fusion_strategies, reliable_choice, \
    fusion_choice, is_super_resolution_choice
from CCID.fusion.reliable_sr import upscale
from CCID.plot_utils import upscale_image_sample

from CCID.confidence.models.ConvNet_region import *
from CCID.confidence.models.DnCNN import *

warnings.filterwarnings("ignore", category=SourceChangeWarning)  # Disable warnings
directory_path = os.path.dirname(os.path.realpath(__file__))

sigma = 25
ground_truth_image = None
noisy_image = None
denoised_image = None
image_cursor = 0 + 8
hide_ground_truth = True

confidence_model = None
denose_model = None


def load_models():
    global confidence_model
    global denoise_model

    confidence_model = load_confidence_model()

    denoise_model = torch.load(
        os.path.join(directory_path, "library/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth"),
        map_location=torch.device("cpu"))
    denoise_model.eval()  # Important


def gui_interaction():
    fig = plt.gcf()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    axcolor = "ivory"

    fusion_slider_ax = plt.axes([0.3, 0.05, 0.55, 0.03], facecolor=axcolor)
    fusion_slider = Slider(fusion_slider_ax, "Controlled fusion", 0.0, 1.0, valinit=0.5, valstep=0.01)

    confidence_threshold_slider_ax = plt.axes([0.3, 0.1, 0.55, 0.03], facecolor=axcolor)
    confidence_threshold_slider = Slider(confidence_threshold_slider_ax, "Confidence threshold", 0.0, 1.0, valinit=0.8,
                                         valstep=0.01)

    confidence_range_slider_ax = plt.axes([0.3, 0.15, 0.55, 0.03], facecolor=axcolor)
    confidence_range_slider = Slider(confidence_range_slider_ax, "Confidence smoothing", 0.25, 10.0, valinit=3,
                                     valstep=0.25)

    problem_type_choice = ["denoising", "super resolution"]
    problem_type_ax = plt.axes([0.025, 0.85, 0.15, 0.1], facecolor=axcolor)
    problem_type_radio = RadioButtons(problem_type_ax, problem_type_choice, active=problem_type_choice.index(
        "super resolution" if is_super_resolution_choice else "denoising"))

    reliable_choices = list(reliable_strategies.keys())
    reliable_ax = plt.axes([0.025, 0.65, 0.15, 0.15], facecolor=axcolor)
    reliable_radio = RadioButtons(reliable_ax, reliable_choices, active=reliable_choices.index(reliable_choice))

    fusion_choices = list(fusion_strategies.keys())
    fusion_ax = plt.axes([0.025, 0.45, 0.15, 0.15], facecolor=axcolor)
    fusion_radio = RadioButtons(fusion_ax, fusion_choices, active=fusion_choices.index(fusion_choice))

    display_zoom_choices = ["normal", "zoom x2^3", "zoom x2^4", "zoom x2^5"]
    display_zoom_ax = plt.axes([0.025, 0.25, 0.15, 0.15], facecolor=axcolor)
    display_zoom_radio = RadioButtons(display_zoom_ax, display_zoom_choices,
                                      active=display_zoom_choices.index("normal"))

    display_information_choices = ["normal", "abs error", "overlay", "hallucinations"]
    display_information_ax = plt.axes([0.025, 0.1, 0.15, 0.1], facecolor=axcolor)
    display_information_radio = RadioButtons(display_information_ax, display_information_choices,
                                             active=display_information_choices.index("hallucinations"))

    module_root = os.path.dirname(CCID.__file__)

    def load_images():
        global ground_truth_image
        global noisy_image
        global denoised_image

        is_super_resolution = problem_type_radio.value_selected == "super resolution"

        if is_super_resolution:
            sr = 4

            def load_image(kind):
                image_names = ["baby", "bird", "butterfly", "head", "woman"]
                image_name = image_names[image_cursor % len(image_names)]
                img_path = os.path.join(module_root, "library/SAN/TestCode/model/BIX4_G20R10P48/results/%s_x%s_%s.png" % (image_name, sr, kind))
                img = imread(img_path)
                bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return np.array(bw, dtype=np.float32) / 255.0

            ground_truth_image = load_image("HR")
            noisy_image = upscale(load_image("LR"), sr)
            denoised_image = load_image("SR")
        else:
            set_size = 12
            image_name = '{:02d}'.format((image_cursor % set_size) + 1)
            ground_truth_image = np.array(
                imread(os.path.join(module_root, "library/dataset/clean_data/Test/Set%s/%s.png" % (set_size, image_name))),
                dtype=np.float32) / 255.0
            noisy_image = ground_truth_image + np.random.normal(0, sigma / 255.0, ground_truth_image.shape)
            noisy_image = noisy_image.astype(np.float32)

            noisy_image_torch = torch.from_numpy(noisy_image).view(1, -1, noisy_image.shape[0], noisy_image.shape[1])
            denoised_image = denoise_model.forward(noisy_image_torch)
            denoised_image = denoised_image.view(noisy_image.shape[0], noisy_image.shape[1])
            denoised_image = denoised_image.detach().numpy().astype(np.float32)

    def update(ignore=None):
        confidence = predict_confidence(noisy_image, denoised_image, confidence_model)
        reliable_image = reliable_strategies[reliable_radio.value_selected](noisy_image)
        fusion_image = fusion_strategies[fusion_radio.value_selected](reliable_image, denoised_image, fusion_slider.val, confidence_map=confidence)
        display_zoom = display_zoom_radio.value_selected
        display_information = display_information_radio.value_selected

        confidence_range = confidence_range_slider.val
        confidence_threshold = confidence_threshold_slider.val

        is_mode_hallucinations = display_information == "hallucinations"

        # Hide/show components depending on mode
        confidence_threshold_slider.ax.set_visible(is_mode_hallucinations)
        confidence_range_slider.ax.set_visible(is_mode_hallucinations)

        def draw(image, nrows, ncols, index, title=None, compare_with=None, noisy=None, reliable=None, denoised=None):
            plt.subplot(nrows, ncols, index)
            ax = plt.gca()
            ax.axis('off')
            if compare_with is None and hide_ground_truth:  # Ground truth
                return
            if title:
                plt.title(title, size=10)

            def crop(img):
                return img if display_zoom == "normal" else upscale_image_sample(img,
                                                                                 int(display_zoom.split("zoom x2^")[1]))

            if compare_with is not None and display_information == "overlay":
                cropped_image = crop(image)
                cropped_compare = crop(compare_with)
                absolute_error = abs(cropped_image - cropped_compare)
                compare_rgb = cv2.cvtColor(cropped_compare, cv2.COLOR_GRAY2RGB)
                zeros = np.zeros(cropped_compare.shape)
                scale = 3  # Rescale for visibility
                compare_rgb += np.rollaxis(np.array([absolute_error * scale, zeros, zeros]), 0, 3)
                ax.imshow(np.clip(compare_rgb, 0, 1), vmin=0, vmax=1)
            elif compare_with is not None and display_information == "hallucinations" and noisy is not None and reliable is not None:
                # Show the confident part of the image
                upscaled = upscale(confidence, 8)
                sigma = confidence_range
                kernel_size = int(4 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                threshold = confidence_threshold
                averaged_confident = (upscaled >= threshold) * (upscaled - threshold) / (np.max(upscaled) - threshold)
                averaged_confident = cv2.GaussianBlur(averaged_confident, ksize=(kernel_size, kernel_size), sigmaX=sigma)
                averaged_unconfident = (upscaled < threshold) * (threshold - upscaled) / (threshold - np.min(upscaled))
                averaged_unconfident = cv2.GaussianBlur(averaged_unconfident, ksize=(kernel_size, kernel_size), sigmaX=sigma)
                cropped_image = crop(image)
                compare_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
                zeros = np.zeros(upscaled.shape)
                compare_rgb += crop(np.rollaxis(np.array([averaged_unconfident, averaged_confident, zeros]), 0, 3))
                ax.imshow(np.clip(compare_rgb, 0, 1), vmin=0, vmax=1)
            else:
                if compare_with is not None and display_information == "abs error":
                    image_to_display = abs(crop(image) - crop(compare_with)) * 2  # Rescale for the same reason
                    cmap = "inferno"
                else:
                    image_to_display = crop(image)
                    cmap = "Greys_r"
                ax.imshow(image_to_display, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)

        def title_with_measures(title, image):
            return "%s\nPSNR: %.2f\nSSIM: %.2f\nMSE: %.5f" % (
                title,
                compare_psnr(ground_truth_image, image),
                compare_ssim(ground_truth_image, image),
                mean_squared_error(ground_truth_image, image)
            )

        draw(ground_truth_image, 2, 4, 1, "Ground truth\n\n")
        draw(reliable_image, 2, 4, 2, title_with_measures("Reliable filter output", reliable_image), ground_truth_image)
        draw(noisy_image, 2, 4, 5, title_with_measures("Noisy input", noisy_image), ground_truth_image)
        draw(denoised_image, 2, 4, 6, title_with_measures("Denoised output", denoised_image), ground_truth_image)
        draw(fusion_image, 1, 2, 2, title_with_measures("Fusion output", fusion_image), ground_truth_image, noisy_image,
             reliable_image, denoised_image)

        fig.canvas.draw_idle()

    def handle_keypress(event):
        global image_cursor
        delta = None
        if event.key == 'right' or event.key == ' ':
            delta = 1
        elif event.key == 'left':
            delta = -1
        if delta is not None:
            image_cursor += delta
            load_images()
            update()

    # Event handling
    def handle_problem_type_change(val):
        # Reload images
        load_images()
        update()

    problem_type_radio.on_clicked(handle_problem_type_change)
    reliable_radio.on_clicked(update)
    fusion_radio.on_clicked(update)
    display_zoom_radio.on_clicked(update)
    display_information_radio.on_clicked(update)

    fusion_slider.on_changed(update)

    confidence_threshold_slider.on_changed(update)
    confidence_range_slider.on_changed(update)

    fig.canvas.mpl_connect('key_press_event', handle_keypress)

    handle_problem_type_change(None)  # Initial draw

    plt.show()


def main():
    """ Load confidence model and denoise model """
    load_models()
    """ Start GUI, allow users to play with different images and options """
    gui_interaction()


if __name__ == '__main__':
    main()
