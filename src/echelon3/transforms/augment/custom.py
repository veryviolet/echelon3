import albumentations
import cv2
import numpy as np
import random
import torch

from albumentations import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform


class CenterCrop512x512(ImageOnlyTransform):
    """Center crop to 512x512"""

    def __init__(self, **kwargs):
        super(CenterCrop512x512, self).__init__(**kwargs)

    def apply(self, img, **params):
        source_height = img.shape[0]
        source_width = img.shape[1]

        # Calculate crop boundaries
        start_y = (source_height - 512) // 2
        start_x = (source_width - 512) // 2

        # Ensure we don't go out of bounds
        if start_y < 0 or start_x < 0:
            raise ValueError(f"Image too small for 512x512 crop. Image shape: {img.shape}")

        # Perform the crop
        return img[start_y:start_y + 512, start_x:start_x + 512, :]

    def get_transform_init_args_names(self):
        return ()


class CropToAspectRatioV1(ImageOnlyTransform):
    height = None
    width = None
    aspect_ratio = None
    deviation = 0

    def __init__(self, height, width, deviation=0, **kwargs):
        self.height = height
        self.width = width
        self.aspect_ratio = (1.0 * self.height) / self.width
        self.deviation = deviation

        assert self.deviation >= 0, "Deviation can't be negative"

        kwargs.pop("height", None)
        kwargs.pop("width", None)

        super(CropToAspectRatioV1, self).__init__(**kwargs)

    def apply(self, img, **params):
        source_height = img.shape[0]
        source_width = img.shape[1]

        greater = (1.0 * source_height) / source_width >= self.aspect_ratio

        if greater:
            new_width = source_width
            new_height = (new_width * self.height) // self.width
            delta = np.abs((source_height - new_height)) // 2

            # Apply random deviation to the vertical cropping
            random_deviation = random.randint(-self.deviation, self.deviation)
            random_deviation = np.clip(random_deviation, -delta, delta)
            return img[
                delta + random_deviation : (delta + random_deviation + new_height), :, :
            ]
        else:
            new_height = source_height
            new_width = (new_height * self.width) // self.height
            delta = np.abs((source_width - new_width)) // 2

            # Apply random deviation to the horizontal cropping
            random_deviation = random.randint(-self.deviation, self.deviation)
            random_deviation = np.clip(random_deviation, -delta, delta)
            return img[
                :, delta + random_deviation : (delta + random_deviation + new_width), :
            ]

    def get_transform_init_args_names(self):
        return ("height", "width", "deviation")


class To01(ImageOnlyTransform):
    def apply(self, img, **params):
        img = img.astype("float32")
        img = img / 255.0
        return img

    def get_transform_init_args_names(self):
        return ()


class From01(ImageOnlyTransform):
    def apply(self, img, **params):
        img = img * 255.0
        img[img < 0] = 0
        img[img > 255.0] = 255.0
        img = img.astype("uint8")

        return img

    def get_transform_init_args_names(self):
        return ()


class CropToAspectRatio(ImageOnlyTransform):
    height = None
    width = None
    aspect_ratio = None

    def __init__(self, **kwargs):
        self.height = kwargs.get("height")
        self.width = kwargs.get("width")
        self.aspect_ratio = (1.0 * self.height) / self.width

        kwargs.pop("height", None)
        kwargs.pop("width", None)

        super(CropToAspectRatio, self).__init__(**kwargs)

    def apply(self, img, **params):
        source_height = img.shape[0]
        source_width = img.shape[1]

        greater = (1.0 * source_height) / source_width >= self.aspect_ratio

        if greater:
            new_width = source_width
            new_height = (new_width * self.height) // self.width
            delta = np.abs((source_height - new_height)) // 2
            return img[delta : (delta + new_height), :, :]
        else:
            new_height = source_height
            new_width = (new_height * self.width) // self.height
            delta = np.abs((source_width - new_width)) // 2
            return img[:, delta : (delta + new_width), :]

    def get_transform_init_args_names(self):
        return ()


class FrequencyNoiseAddition(ImageOnlyTransform):
    def __init__(self, noise_factor=[500.0, 1000.0], p=1.0, always_apply=True):
        super(FrequencyNoiseAddition, self).__init__(always_apply, p)
        self.noise_factor_min = noise_factor[0]
        self.noise_factor_max = noise_factor[1]

    def apply(self, img, **params):
        img = img.astype(np.float32)

        # Convert image to torch tensor
        img_tensor = torch.tensor(img).permute(2, 0, 1)  # Change to [C, H, W]

        # Apply noise to each channel separately
        for channel in range(img_tensor.shape[0]):
            img_tensor[channel] = self._apply_noise(img_tensor[channel])

        # Convert back to numpy and uint8
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_noise(self, img_channel):
        noise_factor = np.random.uniform(self.noise_factor_min, self.noise_factor_max)
        noise = torch.randn_like(img_channel) * noise_factor

        # Convert image channel to frequency domain
        f_transform = torch.fft.fft2(img_channel)
        f_shift = torch.fft.fftshift(f_transform)

        # Add noise
        f_shift_noisy = f_shift + noise

        # Convert back to spatial domain
        f_ishift = torch.fft.ifftshift(f_shift_noisy)
        img_back = torch.fft.ifft2(f_ishift).real

        return img_back


class FrequencyFilter(ImageOnlyTransform):
    def __init__(self, filter_type="low", cutoff=[0.1, 0.5], p=1.0, always_apply=False):
        super(FrequencyFilter, self).__init__(always_apply, p)
        self.filter_type = filter_type
        self.cutoff_min = cutoff[0]
        self.cutoff_max = cutoff[-1]

    def apply(self, img, **params):
        img = img.astype(np.float32)

        # Convert image to torch tensor
        img_tensor = torch.tensor(img).permute(2, 0, 1)  # Change to [C, H, W]

        # Apply the filter to each channel separately
        for channel in range(img_tensor.shape[0]):
            img_tensor[channel] = self._apply_filter(img_tensor[channel])

        # Convert back to numpy and uint8
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_filter(self, img_channel):
        rows, cols = img_channel.shape
        crow, ccol = rows // 2, cols // 2

        cutoff = np.random.uniform(self.cutoff_min, self.cutoff_max)

        # Create a mask
        mask = np.zeros((rows, cols), np.float32)
        if self.filter_type == "low":
            mask[
                crow - int(crow * cutoff) : crow + int(crow * cutoff),
                ccol - int(ccol * cutoff) : ccol + int(ccol * cutoff),
            ] = 1
        elif self.filter_type == "high":
            mask[: crow - int(crow * cutoff), :] = 1
            mask[crow + int(crow * cutoff) :, :] = 1
            mask[:, : ccol - int(ccol * cutoff)] = 1
            mask[:, ccol + int(ccol * cutoff) :] = 1

        mask = torch.tensor(mask)

        # Convert image channel to frequency domain
        f_transform = torch.fft.fft2(img_channel)
        f_shift = torch.fft.fftshift(f_transform)

        # Apply the mask
        f_shift_filtered = f_shift * mask

        # Convert back to spatial domain
        f_ishift = torch.fft.ifftshift(f_shift_filtered)
        img_back = torch.fft.ifft2(f_ishift).real

        return img_back


class RandomSquareCropAndFill(ImageOnlyTransform):
    def __init__(self, square_size, fill_value=0, amount=1, max_deviation=0, **kwargs):
        self.square_size = square_size
        self.fill_value = fill_value
        self.amount = amount
        self.max_deviation = max_deviation
        super(RandomSquareCropAndFill, self).__init__(**kwargs)

    def apply(self, img, **params):
        transformed_image = img.copy()
        img_height, img_width = img.shape[:2]

        center_y, center_x = img_height // 2, img_width // 2

        # Adjust max_deviation if larger than image dimensions
        max_deviation_y = min(self.max_deviation, img_height // 2)
        max_deviation_x = min(self.max_deviation, img_width // 2)

        for _ in range(self.amount):
            # Apply deviation to the center coordinates
            dev_y = random.randint(-max_deviation_y, max_deviation_y)
            dev_x = random.randint(-max_deviation_x, max_deviation_x)
            adjusted_center_y = np.clip(center_y + dev_y, 0, img_height)
            adjusted_center_x = np.clip(center_x + dev_x, 0, img_width)

            # Calculate the range for the random crop center around the adjusted center
            y_start = max(0, adjusted_center_y - self.square_size // 2)
            y_end = min(img_height, adjusted_center_y + self.square_size // 2)
            x_start = max(0, adjusted_center_x - self.square_size // 2)
            x_end = min(img_width, adjusted_center_x + self.square_size // 2)

            # Ensure valid range for random.randint
            if y_end - self.square_size <= y_start:
                crop_y_start = y_start
            else:
                crop_y_start = random.randint(y_start, y_end - self.square_size)

            if x_end - self.square_size <= x_start:
                crop_x_start = x_start
            else:
                crop_x_start = random.randint(x_start, x_end - self.square_size)

            # Fill the cropped area with the specified fill value
            transformed_image[
                crop_y_start : crop_y_start + self.square_size,
                crop_x_start : crop_x_start + self.square_size,
            ] = self.fill_value

        return transformed_image

    def get_transform_init_args_names(self):
        return ("square_size", "fill_value", "amount", "max_deviation")


class Moire(ImageOnlyTransform):
    def __init__(self, **kwargs):
        super(Moire, self).__init__(**kwargs)

    def add_moire_noise(self, src):
        height, width = src.shape[:2]
        center = (height // 2, width // 2)
        degree = random.uniform(0.0005, 0.01)

        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        offset_X = X - center[0]
        offset_Y = Y - center[1]

        theta = np.arctan2(offset_Y, offset_X)
        rou = np.sqrt(offset_X**2 + offset_Y**2)

        new_X = center[0] + rou * np.cos(theta + degree * rou)
        new_Y = center[1] + rou * np.sin(theta + degree * rou)

        new_X = np.clip(new_X, 0, width - 1).astype(np.int32)
        new_Y = np.clip(new_Y, 0, height - 1).astype(np.int32)

        dst = 0.8 * src + 0.2 * src[new_Y, new_X]

        return dst.astype(np.uint8)

    def apply(self, img, **params):
        img = self.add_moire_noise(img)
        return img


class AspectPreservingDownscaleUpscale(ImageOnlyTransform):
    def __init__(
        self,
        downscale_factor_range,
        orig_height,
        orig_width,
        interpolation_down=1,
        interpolation_up=0,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.downscale_factor_range = downscale_factor_range
        self.orig_height = orig_height
        self.orig_width = orig_width
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, **params):
        # Randomly select a downscale factor within the specified range
        downscale_factor = random.uniform(
            self.downscale_factor_range[0], self.downscale_factor_range[1]
        )

        # Calculate new dimensions while maintaining aspect ratio
        h, w = img.shape[:2]
        down_height = int(h * downscale_factor)
        down_width = int(w * downscale_factor)

        # Resize down to smaller dimensions while maintaining aspect ratio
        img_down = cv2.resize(
            img, (down_width, down_height), interpolation=self.interpolation_down
        )

        # Resize back to original dimensions
        img_up = cv2.resize(
            img_down,
            (self.orig_width, self.orig_height),
            interpolation=self.interpolation_up,
        )

        return img_up
