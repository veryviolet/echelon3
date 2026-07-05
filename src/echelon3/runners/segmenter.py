import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from echelon3.runners.video import VideoRunner
from echelon3.runners.images import ImagesRunner
import cv2


def log_fun(frame: np.ndarray, output: np.ndarray, colormap, opacity) -> np.ndarray:
    output = np.squeeze(output)

    color_mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)

    for c, v in colormap.items():
        color_mask[output == int(c)] = v

    blended = cv2.addWeighted(frame, opacity, color_mask, 1 - opacity, 0)

    return blended


class VideoSegmenter(VideoRunner):

    colormap: DictConfig = None
    opacity: float = None

    def __init__(self, source, target, colormap, opacity=0.5, **kwargs):
        super(VideoRunner, self).__init__(source=source, target=target)
        self.colormap = colormap
        self.opacity = opacity

    def log(self, frame: np.ndarray, output: np.ndarray) -> np.ndarray:
        return log_fun(frame, output, self.colormap, self.opacity)


class ImagesSegmenter(ImagesRunner):

    colormap: DictConfig = None
    opacity: float = None

    def __init__(self, source, target, colormap, opacity=0.5, **kwargs):
        super(ImagesRunner, self).__init__(source=source, target=target)
        self.colormap = colormap
        self.opacity = opacity

    def log(self, frame: np.ndarray, output: np.ndarray) -> np.ndarray:
        return log_fun(frame, output, self.colormap, self.opacity)
