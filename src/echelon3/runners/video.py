from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from echelon3.runners.baseline import Runner


class VideoRunner(Runner):

    capture = None
    writer = None

    def __init__(self, source, target):
        super(VideoRunner, self).__init__(source=source, target=target)

    def process(self, model: nn.Module, preprocess: nn.Module, postprocess: nn.Module):

        self.capture = cv2.VideoCapture(self.source)
        target_fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(self.target, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                      target_fps,
                                      (frame_width, frame_height))
        with tqdm() as pbar:
            pbar.set_description("--> Processing")
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = self.process_frame(rgb_frame, model, preprocess, postprocess)
                log = self.log(frame, processed_frame)
                self.writer.write(log)
                pbar.update(1)

        self.capture.release()
        self.writer.release()
