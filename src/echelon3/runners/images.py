from abc import ABC, abstractmethod
import os
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
import torch.nn as nn

from echelon3.runners.baseline import Runner


class ImagesRunner(Runner):

    def __init__(self, source, target):
        super(ImagesRunner, self).__init__(source=source, target=target)

    def process(self, model: nn.Module, preprocess: nn.Module, postprocess: nn.Module):

        for filename in tqdm(glob(os.path.join(self.source, '**', '*.*'), recursive=True)):
            frame = cv2.imread(filename)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.process_frame(rgb_frame, model, preprocess, postprocess)
            log = self.log(frame, processed_frame)
            os.makedirs(os.path.dirname(filename.replace(self.source, self.target)), exist_ok=True)
            cv2.imwrite(filename.replace(self.source, self.target), log)

