from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
import torch

class Runner(ABC):

    source = None
    target = None

    def __init__(self, source, target, device=None):
        self.source = source
        self.target = target
        self.device = torch.device(device) if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_frame(self, frame: np.ndarray, model: nn.Module, preprocess: nn.Module, postprocess: nn.Module):
        src_tensor = torch.from_numpy(frame).unsqueeze(dim=0).to(self.device)
        preprocessed = preprocess(src_tensor)
        output = model(preprocessed)
        postprocessed = postprocess(output)
        output = postprocessed.squeeze(dim=0).detach().cpu().numpy()
        return output

    @abstractmethod
    def log(self, frame: np.ndarray, output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def process(self, model: nn.Module, preprocess: nn.Module, postprocess: nn.Module):
        pass