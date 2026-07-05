import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from echelon3.exporters.baseline import ModelExporter


class OnnxExporter(ModelExporter):
    def __init__(self, net: nn.Module, target: str,
                 preprocess: nn.Module, postprocess: nn.Module,
                 input_names, output_names, input_shape,
                 opset=18, use_tracing=False, do_constant_folding=True, use_aten_fallback=False, dynamic_axes=None,
                 **kwargs):
        super(OnnxExporter, self).__init__(net=net, target=target, preprocess=preprocess, postprocess=postprocess)

        self.opset = opset
        self.use_tracing = use_tracing
        self.do_constant_folding = do_constant_folding
        self.use_aten_fallback = use_aten_fallback
        self.dynamic_axes = dynamic_axes
        self.input_names = input_names
        self.input_shape = input_shape
        self.output_names = output_names

    def export(self):

        addon_kwargs = {}

        if self.use_aten_fallback:
            addon_kwargs['operator_export_type'] = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK

        if self.dynamic_axes is not None:
            addon_kwargs['dynamic_axes'] = OmegaConf.to_container(self.dynamic_axes)

        self.model_to_export.eval()
        self.model_to_export.cpu()

        if self.use_tracing:
            model = self.model_to_export
        else:
            model = torch.jit.script(self.model_to_export, optimize=True)

        arg = torch.randint(0, 255, tuple(self.input_shape), dtype=torch.uint8)

        self.model_to_export(arg)

        if os.path.dirname(self.target):
            os.makedirs(os.path.dirname(self.target), exist_ok=True)

        torch.onnx.export(model=model, args=(arg,), opset_version=self.opset,
                          input_names=self.input_names, output_names=self.output_names,
                          do_constant_folding=self.do_constant_folding,
                          f=self.target,
                          dynamo=False,
                          **addon_kwargs)
