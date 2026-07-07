# Built-in Components

This is the catalog of classes that ship with the `echelon3` package — the
values you can put in a `module:` / `type:` triple without any zoo repository or
custom code. Anything importable also works (e.g. `torch.optim.AdamW`,
`torchmetrics.Accuracy`, `albumentations.HorizontalFlip`, `timm.create_model`);
this page covers only what lives inside `echelon3`.

Each `config` block becomes constructor kwargs, so the "key config" columns list
the notable arguments — see the source for the full signature.

!!! note "Optional extras"
    A few components need extra dependencies:
    `pip install echelon3[sam]` (SAM optimizer, MosaicML),
    `echelon3[smp]` (Lovász loss and the PSP neck, segmentation-models-pytorch),
    `echelon3[export]` (ONNX export). Install with the bracketed extra to use them.

## Datasets — `module: echelon3.data.*`

All datasets subclass `basic.BasicDataset`; they read HWC uint8 images with
OpenCV, run `augment` then `preprocess`, and return `(tensor, label)`.

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.data.imageclassifier` | `FoldersHiveImageClassifierDataset` | Class-per-subfolder classifier data (`folder/<class_id>/**/*.png`). |
| `echelon3.data.imageclassifier` | `DataFrameImageClassifierDataset` | Classifier data from a CSV of filename/label columns. |
| `echelon3.data.imageclassifier` | `FolderWithFixedLabelDataset` | All files under one folder, all assigned a single `fixed_label`. |
| `echelon3.data.image2image` | `ImageSegmentationDataset` | Image + mask pairs for semantic segmentation (optional `class_colors`). |
| `echelon3.data.detection` | `DetectionDataset` | Image + annotation pairs; `bboxes_type` ∈ `yolo`/`coco`/`pascal_voc`/`albumentations`. |
| `echelon3.data.multihead` | `MultiBinarySegDataset` | Multi-binary-head segmentation: image + `{head_name: mask}`; missing head → all-ignore. |
| `echelon3.data.multihead` | `MultiBinarySegConcatDataset` | Concatenation of several `MultiBinarySegDataset`s. |
| `echelon3.data.basic` | `MultiPartDataset` | Weighted mixture of sub-datasets by `share`. |
| `echelon3.data.basic` | `ClasswiseDataset` | One sub-dataset per class, indexed as `(class, i)`. |

The `basic` module also exports the base classes (`BasicDataset`,
`FilesDataset`, `AllFilesDataset`, `PerClassFilesDataset`) you subclass when
writing your own dataset.

### File filters — `module: echelon3.data.filters`

Optional `filter:` triple on a `FilesDataset` config that drops files at
collection time.

| `type` | Purpose |
| --- | --- |
| `OrientationFilter` | Keep only `portrait` or `landscape` images (by aspect ratio). |
| `BaseFilter` | Base class for custom filters (`check_file(filename) -> bool`). |

## DataLoaders — `module: echelon3.dataloaders.*`

Drop-in replacements for `torch.utils.data.DataLoader` in the `dataloaders`
section.

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.dataloaders.balance` | `BalancedDataLoader` | Class-balanced batches from a `PerClassFilesDataset` (each batch has equal per-class quotas). |
| `echelon3.dataloaders.classwise` | `ClasswiseDataLoader` | Batches drawn from a `ClasswiseDataset` with a configurable per-class batch sampler. |
| `echelon3.dataloaders.multipart` | `MultiPartDataLoader` | Interleaves parts of a `MultiPartDataset` by their configured shares. |
| `echelon3.dataloaders.varloader` | `VariableDataLoader` | `DataLoader` with a tuple collate for variable-length samples (e.g. detection). |

Matching batch samplers (`BalancedBatchSampler`, `ClasswiseBatchSampler`,
`MultiPartBatchSampler`) live in the same modules.

## Transforms

### Preprocess — `module: echelon3.transforms.preprocess.basic`

`torch.nn.Module`s composed into a `Sequential` and run on the CHW tensor after
augmentation. Every entry needs a `name` in the config.

| `type` | Purpose |
| --- | --- |
| `To01` | Cast to float and divide by 255 → `[0, 1]`. |
| `Normalize` | Subtract `mean`, divide by `std` (both scaled by `max_pixel`). |
| `ToCHW` / `ToHWC` | Permute channel order for a batched tensor. |
| `PatchesToCHW` | Permute patch tensors `(B, N, H, W, C) -> (B, N, C, H, W)`. |
| `ToTensor` | `numpy` HWC → CHW tensor. |
| `Resize` | Resize to a fixed `size` with a chosen `interpolation`. |
| `SmallestMaxSize` | Resize so the shortest side equals `max_size`, keeping aspect ratio. |
| `CropToAspectRatio` | Center-crop to a target height/width aspect ratio. |
| `ToGrayscale` | RGB → grayscale (returned as 3 channels). |
| `GrayscaleCLAHE` | Grayscale + optional CLAHE for provider-invariant appearance. |
| `Squeeze` | Drop a dimension. |
| `Decode` | Decode an encoded image tensor with `torchvision.io`. |
| `ScoreFromLogits` | Softmax then pick one class score (useful as export postprocess). |
| `Id` | Identity passthrough. |

`module: echelon3.transforms.preprocess.segmentation` adds `ToSegmentationMask`
(argmax → uint8 mask) and `ToBinarySegmentationMask` (sigmoid + threshold),
typically used as export/runner postprocess.

### Augment — `module: echelon3.transforms.augment.custom`

Custom [albumentations](https://albumentations.ai/) transforms (use alongside
the stock albumentations catalog). `ToTensorV2` is appended automatically.

| `type` | Purpose |
| --- | --- |
| `CropToAspectRatio` / `CropToAspectRatioV1` | Aspect-ratio crop (V1 adds random vertical/horizontal deviation). |
| `CenterCrop512x512` | Fixed 512×512 center crop. |
| `RandomSquareCropAndFill` | Erase random square patches (cutout-style). |
| `FrequencyNoiseAddition` | Add noise in the Fourier domain. |
| `FrequencyFilter` | Low-/high-pass frequency filtering. |
| `Moire` | Synthetic moiré pattern artifact. |
| `AspectPreservingDownscaleUpscale` | Downscale then upscale to simulate resolution loss. |
| `To01` / `From01` | Scale to/from `[0, 1]` inside the albumentations pipeline. |

## Losses — `module: echelon3.losses.*`

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.losses.classification` | `WeightedCrossEntropyLoss` | Cross-entropy with per-class `weight`. |
| `echelon3.losses.classification` | `FixedMarginCrossEntropyLoss` | Cross-entropy with a fixed additive margin on class `idx`. |
| `echelon3.losses.classification` | `WMV_Ur_Loss` | Pairwise ranking loss between positive/negative scores. |
| `echelon3.losses.label_smoothed` | `LabelSmoothedWeightedCrossEntropyLoss` | Weighted CE exposing `label_smoothing`. |
| `echelon3.losses.segmentation` | `CrossEntropy2D` | Per-pixel cross-entropy. |
| `echelon3.losses.segmentation` | `OhemCELoss` | Online hard-example-mining cross-entropy. |
| `echelon3.losses.segmentation` | `SoftmaxFocalLoss` | Focal loss over softmax for segmentation. |
| `echelon3.losses.segmentation` | `RescaledSegLoss` | Wraps another loss, resizing labels to the logits' resolution. |
| `echelon3.losses.boundary` | `MultiHeadBoundaryWithIgnore` | Sobel-based boundary F1 loss for thin structures (multi-head). |
| `echelon3.losses.cldice` | `MultiHeadSoftCLDiceWithIgnore` | Centerline-Dice topology loss for tubular structures (multi-head). |
| `echelon3.losses.multibinary` | `MultiHeadBCEWithIgnore` | Per-head BCE-with-logits, masking `ignore_index`. |
| `echelon3.losses.multibinary` | `MultiHeadLovaszWithIgnore` | Per-head binary Lovász hinge (needs `echelon3[smp]`). |
| `echelon3.losses.aux_heads` | `MultiHeadAuxEdgeBCE` | BCE on auxiliary edge heads. |
| `echelon3.losses.aux_heads` | `MultiHeadAuxCenterlineBCE` | BCE on auxiliary centerline heads (soft or binary targets). |
| `echelon3.losses.aux_heads` | `MultiHeadAuxOrientationCE` | Cross-entropy on auxiliary orientation-bin heads. |
| `echelon3.losses.detection` | `HeatmapBasedDetectionLoss` | Focal + size loss for heatmap detection. |
| `echelon3.losses.detection` | `DetectionFocalLoss` | CornerNet-style penalty-reduced focal loss. |
| `echelon3.losses.detection` | `WidthHeightLoss` | L1 loss on box width/height at positive locations. |

The multi-head losses expect dict-shaped predictions/labels and pair with
`MultiHeadTrainer`.

## Metrics — `module: echelon3.metrics.*`

Torchmetrics-style objects (`update`/`compute`/`reset`).

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.metrics.classification` | `EER` | Equal error rate from the ROC curve. |
| `echelon3.metrics.classification` | `AUC` | ROC AUC. |
| `echelon3.metrics.classification` | `FrrAtFar` | False reject rate at a given false accept rate (`at_far`). |
| `echelon3.metrics.segmentation` | `IoU` | Mean intersection-over-union (via an internal confusion matrix). |
| `echelon3.metrics.segmentation` | `ConfusionMatrix` | Multi-class confusion matrix. |
| `echelon3.metrics.detection` | `mAP` | Mean average precision (wraps torchmetrics `MeanAveragePrecision`). |
| `echelon3.metrics.multibinary` | `MultiHeadBinaryIoU` | Per-head binary IoU + macro mean. |
| `echelon3.metrics.base` | `Metric` | Base class for custom metrics. |

## Networks — `module: echelon3.nets.*`

### Assemblable nets

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.nets.classifier` | `ClassifierNet` | `backbone -> head` classifier; both are nested triples. |
| `echelon3.nets.segmenter` | `Segmenter` | `backbone -> neck -> head`, bilinear-upsampled to input size. |
| `echelon3.nets.segmenter` | `LightSegmenter` | Neckless segmenter for edge deployment. |
| `echelon3.nets.detector` | `HeatmapDetector` | Anchor-free heatmap detector with a decode step at inference. |
| `echelon3.nets.timm_core_transformer` | `CoreTransformer` | A timm ViT with its patch-embed removed (operates on patch tokens). |

### Classifier heads — `module: echelon3.nets.heads.classifier_heads`

| `type` | Purpose |
| --- | --- |
| `DenseClassifierHead` | Dropout + linear over flattened features. |
| `Conv1x1ClassifierHead` | 1×1 conv head with spatial mean pooling. |
| `AgnosticClassifierHead` | Feature-norm-based open-set score. |
| `CrossVitHead` / `CrossVitAgnosticHead` | Heads for dual-branch CrossViT features. |

`module: echelon3.nets.heads.pattern_heads` adds `MaxPatternHead`,
`AveragePatternHead`, `LinearPatternHead` for pattern/patch aggregation.

### Layers — `module: echelon3.nets.layers.cdc`

Central Difference Convolution family used in anti-spoofing backbones: `CDC`,
`CDCMasked`, `Conv2d_Hori_Veri_Cross`, `Conv2d_Diag_Cross`, `C_CDN`, `DC_CDN`.

### Segmentation building blocks — `module: echelon3.nets.segmentation.*`

| `module` | `type` | Purpose |
| --- | --- | --- |
| `...backbones.timm` | `TimmSegmentationBackbone` | Any timm model in `features_only` mode as a multi-scale backbone. |
| `...backbones.ddrnet` | `DDRNet23SlimBackbone` | DDRNet-23-slim backbone with DAPPM fusion. |
| `...heads.simple_head` | `SimpleHead` / `SimpleAggregatingHead` | SegFormer-style lightweight decode heads. |
| `...heads.light` | `LightHead` | Minimal conv-BN-act head for edge models. |
| `...heads.decode_head` | `BaseDecodeHead` | Base class for MMSeg-style decode heads. |
| `...necks.daspp` | `DASPPneck` / `DASPPneck2` | Dense ASPP context necks. |
| `...necks.psp` | `PSPDecoder` | Pyramid pooling decoder (needs `echelon3[smp]`). |
| `...necks.fpn_like` | `FPNLikeNeck` | Tiny FPN-style multi-scale fusion. |
| `...necks.dlinknet_block` | `DLinkNetFPNLikeNeck` | FPN-like neck with a D-LinkNet dilated central block. |

## Trainers — `module: echelon3.trainers.*`

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.trainers.baseline` | `Trainer` | The default loop: train/validate/keep-best/checkpoint, DDP- and DataParallel-aware. |
| `echelon3.trainers.multihead` | `MultiHeadTrainer` | `Trainer` subclass that handles dict-shaped predictions and labels. |

## Optimizers — `module: echelon3.optimizers.sam`

| `type` | Purpose |
| --- | --- |
| `SAMOptimizer` | Sharpness-Aware Minimization wrapping a base optimizer (requires `echelon3[sam]`). |
| `ASAM` | Adaptive SAM over SGD (no extra dependency). |

The baseline trainer detects `SAMOptimizer` (and `LBFGS`) and drives them with a
closure automatically.

## Weight loaders — `module: echelon3.weightloaders.*`

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.weightloaders.basic` | `WeightsLoader` | Strict `load_state_dict` from a checkpoint. |
| `echelon3.weightloaders.partial` | `PartialWeightsLoader` | Load only name-and-shape-matching tensors (`strip_prefix` optional). |

## Exporters — `module: echelon3.exporters.*`

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.exporters.onnx` | `OnnxExporter` | Export `preprocess -> net -> postprocess` to ONNX (requires `echelon3[export]`). |
| `echelon3.exporters.baseline` | `ModelExporter` | Abstract base wrapping the fused graph. |

## Loggers — `module: echelon3.mlops.tensorboard`

| `type` | Purpose |
| --- | --- |
| `TensorboardLogger` | Scalar loss/metric logging to TensorBoard (the default logger). |
| `ClassifierTensorboardLogger` | Classifier variant. |
| `SegmentationTensorboardLogger` | Segmentation variant. |
| `DetectionTensorboardLogger` | Detection variant that also logs images with boxes/heatmaps. |

## Runners — `module: echelon3.runners.*`

For `echelon3-run` batch inference.

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.runners.images` | `ImagesRunner` | Iterate a folder of images through the model. |
| `echelon3.runners.video` | `VideoRunner` | Iterate a video file frame by frame. |
| `echelon3.runners.segmenter` | `ImagesSegmenter` / `VideoSegmenter` | Overlay a segmentation `colormap` on images/video. |

## Wrappers — `module: echelon3.wrappers.spatial`

Wrap a net (as `export.wrapper`) to adapt input geometry.

| `type` | Purpose |
| --- | --- |
| `UpsampleDownsample` | Run the core at a fixed smallest side, then resize the output back. |
| `PadToMultiplier` | Pad input to a multiple of `m`, then crop the output back. |

`module: echelon3.wrappers.basic` exports `BasicWrapper`, the base class.

## Evaluators — `module: echelon3.evaluators.*`

For `echelon3-evaluate`.

| `module` | `type` | Purpose |
| --- | --- | --- |
| `echelon3.evaluators.classifier` | `ClassifierEvaluator` | Evaluate a classifier and save misclassified samples to disk. |
| `echelon3.evaluators.basic` | `Evaluator` | Base class for custom evaluators. |

## Next

- [Config Schema](config-schema.md) — how these fit into each section.
- [Extending](../guide/extending.md) — add your own classes the same way.
</content>
