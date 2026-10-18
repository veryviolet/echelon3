"""Dataset pipelines replace independently and agree across all validation entry points."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from echelon3.creator import (
    create_augments, create_preprocesses, create_datasets, resolve_dataset_transforms,
)
from echelon3.checkpoint.manager import CheckpointManager
from echelon3.data.basic import MultiPartDataset


class Scale(torch.nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x.float() * self.factor


class ImageDataset:
    def __init__(self, augment=None, preprocess=None, label_type=None):
        self.augment, self.preprocess = augment, preprocess

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        image = self.augment(image=image)['image']
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image, torch.zeros_like(image, dtype=torch.float32)


class ImageNet(torch.nn.Module):
    seen = []

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, image):
        self.seen.append(image.detach().cpu().clone())
        return image.float() * self.weight


class ImageEvaluator:
    def __init__(self, net, dataloader, metric, preprocess, postprocess):
        self.net, self.loader = net, dataloader
        assert preprocess is dataloader.dataset.preprocess

    def evaluate(self):
        with torch.no_grad():
            return sum(float(self.net(x).mean()) for x, _ in self.loader) / len(self.loader)


class PairImageDataset(ImageDataset):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return (image, image), target


class PairImageNet(ImageNet):
    def forward(self, base, query):
        return super().forward(base)


def _dataset(**extra):
    return {'module': __name__, 'type': 'ImageDataset', **extra}


def _scale(factor):
    return {'scale': {'module': __name__, 'type': 'Scale', 'name': 'scale',
                      'config': {'factor': factor}}}


def _common():
    return OmegaConf.create({'test': {
        'augment': {'flip': {'module': 'albumentations', 'type': 'HorizontalFlip', 'config': {'p': 1}}},
        'preprocess': _scale(2),
    }})


def _build(data, transforms=None):
    transforms = _common() if transforms is None else transforms
    ta, va = create_augments(transforms)
    tp, vp = create_preprocesses(transforms)
    return create_datasets(OmegaConf.create(data), ta, tp, va, vp)


def _raw():
    return torch.arange(12).reshape(2, 2, 3).permute(2, 0, 1)


def test_named_test_overrides_replace_independently_without_mutating_defaults():
    data = {'train': _dataset(), 'test': {
        'common': _dataset(),
        'plain': _dataset(transform={'augment': {}}),
        'scaled': _dataset(transform={'preprocess': _scale(3)}),
        'neither': _dataset(transform={'augment': {}, 'preprocess': {}}),
    }}
    original = OmegaConf.create(data)
    _, sets = _build(data)
    raw = _raw()
    assert torch.equal(sets['common'][0][0], raw.flip(-1) * 2)
    assert torch.equal(sets['plain'][0][0], raw * 2)
    assert torch.equal(sets['scaled'][0][0], raw.flip(-1) * 3)
    assert torch.equal(sets['neither'][0][0], raw)
    assert OmegaConf.create(data) == original


def test_local_augment_replaces_instead_of_appending():
    _, sets = _build({'test': {'vertical': _dataset(transform={'augment': {
        'vertical': {'module': 'albumentations', 'type': 'VerticalFlip', 'config': {'p': 1}},
    }})}})
    assert torch.equal(sets['vertical'][0][0], _raw().flip(-2) * 2)


def test_single_test_and_omitted_common_transforms():
    _, ds = _build({'test': _dataset(transform={'preprocess': _scale(3)})}, OmegaConf.create({}))
    assert torch.equal(ds[0][0], _raw() * 3)


def test_null_chains_clear_common_pipelines():
    _, ds = _build({'test': _dataset(transform={'augment': None, 'preprocess': None})})
    assert ds.preprocess is None
    assert torch.equal(ds[0][0], _raw())


def test_multipart_names_are_optional_and_overrides_do_not_leak():
    cfg = _common()
    _, augment = create_augments(cfg)
    _, preprocess = create_preprocesses(cfg)
    parts = OmegaConf.create([
        _dataset(share=0.5, name='plain', transform={'augment': {}}),
        _dataset(share=0.5),
    ])
    ds = MultiPartDataset(parts, augment, preprocess)
    assert ds.part_names == ['plain', None]
    assert ds.part_share(0) == ds.part_share(1) == 0.5
    assert torch.equal(ds[0, 0][0], _raw() * 2)
    assert torch.equal(ds[1, 0][0], _raw().flip(-1) * 2)


@pytest.mark.parametrize('names', [['duplicate', 'duplicate'], ['', None], [4, None]])
def test_invalid_part_names_fail(names):
    parts = OmegaConf.create([_dataset(share=0.5, name=n) for n in names])
    with pytest.raises(ValueError, match='part names'):
        MultiPartDataset(parts, None, None)


def test_bbox_contract_is_inherited_and_can_be_replaced():
    cfg = _common()
    cfg.test.config = {'bbox_params': {'format': 'pascal_voc', 'label_fields': ['labels']}}
    _, augment = create_augments(cfg)
    local, _ = resolve_dataset_transforms(OmegaConf.create({'transform': {'augment': {}}}), augment, None)
    result = local(image=np.zeros((10, 10, 3), np.uint8), bboxes=[[1, 2, 4, 6]], labels=[7])
    np.testing.assert_allclose(result['bboxes'], [[1, 2, 4, 6]], atol=1e-6)
    assert result['labels'] == [7]
    replaced, _ = resolve_dataset_transforms(OmegaConf.create({'transform': {
        'augment': {}, 'config': {'bbox_params': {'format': 'yolo', 'label_fields': ['labels']}},
    }}), augment, None)
    assert replaced.processors['bboxes'].params.format == 'yolo'
    assert augment.processors['bboxes'].params.format == 'pascal_voc'


def _run_config(tmp_path):
    common = _common().test
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader', 'config': {'batch_size': 2}}
    return OmegaConf.create({
        'device': 'cpu',
        'net': {'module': __name__, 'type': 'ImageNet'},
        'data': {
            'train': {'module': 'echelon3.data.basic', 'type': 'MultiPartDataset', 'config': {'parts': [
                _dataset(share=0.5, name='plain', transform={'augment': {}}),
                _dataset(share=0.5, name='flipped'),
            ]}},
            'test': {'plain': _dataset(transform={'augment': {}}), 'flipped': _dataset()},
        },
        'transform': {'train': common, 'test': common},
        'dataloaders': {
            'train': {'module': 'echelon3.dataloaders.multipart', 'type': 'MultiPartDataLoader',
                      'config': {'batch_size': 2}},
            'test': {'plain': loader, 'flipped': loader},
        },
        'loss': [{'mse': {'module': 'torch.nn', 'type': 'MSELoss'}}],
        'optimizer': {'module': 'torch.optim', 'type': 'SGD', 'config': {'lr': 0.0001}},
        'trainer': {'module': 'echelon3.trainers.baseline', 'type': 'Trainer',
                    'config': {'epochs': 1, 'times_to_validate_per_epoch': 2}},
        'target': {'path': str(tmp_path), 'checkpoints_to_keep': 2},
    })


@pytest.mark.parametrize('command', ['train', 'finetune'])
def test_training_initial_and_intermediate_validation_use_local_transforms(tmp_path, command):
    from echelon3.cli.train import trainer_app
    from echelon3.cli.finetune import finetune_app
    ImageNet.seen = []
    (trainer_app if command == 'train' else finetune_app)(_run_config(tmp_path))
    seen = ImageNet.seen
    raw, flipped = _raw() * 2, _raw().flip(-1) * 2
    # Initial validation: two batches of each set. Training: mixed multipart batches.
    assert all(torch.equal(batch[0], raw) for batch in seen[:2])
    assert all(torch.equal(batch[0], flipped) for batch in seen[2:4])
    assert torch.equal(seen[4][0], raw) and torch.equal(seen[4][1], flipped)
    # Two more validations, with the same pipelines, interleaved with four train batches.
    assert len(seen) == 16
    assert sum(torch.equal(batch[0], flipped) for batch in seen) == 6


def test_evaluate_uses_same_resolver_and_applies_preprocess_once(tmp_path, capsys):
    from echelon3.cli.evaluate import evaluate_app
    cfg = _run_config(tmp_path)
    # Evaluate must not construct unused train-only augmentation components.
    cfg.transform.train.augment = {'missing': {'module': 'not_installed', 'type': 'Transform'}}
    cfg.data.test.plain.transform.preprocess = _scale(3)
    cfg.metrics = [{'mean': {'module': 'torchmetrics', 'type': 'MeanSquaredError'}}]
    cfg.evaluator = {'module': __name__, 'type': 'ImageEvaluator', 'metric': 'mean'}
    CheckpointManager(path=str(tmp_path)).save_checkpoint({'model_state_dict': ImageNet().state_dict()})
    ImageNet.seen = []
    evaluate_app(cfg)
    assert torch.equal(ImageNet.seen[0][0], _raw() * 3)
    assert torch.equal(ImageNet.seen[-1][0], _raw().flip(-1) * 2)
    out = capsys.readouterr().out
    assert 'Validation [plain] mean: 16.5' in out
    assert 'Validation [flipped] mean: 11.0' in out


def test_pair_evaluator_receives_preprocessed_pairs_without_reapplying(tmp_path):
    from echelon3.cli.evaluate import evaluate_app
    cfg = _run_config(tmp_path)
    cfg.net.type = 'PairImageNet'
    for ds in cfg.data.test.values():
        ds.type = 'PairImageDataset'
    cfg.data.test.plain.transform.preprocess = _scale(3)
    cfg.metrics = [{'mse': {'module': 'torchmetrics', 'type': 'MeanSquaredError'}}]
    cfg.evaluator = {'module': 'echelon3.evaluators.pair', 'type': 'PairEvaluator', 'metric': 'mse'}
    CheckpointManager(path=str(tmp_path)).save_checkpoint({'model_state_dict': PairImageNet().state_dict()})
    ImageNet.seen = []
    evaluate_app(cfg)
    assert torch.equal(ImageNet.seen[0][0], _raw() * 3)
    assert torch.equal(ImageNet.seen[-1][0], _raw().flip(-1) * 2)
