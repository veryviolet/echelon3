"""Evaluate must bind before construction and never restore a saved GPU placement."""
import pytest
import torch
from omegaconf import OmegaConf

from echelon3.checkpoint.manager import CheckpointManager
from echelon3.cli.evaluate import evaluate_app
from echelon3.creator import create_evaluator
from echelon3.evaluators.basic import Evaluator
from echelon3.evaluators.pair import PairEvaluator


class PairNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, base, query):
        return (base + query) * self.weight


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, augment=None, preprocess=None):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (torch.ones(1, 2, 2), torch.ones(1, 2, 2)), torch.full((1, 2, 2), 2.)


class Metric:
    def to(self, device):
        self.device = torch.device(device)
        return self

    def reset(self):
        self.count = 0

    def update(self, predicted, target):
        torch.testing.assert_close(predicted, target)
        self.count += len(target)

    def compute(self):
        return self.count


def config(tmp_path, monkeypatch, named=False):
    # Real checkpoint whose storage tags refer to another CUDA GPU, created on CPU.
    # CPU map_location must prevent restoration on that original device.
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, 'location_tag', lambda storage: 'cuda:7')
        CheckpointManager(path=str(tmp_path)).save_checkpoint({
            'model_state_dict': PairNet().state_dict(),
            'optimizer_state_dict': {'unused': torch.ones(3)},
        })
    dataset = {'module': __name__, 'type': 'PairDataset'}
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader', 'config': {'batch_size': 2}}
    return OmegaConf.create({
        'device': 'cpu', 'gpus': [],
        'net': {'module': __name__, 'type': 'PairNet'},
        'data': {'test': {'a': dataset, 'b': dataset} if named else dataset},
        'dataloaders': {'test': {'a': loader, 'b': loader} if named else loader},
        'metrics': [{'metric': {'module': __name__, 'type': 'Metric'}}],
        'evaluator': {'module': 'echelon3.evaluators.pair', 'type': 'PairEvaluator',
                      'metric': 'metric'},
        'target': {'path': str(tmp_path)},
    })


def forbidden(*args, **kwargs):
    pytest.fail('CPU evaluate touched CUDA/autocast')


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('precision', ['auto', 'bf16', 'fp16', 'fp32'])
def test_cpu_evaluate_never_touches_cuda(tmp_path, monkeypatch, capsys, named, precision):
    cfg = config(tmp_path, monkeypatch, named)
    cfg.precision = precision
    # Root CPU must also override a conflicting legacy evaluator config.
    cfg.evaluator.config = {'device': 'cuda:0'}
    for attr in ('is_available', 'set_device', '_lazy_init', 'is_bf16_supported'):
        monkeypatch.setattr(torch.cuda, attr, forbidden)
    monkeypatch.setattr(torch, 'autocast', forbidden)
    evaluate_app(cfg)
    assert capsys.readouterr().out.count('metric: 2') == (2 if named else 1)


@pytest.mark.parametrize('selection', [{'device': 'cuda', 'gpus': [6]}, {'device': 'cuda:6'}])
def test_gpu_selected_before_net_checkpoint_and_amp(tmp_path, monkeypatch, selection):
    import echelon3.cli.evaluate as cli
    cfg = config(tmp_path, monkeypatch)
    del cfg.gpus
    cfg.update(selection)
    events = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda dev: events.append(('bind', str(dev))))
    original_create = cli.create_net

    def create_net(spec):
        assert events == [('bind', 'cuda:6')]
        events.append(('construct', 'cuda:6'))
        net = original_create(spec)
        net.to = lambda dev: events.append(('net', str(dev))) or net
        return net

    monkeypatch.setattr(cli, 'create_net', create_net)
    original_load = torch.load

    def load(*args, **kwargs):
        assert str(kwargs['map_location']) == 'cpu'
        events.append(('checkpoint', 'cpu'))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(cli.runtime, 'setup_fast_matmul', lambda **kwargs: None)

    def supports_bf16():
        assert events[0] == ('bind', 'cuda:6')
        return True

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', supports_bf16)

    class Autocast:
        def __init__(self, device_type, dtype):
            assert device_type == 'cuda' and dtype == torch.bfloat16

        def __enter__(self):
            events.append(('autocast', 'enter'))

        def __exit__(self, *args):
            events.append(('autocast', 'exit'))

    monkeypatch.setattr(torch, 'autocast', Autocast)

    def evaluate(self):
        assert self.device == self.metric.device == torch.device('cuda:6')
        assert events[-1] == ('autocast', 'enter')
        return 2

    monkeypatch.setattr(PairEvaluator, 'evaluate', evaluate)
    evaluate_app(cfg)
    assert ('checkpoint', 'cpu') in events and ('net', 'cuda:6') in events
    assert events[-1] == ('autocast', 'exit')


def test_base_evaluator_infers_model_device_without_cuda_probe(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    for net, expected in [(PairNet(), 'cpu'), (torch.nn.Identity(), 'cpu')]:
        evaluator = Evaluator(net, None, None, None, None)
        assert str(evaluator.device) == expected
    net = torch.nn.Module()
    net.register_buffer('only_buffer', torch.empty(1, device='meta'))
    assert Evaluator(net, None, None, None, None).device.type == 'meta'


def test_legacy_evaluator_closed_constructor_still_works(monkeypatch):
    import echelon3.creator as creator

    class Legacy:
        def __init__(self, net, dataloader, metric, preprocess, postprocess):
            self.net = net

    monkeypatch.setattr(creator, 'get_attr_from_module', lambda *args: Legacy)
    net = PairNet()
    ev = create_evaluator(OmegaConf.create({'module': 'legacy', 'type': 'Legacy'}),
                          net, None, None, None, None, None, device=torch.device('cpu'))
    assert ev.net is net
