"""Real 28-step profiling window with persistent DataLoader workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from .test_ddp_interrupt import _session_processes


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process cleanup check')
@pytest.mark.parametrize('command', ['train', 'finetune'])
def test_profile_stop_ddp(tmp_path, command):
    script = Path(__file__).with_name('_profile_stop_probe.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    native_gpus = env.get('ECHELON3_PROFILE_TEST_GPUS')
    world_size = len(native_gpus.split(',')) if native_gpus else 4
    if not env.get('ECHELON3_PROFILE_TEST_GPUS'):
        env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = os.pathsep.join([
        str(script.parent), str(script.parent.parent / 'src'), env.get('PYTHONPATH', '')])
    log = tmp_path / 'output.log'
    with log.open('w') as output:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path), command],
                                   env=env, stdout=output, stderr=output, start_new_session=True)
        try:
            process.wait(timeout=180)
            text = log.read_text()
            assert process.returncode == 0, text
            assert '35% (28/80)' in text, text
            assert '100% (80/80)' not in text, text
            for rank in range(world_size):
                report = json.loads((tmp_path / f'closed-{rank}.json').read_text())
                assert report['steps'] == 28 and report['completed_epochs'] == 0, report
                assert report['scheduler_epoch'] == 0, report
                assert len(report['workers']) == 2, report
                assert all(not w['alive'] and w['exitcode'] == 0 for w in report['workers']), report
            assert not (tmp_path / 'target' / 'last.tar').exists()
            assert list((tmp_path / 'trace').glob('*.pt.trace.json')), text
            for error in ('terminate called', 'Traceback', 'leaked semaphore', 'Fatal Python error'):
                assert error not in text, text
            deadline = time.monotonic() + 5
            while _session_processes(process.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not _session_processes(process.pid), text
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
