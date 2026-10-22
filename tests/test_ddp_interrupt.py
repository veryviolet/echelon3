"""A single terminal-style SIGINT must stop and reap persistent loader workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def _session_processes(session_id):
    """Include multiprocessing's resource tracker as well as ranks and workers."""
    found = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rpartition(')')[2].split()
            if int(fields[3]) == session_id:
                found.append(int(path.parent.name))
        except (OSError, IndexError, ValueError):
            continue  # A process may exit while /proc is being traversed.
    return found


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process group regression')
@pytest.mark.parametrize('command', ['train', 'finetune'])
@pytest.mark.parametrize('backend', ['gloo', 'nccl'])
def test_single_sigint_stops_persistent_workers(tmp_path, command, backend):
    if backend == 'nccl' and not os.environ.get('ECHELON3_TEST_DDP_GPUS'):
        pytest.skip('Set ECHELON3_TEST_DDP_GPUS to two available physical GPU indices')
    if backend == 'nccl':
        gpus = os.environ['ECHELON3_TEST_DDP_GPUS'].split(',')
        assert len(gpus) == 2 and len(set(gpus)) == 2, 'Select exactly two distinct GPUs'
    script = Path(__file__).with_name('_ddp_interrupt_probe.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    env['PYTHONPATH'] = os.pathsep.join([
        str(script.parent), str(script.parent.parent / 'src'), env.get('PYTHONPATH', ''),
    ])
    if backend == 'gloo':
        env['CUDA_VISIBLE_DEVICES'] = ''
    log = tmp_path / 'output.log'
    with log.open('w') as output:
        process = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), command, backend],
            cwd=tmp_path, env=env, stdout=output, stderr=output, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 90
            while len(list(tmp_path.glob('ready-*.json'))) < 2:
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            ready = [json.loads(path.read_text()) for path in tmp_path.glob('ready-*.json')]
            assert all(report['backend'] == backend for report in ready)
            assert all(report['pin_memory'] == (backend == 'nccl') for report in ready)
            # Exactly one signal to the foreground process group, as with Ctrl-C.
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=45)
            text = log.read_text()
            assert process.returncode == 130, text
            reports = list(tmp_path.glob('closed-*.json'))
            assert len(reports) == 2, text
            for path in reports:
                workers = json.loads(path.read_text())
                assert len(workers) == 4, text
                assert all(not w['alive'] and w['exitcode'] == 0 for w in workers), (workers, text)
            pids = [process.pid] + [pid for r in ready for pid in [r['rank_pid'], *r['worker_pids']]]
            assert not [pid for pid in pids if Path(f'/proc/{pid}').exists()], text
            deadline = time.monotonic() + 5
            while _session_processes(process.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not _session_processes(process.pid), text
            for failure in ('Fatal Python error', 'terminate called', 'Segmentation fault',
                            'Traceback', 'leaked semaphore'):
                assert failure not in text, text
        finally:
            # Only our own session/process group; never touch other jobs.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
