"""Fatal rank exceptions must terminate the job even when diagnostics/cleanup fail."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from .test_ddp_interrupt import _session_processes


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process/signal regression')
@pytest.mark.parametrize('command,scenario', [
    ('train', 'normal'), ('finetune', 'normal'), ('train', 'gil'), ('finetune', 'stderr'),
    ('train', 'error_file'),
])
def test_enospc_exits_all_ranks_without_touching_checkpoints(tmp_path, command, scenario):
    script = Path(__file__).with_name('_ddp_failure_probe.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    env['PYTHONPATH'] = os.pathsep.join([
        str(script.parent), str(script.parent.parent / 'src'), env.get('PYTHONPATH', ''),
    ])
    log = tmp_path / 'output.log'
    with log.open('w') as output:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path), command, scenario],
                                   cwd=tmp_path, env=env, stdout=output, stderr=output,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while len(list(tmp_path.glob('ready-*.json'))) < 2:
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            ready = [json.loads(path.read_text()) for path in tmp_path.glob('ready-*.json')]
            initial = next((tmp_path / 'target').glob('checkpoint-*.tar'))
            (tmp_path / 'target' / 'last.tar').write_bytes(initial.read_bytes())
            checkpoints = {p.name: p.read_bytes() for p in (tmp_path / 'target').glob('*.tar')}
            assert checkpoints
            (tmp_path / 'inject').touch()
            # No Ctrl-C and no external stop signal: only the injected logger error.
            process.wait(timeout=30 if scenario == 'normal' else 12)
            text = log.read_text()
            assert process.returncode != 0, text
            assert (tmp_path / 'failed.json').exists(), text
            assert 'No space left on device' in text, text
            assert checkpoints == {p.name: p.read_bytes() for p in (tmp_path / 'target').glob('*.tar')}
            # Even when stderr fails, elastic must retain the original exception.
            errors = list((tmp_path / 'elastic').rglob('error.json'))
            if scenario != 'error_file':
                assert any('injected TensorBoard' in p.read_text() for p in errors), text
            deadline = time.monotonic() + 5
            while _session_processes(process.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not _session_processes(process.pid), text
            pids = [pid for r in ready for pid in [r['rank_pid'], *r['worker_pids']]]
            assert not [pid for pid in pids if Path(f'/proc/{pid}').exists()], text
            failure = json.loads((tmp_path / 'failed.json').read_text())
            (tmp_path / 'result.json').write_text(json.dumps({
                'scenario': scenario, 'command': command,
                'seconds_after_failure': time.monotonic() - failure['time'],
                'exitcode': process.returncode, 'checkpoints_unchanged': True,
                'remaining_processes': [],
            }))
        finally:
            if process.poll() is None or _session_processes(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
