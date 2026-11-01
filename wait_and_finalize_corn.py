"""Wait (console-free) until the corn training writes status=completed, then run the finalizer steps.

Run with pythonw.exe so a console CLOSE event from the harness cannot abort it
(forrtl aborts with 'window-CLOSE event' when the console is closed mid-training).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATUS = BASE / 'food_fullmask_corn_multiscale45_v1' / 'status.json'
LOG = BASE / 'corn_waiter.log'
DEADLINE_HOURS = 6


def log(message):
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def stage():
    try:
        return json.loads(STATUS.read_text(encoding='utf-8')).get('stage')
    except Exception:  # noqa: BLE001
        return None


def main():
    # 鍙厑璁镐竴涓瓑寰呭櫒杩愯锛氭姠鍗犻攣澶辫触灏遍€€鍑猴紝閬垮厤璁粌瀹屾垚鍚庨噸澶嶈窇璇勪及
    lock = BASE / 'corn_waiter.lock'
    try:
        handle = open(lock, 'x', encoding='utf-8')
    except FileExistsError:
        log('another waiter already holds the lock, exiting')
        return 0
    handle.write(str(os.getpid()))
    handle.close()
    log(f'waiter started pid={os.getpid()}')
    deadline = time.monotonic() + DEADLINE_HOURS * 3600
    while time.monotonic() < deadline:
        current = stage()
        if current == 'completed':
            log('training completed, running finalizer')
            result = subprocess.run([sys.executable, str(BASE / 'finalize_corn_fullmask_training.py')],
                                    cwd=BASE, capture_output=True, text=True)
            log(f'finalizer returncode {result.returncode}')
            log('stdout tail: ' + (result.stdout or '')[-1500:])
            log('stderr tail: ' + (result.stderr or '')[-1500:])
            return result.returncode
        time.sleep(30)
    log('timed out waiting for completion')
    return 2


if __name__ == '__main__':
    sys.exit(main())

