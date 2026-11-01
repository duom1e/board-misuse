"""启动训练/等待器等长任务：脱离控制台与调用者作业对象，避免 CTRL_CLOSE 与被连带杀死。

关键点（踩过的坑）：
  - 不能同时用 CREATE_NO_WINDOW：它会为控制台程序新建一个隐藏控制台，进程仍然会收到
    window-CLOSE 事件，MKL/Fortran 运行时直接 abort（forrtl error 200）。
  - 只保留 DETACHED_PROCESS（不继承、也不新建控制台）+ CREATE_NEW_PROCESS_GROUP
    + CREATE_BREAKAWAY_FROM_JOB（脱离 harness 的作业对象，进程不被整体回收）。
  - 可执行文件优先用 pythonw.exe（GUI 子系统，永远不分配控制台）。

用法：
  python launch_detached.py resume_corn_fullmask_training.py resume_corn_fullmask_training.log
  python launch_detached.py wait_and_finalize_corn.py corn_waiter.out.log
"""
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def choose_executable():
    candidate = Path(sys.executable).with_name('pythonw.exe')
    return str(candidate) if candidate.exists() else sys.executable


def spawn(flags, script, log_name, executable):
    out = (BASE / log_name).open('a', encoding='utf-8', errors='replace')
    err = (BASE / (Path(log_name).stem + '.err.log')).open('a', encoding='utf-8', errors='replace')
    return subprocess.Popen(
        [executable, str(BASE / script)], cwd=str(BASE), stdout=out, stderr=err,
        stdin=subprocess.DEVNULL, close_fds=True, creationflags=flags)


def main():
    script = sys.argv[1]
    log_name = sys.argv[2] if len(sys.argv) > 2 else Path(script).stem + '.log'
    executable = choose_executable()
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        process = spawn(flags | CREATE_BREAKAWAY_FROM_JOB, script, log_name, executable)
        broke_free = True
    except OSError as exc:
        print(f'breakaway refused ({exc}); retrying without it')
        process = spawn(flags, script, log_name, executable)
        broke_free = False
    print(f'launched pid={process.pid} script={script} executable={executable} breakaway={broke_free} log={log_name}')


if __name__ == '__main__':
    main()
