#!/usr/bin/env python3
"""
sigint_helper.py — 测试辅助：以非异步方式启动一个命令（使其 SIGINT 可被 trap），
轮询触发文件出现后向它发送 SIGINT，并记录退出码。

背景：从 shell 用 `&` 启动的后台脚本会被 bash 忽略 SIGINT，无法测试 Ctrl+C 收尾；
通过 Python fork+exec 启动则继承默认信号处置，脚本内的 `trap ... INT` 正常工作。

用法：
  sigint_helper.py [选项] [KEY=VALUE ...] -- CMD [ARGS ...]

选项：
  --cwd DIR       子进程工作目录（默认当前目录）
  --log FILE      子进程 stdout+stderr 重定向到 FILE
  --trigger FILE  该文件出现时向子进程发送 SIGINT（测试用 touch 触发）
  --exitfile FILE 子进程退出码写入该文件；timeout 时写入字符串 "timeout"
  --pidfile FILE  子进程 PID 写入该文件
  --timeout S     超时秒数（默认 60；超时 SIGKILL 子进程并写 "timeout"）
"""

import os
import signal
import sys
import time

# 后台启动的进程会从 shell 继承 SIGINT=SIG_IGN（job control 关闭时 bash 对
# 异步命令置 IGN），这会让子脚本的 `trap INT` 失效。这里强制恢复默认处置，
# 使 fork 出的子进程可正常收到 SIGINT 并触发 trap。
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGQUIT, signal.SIG_DFL)

argv = sys.argv[1:]

def parse_args(argv):
    opts = {"cwd": ".", "log": None, "trigger": None,
            "exitfile": None, "pidfile": None, "timeout": 60.0}
    envs = {}
    cmd = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--cwd", "--log", "--trigger", "--exitfile", "--pidfile", "--timeout"):
            opts[a[2:]] = argv[i + 1]
            i += 2
        elif a == "--":
            cmd = argv[i + 1:]
            break
        elif "=" in a and not a.startswith("--"):
            k, v = a.split("=", 1)
            envs[k] = v
            i += 1
        else:
            raise SystemExit("bad arg: %s" % a)
    if not cmd:
        raise SystemExit("missing -- CMD")
    if opts["timeout"]:
        opts["timeout"] = float(opts["timeout"])
    return opts, envs, cmd

opts, envs, cmd = parse_args(argv)

env = dict(os.environ)
env.update(envs)

pid = os.fork()
if pid == 0:
    # child：新建会话/进程组，超时清理可用 killpg 整组终止，避免孤儿进程
    try:
        os.setsid()
    except OSError:
        pass
    if opts["log"]:
        try:
            fd = os.open(opts["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)
        except OSError:
            pass
    try:
        os.chdir(opts["cwd"])
    except OSError:
        os._exit(126)
    # 兜底：确保子进程 SIGINT 为默认处置（SIG_IGN 会在 exec 后保留）
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError:
        os._exit(127)

if opts["pidfile"]:
    with open(opts["pidfile"], "w") as f:
        f.write(str(pid))

def write_exit(code):
    if opts["exitfile"]:
        with open(opts["exitfile"], "w") as f:
            f.write(str(code))

def cleanup_group():
    """best-effort 清扫：子进程（会话 leader）已退出后，组内残余孙进程（如
    mock h3 的 sleep）仍可能存活到自然结束，这里整组 SIGKILL 避免孤儿。"""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


deadline = time.time() + opts["timeout"]
while True:
    try:
        rpid, st = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        cleanup_group()
        write_exit("gone")
        sys.exit(0)
    if rpid == pid:
        code = os.waitstatus_to_exitcode(st) if hasattr(os, "waitstatus_to_exitcode") else (st >> 8)
        cleanup_group()
        write_exit(code)
        sys.exit(0)
    if opts["trigger"] and os.path.exists(opts["trigger"]):
        break
    if time.time() > deadline:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        write_exit("timeout")
        sys.exit(2)
    time.sleep(0.05)

# 触发文件出现：发送 SIGINT（模拟 Ctrl+C），然后等待子进程退出（带超时）
try:
    os.kill(pid, signal.SIGINT)
except ProcessLookupError:
    pass

wait_deadline = time.time() + 15.0
while True:
    try:
        rpid, st = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        cleanup_group()
        write_exit("gone")
        sys.exit(0)
    if rpid == pid:
        code = os.waitstatus_to_exitcode(st) if hasattr(os, "waitstatus_to_exitcode") else (st >> 8)
        cleanup_group()
        write_exit(code)
        sys.exit(0)
    if time.time() > wait_deadline:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        write_exit("sigkill")
        sys.exit(3)
    time.sleep(0.1)
