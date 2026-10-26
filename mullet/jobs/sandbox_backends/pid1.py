"""PID 1 of every container the docker and podman backends start.

The container backend passes this file's source to the image's Python as
``python3 -I -S -c <source> merged|split <command...>`` under ``env -i``. It is
never imported on the host, and refuses to run anywhere but as PID 1: its
last act, ``kill(-1, SIGKILL)``, would hit every process the user owns.

1. It reads the sandbox's environment from stdin: one JSON object on one line,
   and refuses (exit 125) names and values that execve can't take.
2. It makes itself non-dumpable, so a process in the container (same uid, no
   capabilities) can't ptrace it or read its memory or /proc/1/environ.
3. It starts the command with stdin on /dev/null and that environment; a
   command that can't be found exits 127, one that can't be run 126.
4. It watches the rest of stdin, the lifeline: Mullet holds the pipe's write
   end, so EOF means Mullet or the docker client is gone. It then kills every
   process in the container and exits, which ends the container.
5. It reaps zombies, forwards TERM, INT, HUP, QUIT, USR1 and USR2 to the
   command's processes, and exits with the command's status when it ends.

The kernel delivers a signal from inside a PID namespace to its init only when
init has a handler for it, and never SIGKILL or SIGSTOP. The handlers here
only forward or wake the loop, so nothing in the container can stop the watch.
"""

import ctypes
import errno
import json
import os
import select
import signal
import sys

FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2)
PR_SET_DUMPABLE = 4
MAX_PAYLOAD = 1 << 20


def die(message, code=125):
    os.write(2, ("mullet: " + message + "\n").encode())
    os._exit(code)


def read_environment(fd=0):
    data = b""
    while b"\n" not in data:
        chunk = os.read(fd, 65536)
        if not chunk or len(data) > MAX_PAYLOAD:
            die("the sandbox environment did not arrive")
        data += chunk
    try:
        env = json.loads(data.split(b"\n", 1)[0])
    except ValueError:
        env = None
    if not isinstance(env, dict):
        die("the sandbox environment did not arrive")
    for name, value in env.items():
        if not isinstance(name, str) or not isinstance(value, str):
            die("the sandbox environment must map names to strings")
        if not name or "=" in name or "\0" in name:
            die("refusing the sandbox environment: %s is not a variable name" % ascii(name))
        if "\0" in value:
            die("refusing the sandbox environment: the value of %s holds a NUL byte" % ascii(name))
    return env


def forward(signum, frame):
    try:
        os.kill(-1, signum)
    except OSError:
        pass


def start(command, env, merged, wakeup):
    pid = os.fork()
    if pid:
        return pid
    # The child never returns from here, whatever fails: main()'s cleanup is PID 1's alone.
    code = 126
    try:
        try:
            for sig in (*FORWARDED, signal.SIGCHLD, signal.SIGPIPE, signal.SIGXFSZ):
                signal.signal(sig, signal.SIG_DFL)
            signal.set_wakeup_fd(-1)
            for fd in wakeup:
                os.close(fd)
            null = os.open("/dev/null", os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
            if merged:
                os.dup2(1, 2)
            signal.pthread_sigmask(signal.SIG_SETMASK, [])
            os.execvpe(command[0], command, env)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                code = 127
            reason = exc.strerror or str(exc)
        except BaseException as exc:
            reason = "can't be run: %s" % (str(exc) or type(exc).__name__)
        os.write(2, ("mullet: %s: %s\n" % (command[0], reason)).encode(errors="replace"))
    finally:
        os._exit(code)


def status_code(status):
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def watch(child, wakeup_read):
    while True:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if pid == 0:
                break
            if pid == child:
                return status_code(status)
        ready, _, _ = select.select([0, wakeup_read], [], [])
        if wakeup_read in ready:
            try:
                os.read(wakeup_read, 4096)
            except BlockingIOError:
                pass
        if 0 in ready:
            try:
                data = os.read(0, 65536)
            except OSError:
                data = b""
            if not data:
                return 137


def main():
    if os.getpid() != 1:
        die("pid1 runs only as a sandbox container's PID 1")
    # Python's own SIGINT handler would make PID 1 signalable from inside.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if len(sys.argv) < 3 or sys.argv[1] not in ("merged", "split"):
        die("usage: pid1 merged|split command...")
    merged, command = sys.argv[1] == "merged", sys.argv[2:]
    try:
        if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl failed")
    except (OSError, AttributeError) as exc:
        die("can't make the sandbox's PID 1 non-dumpable: %s" % exc)
    env = read_environment()
    wakeup = os.pipe()
    for fd in wakeup:
        os.set_blocking(fd, False)
    code = 125
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, (*FORWARDED, signal.SIGCHLD))
        signal.set_wakeup_fd(wakeup[1], warn_on_full_buffer=False)
        signal.signal(signal.SIGCHLD, lambda signum, frame: None)
        for sig in FORWARDED:
            signal.signal(sig, forward)
        child = start(command, env, merged, wakeup)
        del env
        signal.pthread_sigmask(signal.SIG_SETMASK, [])
        code = watch(child, wakeup[0])
    finally:
        try:
            os.kill(-1, signal.SIGKILL)
        except OSError:
            pass
        os._exit(code)


if __name__ == "__main__":
    main()
