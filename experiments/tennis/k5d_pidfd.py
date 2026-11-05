"""Linux pidfd bindings for the bounded native-process oracle.

Use Python's bindings when both are present; otherwise use only exported libc
functions. Never fall back to PID-based signals or architecture-specific syscall
numbers. Unsupported runtimes fail closed.
"""
from __future__ import annotations

import ctypes
import os
import platform
import signal
import sys
from typing import Any


class Pidfd:
    def __init__(self) -> None:
        if sys.platform != 'linux':
            raise RuntimeError('Linux pidfd required')
        self.libc = ctypes.CDLL(None, use_errno=True)
        try:
            version = self.libc.gnu_get_libc_version
            version.argtypes = []
            version.restype = ctypes.c_char_p
            raw_version = version()
            if raw_version is None:
                raise RuntimeError('libc version unavailable')
            self.libc_version = raw_version.decode('ascii')
        except AttributeError as exc:
            raise RuntimeError('GNU libc version unavailable') from exc
        self.python_backend = callable(getattr(os, 'pidfd_open', None)) and callable(getattr(signal, 'pidfd_send_signal', None))
        if not self.python_backend:
            try:
                self.open_fn = self.libc.pidfd_open
                self.open_fn.argtypes = [ctypes.c_int, ctypes.c_uint]
                self.open_fn.restype = ctypes.c_int
                self.signal_fn = self.libc.pidfd_send_signal
                self.signal_fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                self.signal_fn.restype = ctypes.c_int
            except AttributeError as exc:
                raise RuntimeError('libc pidfd exports unavailable') from exc

    def open(self, pid: int) -> int:
        if type(pid) is not int or pid <= 0:
            raise ValueError('positive PID required')
        if self.python_backend:
            return int(getattr(os, 'pidfd_open')(pid))
        result = self.open_fn(pid, 0)
        if result < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_open failed')
        return int(result)

    def send(self, fd: int, signum: int) -> None:
        if type(fd) is not int or fd < 0 or type(signum) is not int or signum < 0:
            raise ValueError('invalid pidfd or signal')
        if self.python_backend:
            getattr(signal, 'pidfd_send_signal')(fd, signum)
            return
        if self.signal_fn(fd, signum, None, 0) != 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')

    def self_check(self) -> dict[str, Any]:
        fd = self.open(os.getpid())
        try:
            self.send(fd, 0)
        finally:
            os.close(fd)
        return {'backend': 'python' if self.python_backend else 'libc-exports',
                'libc_version': self.libc_version, 'python_version': platform.python_version(),
                'machine': platform.machine(), 'own_pid_signal_zero': True}
