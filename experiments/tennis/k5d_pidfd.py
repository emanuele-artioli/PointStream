"""Linux pidfd bindings for the bounded native-process oracle.

Use Python bindings or exported libc functions when available. The final
backend is restricted to the Linux x86_64 ABI; it never signals by PID and
unsupported runtimes fail closed.
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
        self.raw_backend = False
        if not self.python_backend:
            try:
                self.open_fn = self.libc.pidfd_open
                self.open_fn.argtypes = [ctypes.c_int, ctypes.c_uint]
                self.open_fn.restype = ctypes.c_int
                self.signal_fn = self.libc.pidfd_send_signal
                self.signal_fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                self.signal_fn.restype = ctypes.c_int
            except AttributeError as exc:
                if (platform.machine() != 'x86_64' or ctypes.sizeof(ctypes.c_void_p) != 8
                        or ctypes.sizeof(ctypes.c_long) != 8):
                    raise RuntimeError('unsupported Linux pidfd syscall ABI') from exc
                try:
                    self.syscall_fn = self.libc.syscall
                    self.syscall_fn.argtypes = [ctypes.c_long]
                    self.syscall_fn.restype = ctypes.c_long
                except AttributeError as syscall_exc:
                    raise RuntimeError('libc syscall export unavailable') from syscall_exc
                # Linux v6.8 x86_64 table: pidfd_open=434, pidfd_send_signal=424.
                # https://raw.githubusercontent.com/torvalds/linux/v6.8/arch/x86/entry/syscalls/syscall_64.tbl
                self.raw_backend = True

    def open(self, pid: int) -> int:
        if type(pid) is not int or pid <= 0:
            raise ValueError('positive PID required')
        if self.python_backend:
            return int(getattr(os, 'pidfd_open')(pid))
        result = (self.syscall_fn(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
                  if self.raw_backend else self.open_fn(pid, 0))
        if result < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_open failed')
        return int(result)

    def send(self, fd: int, signum: int) -> None:
        if type(fd) is not int or fd < 0 or type(signum) is not int or signum < 0:
            raise ValueError('invalid pidfd or signal')
        if self.python_backend:
            getattr(signal, 'pidfd_send_signal')(fd, signum)
            return
        result = (self.syscall_fn(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(signum),
                                  ctypes.c_void_p(), ctypes.c_uint(0))
                  if self.raw_backend else self.signal_fn(fd, signum, None, 0))
        if result != 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')

    def self_check(self) -> dict[str, Any]:
        fd = self.open(os.getpid())
        try:
            self.send(fd, 0)
        finally:
            os.close(fd)
        return {'backend': 'python' if self.python_backend else ('linux-x86_64-syscall' if self.raw_backend else 'libc-exports'),
                'abi_source': ('https://raw.githubusercontent.com/torvalds/linux/v6.8/arch/x86/entry/syscalls/syscall_64.tbl'
                               if self.raw_backend else None),
                'libc_version': self.libc_version, 'python_version': platform.python_version(),
                'machine': platform.machine(), 'own_pid_signal_zero': True}
