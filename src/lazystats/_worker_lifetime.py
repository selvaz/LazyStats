"""Native Windows kill-on-close jobs, with an independent watchdog fallback."""
from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from typing import Any


class WindowsJob:
    def __init__(self, dll: Any, handle: Any) -> None:
        self._dll = dll
        self._handle = handle

    def close(self) -> None:
        if self._handle:
            self._dll.CloseHandle(self._handle)
            self._handle = None


def _windows_api() -> Any:
    if sys.platform != "win32":
        raise OSError("the Windows API is only available on Windows")
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    dll.OpenProcess.restype = wintypes.HANDLE
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    dll.WaitForSingleObject.restype = wintypes.DWORD
    dll.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    dll.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    dll.CreateJobObjectW.restype = wintypes.HANDLE
    dll.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD]
    dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    return dll


def kill_on_close_job(pid: int) -> WindowsJob | None:
    if sys.platform != "win32":
        return None

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    dll = _windows_api()
    handle = dll.CreateJobObjectW(None, None)  # non-inheritable; held only by parent
    if not handle:
        return None
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    process = dll.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
    try:
        if (not process or not dll.SetInformationJobObject(
                handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not dll.AssignProcessToJobObject(handle, process)):
            dll.CloseHandle(handle)
            return None
        return WindowsJob(dll, handle)
    finally:
        if process:
            dll.CloseHandle(process)


def _posix_alive(pid: int, proc_root: str = "/proc") -> bool:
    """Alive and not a zombie: a killed but unreaped parent still answers
    ``os.kill(pid, 0)``, so read its state where ``/proc`` exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"{proc_root}/{pid}/stat", encoding="ascii", errors="replace") as stat:
            state = stat.read().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return True
    return state not in ("Z", "X")


def watch_parent(parent_pid: int, worker_pid: int) -> None:
    """Independent process: kill even a worker whose fit holds the GIL.

    Exit when the worker exits normally or is reaped. Windows handles pin
    process identity, avoiding PID reuse during the watch.
    """
    if sys.platform == "win32":
        dll = _windows_api()
        parent = dll.OpenProcess(0x00100000, False, parent_pid)
        worker = dll.OpenProcess(0x00100000 | 0x0001, False, worker_pid)
        try:
            if not worker:
                return
            while dll.WaitForSingleObject(worker, 0) == 258:  # WAIT_TIMEOUT
                if not parent or dll.WaitForSingleObject(parent, 0) != 258:
                    dll.TerminateProcess(worker, 0)
                    return
                time.sleep(0.1)
        finally:
            if parent:
                dll.CloseHandle(parent)
            if worker:
                dll.CloseHandle(worker)
    else:
        alive = _posix_alive

        while alive(worker_pid):
            if not alive(parent_pid):
                import signal

                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            time.sleep(0.1)
