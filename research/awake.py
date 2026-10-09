"""Keeps the PC from going to sleep while an answer is being made.

A question that runs for minutes outlives Windows' idle timer: one laptop went into Modern
Standby six minutes into a parallel question, the network dropped, and the answer never came.
While any `awake()` block is open, Windows is told the system is busy (a power request, the
same thing a video player makes). The screen may still turn off. Closing the lid still sleeps.
Does nothing on other systems.
"""
from __future__ import annotations

import ctypes
import sys
import threading
from contextlib import contextmanager

_lock = threading.Lock()
_users = 0
_handle = None

POWER_REQUEST_SYSTEM_REQUIRED = 1     # don't idle into sleep
POWER_REQUEST_EXECUTION_REQUIRED = 3  # keep running in Modern Standby (screen off)


class _ReasonContext(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("Flags", ctypes.c_ulong), ("Reason", ctypes.c_wchar_p)]


def _request():
    k = ctypes.windll.kernel32
    k.PowerCreateRequest.restype = ctypes.c_void_p
    k.PowerSetRequest.argtypes = k.PowerClearRequest.argtypes = (ctypes.c_void_p, ctypes.c_int)
    k.CloseHandle.argtypes = (ctypes.c_void_p,)
    reason = _ReasonContext(0, 1, "research: answering a question")  # 0x1 = simple string
    handle = k.PowerCreateRequest(ctypes.byref(reason))
    if not handle or handle == ctypes.c_void_p(-1).value:
        return None
    for kind in (POWER_REQUEST_SYSTEM_REQUIRED, POWER_REQUEST_EXECUTION_REQUIRED):
        k.PowerSetRequest(handle, kind)
    return handle


def _release(handle) -> None:
    k = ctypes.windll.kernel32
    for kind in (POWER_REQUEST_SYSTEM_REQUIRED, POWER_REQUEST_EXECUTION_REQUIRED):
        k.PowerClearRequest(handle, kind)
    k.CloseHandle(handle)


@contextmanager
def awake():
    """Hold the PC awake for the duration of the block (nested and concurrent blocks share one request)."""
    global _users, _handle
    if sys.platform != "win32":
        yield
        return
    with _lock:
        _users += 1
        if _users == 1:
            try:
                _handle = _request()
            except (OSError, AttributeError):  # never fail a question over this
                _handle = None
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0 and _handle:
                try:
                    _release(_handle)
                except OSError:
                    pass
                _handle = None
