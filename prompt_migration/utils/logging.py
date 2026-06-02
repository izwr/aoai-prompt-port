from __future__ import annotations

import sys


VERBOSE = True


def set_verbose(value: bool) -> None:
    global VERBOSE
    VERBOSE = value


def progress(message: str) -> None:
    if VERBOSE:
        print(message, file=sys.stderr, flush=True)
