"""Native 'choose a folder' dialog, usable from any thread.

Backends in order: macOS ``osascript`` (a Finder sheet, no extra packages),
tkinter where it is installed, ``zenity`` on Linux desktops. Returns the
chosen path or ``None`` when the user cancels or no backend is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "dataanalyzer"
LAST_ROOT_FILE = CONFIG_DIR / "last_root"


def remembered_root() -> Path | None:
    try:
        p = Path(LAST_ROOT_FILE.read_text().strip())
        return p if p.is_dir() else None
    except OSError:
        return None


def remember_root(path: str | Path) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LAST_ROOT_FILE.write_text(str(Path(path).resolve()))
    except OSError:
        pass


LAST_ERROR: str | None = None     # why the last dialog produced nothing, when it was not simply cancelled


def _osascript(initial: Path | None, title: str) -> str | None:
    global LAST_ERROR
    LAST_ERROR = None
    loc = f' default location (POSIX file "{initial}")' if initial else ""
    # "activate" brings the dialog to the front; started from a terminal it otherwise opens behind other windows
    script = f'activate\nset p to choose folder with prompt "{title}"{loc}\nPOSIX path of p'
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    except OSError as e:
        LAST_ERROR = str(e)
        return None
    if out.returncode != 0:
        err = out.stderr.strip()
        if "-128" not in err and "User canceled" not in err:       # anything but the Cancel button
            LAST_ERROR = err or f"osascript exited with status {out.returncode}"
        return None
    path = out.stdout.strip()
    return path.rstrip("/") or None


def _tkinter(initial: Path | None, title: str) -> str | None:
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(title=title, initialdir=str(initial) if initial else None, mustexist=True)
    finally:
        root.destroy()
    return chosen or None


def _zenity(initial: Path | None, title: str) -> str | None:
    if not shutil.which("zenity"):
        return None
    cmd = ["zenity", "--file-selection", "--directory", f"--title={title}"]
    if initial:
        cmd.append(f"--filename={initial}/")
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout.strip() or None if out.returncode == 0 else None


def available() -> bool:
    if sys.platform == "darwin" and shutil.which("osascript"):
        return True
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        pass
    return bool(shutil.which("zenity"))


def pick_folder(initial: str | Path | None = None, title: str = "Choose a recording or a folder of recordings") -> str | None:
    initial = Path(initial) if initial else remembered_root()
    if initial is not None and not initial.is_dir():
        initial = None
    backends = [_osascript] if sys.platform == "darwin" else []
    backends += [_tkinter, _zenity]
    for backend in backends:
        try:
            chosen = backend(initial, title)
        except Exception:
            chosen = None
        if chosen:
            remember_root(chosen)
            return chosen
        if backend is _osascript and sys.platform == "darwin" and shutil.which("osascript"):
            return None  # the dialog was shown and cancelled; don't fall through to another one
    return None
