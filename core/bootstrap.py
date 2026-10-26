"""
bootstrap.py – Auto-redirects to the project's .venv Python and loads
Windows DLL directories required by MediaPipe. Import at the very top
of every script before any third-party imports.
"""
import os, sys, subprocess

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (one up from core/)
_venv_py = os.path.join(_base, ".venv", "Scripts", "python.exe")

# Re-launch under the virtual environment if needed
if os.path.exists(_venv_py) and sys.executable.lower() != os.path.abspath(_venv_py).lower():
    sys.exit(subprocess.call([_venv_py] + sys.argv))

# Register Windows DLL directories so MediaPipe C++ DLLs can be found
if hasattr(os, "add_dll_directory"):
    for _p in [
        os.path.join(_base, ".venv", "Lib", "site-packages", "mediapipe", "python"),
        os.path.join(_base, ".venv", "Scripts"),
    ]:
        if os.path.exists(_p):
            try:
                os.add_dll_directory(_p)
            except Exception:
                pass

BASE_DIR = _base  # Expose to importing scripts
