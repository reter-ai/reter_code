"""
Lightweight server entry point.

Sets RETER_PROJECT_ROOT from --project arg BEFORE heavy imports,
matching the behavior of `python -m reter_code.server`.
"""

import os
import sys


def _early_setup():
    """Parse --project early to set RETER_PROJECT_ROOT before imports."""
    project_path = None
    for i, arg in enumerate(sys.argv):
        if arg in ('--project', '-p') and i + 1 < len(sys.argv):
            project_path = os.path.abspath(sys.argv[i + 1])
            break
        elif arg.startswith('--project='):
            project_path = os.path.abspath(arg.split('=', 1)[1])
            break
    if not project_path:
        project_path = os.getcwd()
    os.environ['RETER_PROJECT_ROOT'] = project_path


def _check_windows_prerequisites():
    """Check Windows-specific prerequisites and print helpful messages."""
    if sys.platform != 'win32':
        return
    import ctypes
    for dll in ('msvcp140.dll', 'vcruntime140.dll'):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            print(
                f"\n[ERROR] {dll} not found — Microsoft Visual C++ Redistributable is required.\n"
                "\n"
                "Install it with Chocolatey:\n"
                "\n"
                "    choco install vcredist140\n"
                "\n"
                "Or download manually from:\n"
                "\n"
                "    https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "\n"
                "Then restart your terminal and try again.\n",
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    """Server entry point with early env setup."""
    _early_setup()
    _check_windows_prerequisites()
    from .server.reter_server import main as server_main
    server_main()
