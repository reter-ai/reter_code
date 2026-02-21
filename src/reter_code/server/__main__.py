"""
RETER Server entry point.

Run with: python -m reter_code.server [options]

::: This is-in-layer Server-Layer.
::: This is-in-component ZeroMQ-Server.
"""

import os
import sys
import faulthandler
faulthandler.enable()

def _early_setup():
    """Set up environment BEFORE importing reter modules."""
    # Parse --project/-p argument early to set RETER_PROJECT_ROOT
    # before any reter modules are imported (which trigger logger creation)
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

# Run early setup BEFORE importing reter_server
_early_setup()

from .reter_server import main

if __name__ == "__main__":
    main()
