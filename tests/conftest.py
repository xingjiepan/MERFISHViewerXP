import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

_session_exitstatus: int | None = None


def pytest_sessionfinish(session, exitstatus):
    global _session_exitstatus
    _session_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Exit before PyQt6's own interpreter-exit cleanup runs.

    After a session that created and closed many napari viewers, PyQt6's
    atexit handler (QtCore ``cleanup_on_exit``) intermittently dereferences a
    stale sip wrapper and segfaults -- after every test has passed and the
    report has been printed -- turning a green run into exit code 139.
    PyQt6's sip no longer offers ``setdestroyonexit(False)``, so skip the
    handler by exiting with pytest's own status once reporting is done.
    """
    if _session_exitstatus is None or "PyQt6.QtCore" not in sys.modules:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_session_exitstatus)
