"""
modules/common.py
Shared utilities untuk semua modul Zaneva ImageKit
"""
import os
import uuid
import tempfile
from pathlib import Path
from functools import wraps
from flask import session, redirect, url_for

MAX_FILES = int(os.environ.get('MAX_FILES', 30))
MAX_FILE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', 20))
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp'}

TMP_BASE = Path(os.environ.get('TMP_DIR') or (Path(tempfile.gettempdir()) / 'zaneva_imagekit'))
TMP_BASE.mkdir(parents=True, exist_ok=True)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def get_work_dir(tool: str) -> Path:
    """Return per-session per-tool work directory."""
    sess_id = session.get('sess_id')
    if not sess_id:
        sess_id = uuid.uuid4().hex
        session['sess_id'] = sess_id
    work = TMP_BASE / sess_id / tool
    (work / 'input').mkdir(parents=True, exist_ok=True)
    (work / 'output').mkdir(parents=True, exist_ok=True)
    return work


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT
