"""Small file and process helpers shared by data, training, and evaluation.

Importing this module does not initialize Paddle, Pillow, or an OCR runtime.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    """流式计算文件 SHA-256，供输入锁与产物身份核对。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, data):
    """先写临时文件再替换，避免训练状态半写入。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def acquire_training_lock(runtime):
    """阻止同一运行目录被训练或验收进程同时写入。"""
    handle = (Path(runtime) / "runs/.training.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("another project training process owns the GPU training lock")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
