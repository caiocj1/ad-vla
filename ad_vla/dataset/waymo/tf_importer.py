"""Helpers for importing TensorFlow alongside PyTorch without CUDA warnings.

TF is only needed for the Waymo dataset (tf.io.decode_image, protobuf parsing).
When TF loads its shared libraries, it tries to register CUDA plugins (cuDNN,
cuBLAS) that PyTorch has already registered, producing "already registered"
errors. JAX (pulled in by TF/Waymo deps) also registers XLA computation
placers that conflict.

These C++ messages fire at dlopen() time — before absl::InitializeLog() —
so TF_CPP_MIN_LOG_LEVEL cannot suppress them. The only reliable fix is to
redirect the stderr file descriptor during the TF import.
"""

import logging
import os
import sys

_tf_imported = False


def import_tf_silent():
    """Import TensorFlow while suppressing C++ CUDA registration warnings.

    Safe to call multiple times; the actual import only happens once.
    After import, TF GPU visibility is disabled so it doesn't compete with
    PyTorch for VRAM.
    """
    global _tf_imported
    if _tf_imported or "tensorflow" in sys.modules:
        _tf_imported = True
        return

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        import tensorflow as tf  # noqa: F401
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)

    # Hide GPUs from TF so it doesn't compete with PyTorch for VRAM
    try:
        visible_devices = tf.config.list_physical_devices("GPU")
        if visible_devices:
            tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    _tf_imported = True
