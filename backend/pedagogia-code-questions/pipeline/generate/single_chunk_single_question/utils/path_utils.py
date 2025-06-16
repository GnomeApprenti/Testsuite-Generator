import os
import tempfile
import shutil
import zipfile
import tarfile
from pathlib import Path


def _extract_zip(path: Path) -> str:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


def _extract_tgz(path: Path) -> str:
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(path, 'r:gz') as tar_ref:
        tar_ref.extractall(temp_dir)
    return temp_dir


def prepare_input_path(path: str) -> str:
    """Handles different input types: directories, files, zip or tgz archives."""
    path_obj = Path(path)
    if path_obj.is_dir():
        return str(path_obj)

    if path_obj.suffix == '.zip':
        return _extract_zip(path_obj)
    elif path_obj.suffix in {'.tgz', '.tar.gz'}:
        return _extract_tgz(path_obj)
    elif path_obj.is_file():
        # Copy single file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        shutil.copy(path_obj, temp_dir)
        return temp_dir
    else:
        raise ValueError(f"Unsupported path type or extension: {path}")
