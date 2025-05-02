from pathlib import Path
from typing import Literal

from soundscapy.sspylogging import get_logger

logger = get_logger()


def ensure_path_type(filepath: str | Path) -> Path:
    if isinstance(filepath, str):
        return Path(filepath)
    if isinstance(filepath, Path):
        return filepath
    msg = "`filepath` must be either a valid path str or Path object."
    raise TypeError(msg)


def ensure_input_path(filepath: str | Path) -> Path:
    filepath = ensure_path_type(filepath)
    if filepath.exists():
        return filepath
    msg = f"{filepath.as_posix()} does not exist."
    raise Warning(msg)


def ensure_output_filepath_exists(
    filepath: str | Path, *, create_missing: bool = True
) -> Path | Literal[False]:
    filepath = ensure_path_type(filepath)
    if not filepath.exists():
        logger.info("Output file %s does not exist.", filepath)
        if not create_missing:
            return False
        logger.info("Creating new file at %s", filepath.absolute())
        filepath.touch()
    return filepath


def ensure_output_dirpath_exists(
    dirpath: str | Path, *, create_missing: bool = True
) -> Path | Literal[False]:
    dirpath = ensure_path_type(dirpath)
    if not dirpath.exists():
        logger.info("Output directory %s does not exist.", dirpath)
        if not create_missing:
            return False
        logger.info("Creating new directory at %s", dirpath.absolute())
        dirpath.mkdir()
    return dirpath
