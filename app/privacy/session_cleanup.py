import os
import shutil
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def temporary_video_scope(file_path: str, allow_persist: bool = False) -> Generator[str, None, None]:
    """
    Context manager that guarantees temporary video files are securely removed
    after processing unless raw persistence is explicitly opted in.
    """
    try:
        yield file_path
    finally:
        if not allow_persist and os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Privacy enforcement: Cleaned up temp video {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.info(f"Privacy enforcement: Cleaned up temp directory {file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary video file {file_path}: {str(e)}")
        elif allow_persist:
            logger.info(f"Student opted in to video storage. Persisting file at {file_path}")
