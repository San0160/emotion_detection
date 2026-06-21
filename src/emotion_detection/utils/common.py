from pathlib import Path
import yaml
import sys
import os

from box import ConfigBox
from ensure import ensure_annotations

from emotion_detection.logging import logger
from emotion_detection.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox object.

    Args:
        path_to_yaml (Path): Path to YAML file

    Returns:
        ConfigBox: Parsed YAML content
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:

            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("YAML file is empty")

            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")

            return ConfigBox(content)

    except Exception as e:
        logger.error(f"Failed to read YAML file: {path_to_yaml}")

        raise CustomException(e, sys) from e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """create list of directories
    
    Args:
        path_to_directories (list):
            List of directories to create.

        verbose (bool, optional):
            Whether to log directory creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb

    args:
        path (Path): path of the file
        
    Return:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"