# IMP new template improved ones

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "emotion-detection"

# List of files/folders to scaffold for the emotion-detection project
list_of_files = [
    # --- backend ---
    "backend/main.py",
    "backend/app/__init__.py",
    "backend/models/__init__.py",
    "backend/services/__init__.py",
    "backend/utils/__init__.py",
    "backend/tests/__init__.py",

    # --- frontend ---
    "frontend/src/.gitkeep",
    "frontend/public/.gitkeep",
    "frontend/tests/.gitkeep",

    # --- ml ---
    "ml/notebooks/.gitkeep",
    "ml/data/.gitkeep",
    "ml/scripts/__init__.py",
    "ml/models/.gitkeep",
    "ml/configs/.gitkeep",
    "ml/training/__init__.py",

    # --- docs ---
    "docs/architecture/.gitkeep",
    "docs/api/.gitkeep",
    "docs/images/.gitkeep",

    # --- github workflows / templates ---
    ".github/workflows/.gitkeep",
    ".github/ISSUE_TEMPLATE/.gitkeep",
    ".github/PULL_REQUEST_TEMPLATE.md",

    # --- docker ---
    "docker/docker-compose.yml",

    # --- root-level files ---
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "requirements.txt",
]

# logic to create the above files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")
