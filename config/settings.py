"""
Created by Analitika at 12/08/2024
contact@analitika.fr
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJ_ROOT / "data"
logger.info(f"Project older is {PROJ_ROOT}")

# TOKENS
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
