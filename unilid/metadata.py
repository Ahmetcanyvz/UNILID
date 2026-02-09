import json
import multiprocessing
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import logging
logger = logging.getLogger(__name__)


def _get_metadata_path(tokenizer_path: str) -> str:
    """Convert tokenizer path to metadata path."""
    if tokenizer_path.endswith('.json'):
        return tokenizer_path[:-5] + '.metadata.json'
    else:
        return tokenizer_path + '.metadata.json'


def _save_tokenizer_metadata(tokenizer_path: str, metadata: Dict[str, Any]) -> str:
    """
    Save metadata to a separate .metadata.json file alongside the tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer JSON file
        metadata: Metadata dictionary to save

    Returns:
        Path to the created metadata file
    """
    try:
        metadata_path = _get_metadata_path(tokenizer_path)

        metadata_with_ref = metadata.copy()
        metadata_with_ref["tokenizer_file"] = Path(tokenizer_path).name

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_with_ref, f, indent=2, ensure_ascii=False, separators=(',', ': '))

        logger.debug(f"Successfully saved metadata to {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.error(f"Failed to save metadata for {tokenizer_path}: {e}")
        return ""


def _create_base_metadata(tokenizer_class: str, tokenizer_type: str) -> Dict[str, Any]:
    """Create base metadata structure for any tokenizer"""
    try:
        return {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": str(tokenizer_class),
            "tokenizer_type": str(tokenizer_type),
            "environment_info": {
                "python_version": str(sys.version).replace('\n', ' ').strip(),
                "platform": str(platform.platform()),
                "num_processes": int(multiprocessing.cpu_count())
            }
        }
    except Exception as e:
        return {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": str(tokenizer_class),
            "tokenizer_type": str(tokenizer_type),
            "environment_info": {}
        }
