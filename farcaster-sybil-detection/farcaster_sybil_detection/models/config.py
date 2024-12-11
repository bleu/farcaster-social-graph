from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for models"""

    name: str
    params: Dict[str, Any]
    checkpoint_path: Optional[Path] = None
