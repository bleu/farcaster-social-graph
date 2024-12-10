from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List


@dataclass
class Config:
    """Global configuration"""

    data_path: Path
    checkpoint_dir: Path
    checkpoint_enabled = True
    model_dir: Path
    debug_mode: bool = False
    cache_enabled: bool = True
    authenticity_thresholds: Optional[Dict[str, float]] = None
    sample_size: int = 200_000
    fids_to_ensure: Optional[List[int]] = None

    def __post_init__(self):
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
