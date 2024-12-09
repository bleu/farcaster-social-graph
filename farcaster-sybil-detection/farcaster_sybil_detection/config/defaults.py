from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
from farcaster_sybil_detection.features.config import (
    FeatureConfig,
)  # Ensure this import is correct


@dataclass
class Config:
    """Global configuration"""

    data_path: Path
    checkpoint_dir: Path
    checkpoint_enabled = True
    model_dir: Path
    debug_mode: bool = False
    cache_enabled: bool = True
    confidence_thresholds: Optional[Dict[str, float]] = None
    authenticity_thresholds: Optional[Dict[str, float]] = None
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    sample_size: int = 10_000
    fids_to_ensure: Optional[List[int]] = None

    def __post_init__(self):
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)

        if self.confidence_thresholds is None:
            self.confidence_thresholds = {"high": 0.95, "medium": 0.85, "low": 0.70}
        if self.authenticity_thresholds is None:
            self.authenticity_thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}
