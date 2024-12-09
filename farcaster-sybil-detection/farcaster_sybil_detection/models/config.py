from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for models"""
    name: str
    params: Dict[str, Any]
    checkpoint_path: Optional[Path] = None
    confidence_thresholds: Optional[Dict[str, float]] = None
    enable_calibration: bool = True
    enable_stacking: bool = True
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                'high': 0.95,
                'medium': 0.85,
                'low': 0.70
            }
