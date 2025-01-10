from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # General feature extractor configurations
    enable_feature_x: bool = True
    feature_x_params: Dict[str, Any] = field(default_factory=dict)
    # Add other feature-specific configurations as needed
    
    # Example: Thresholds specific to feature extractors
    reaction_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'like_ratio': 0.5,
        'recast_ratio': 0.3
    })
    