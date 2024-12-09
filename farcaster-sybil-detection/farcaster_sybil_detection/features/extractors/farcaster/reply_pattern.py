from typing import List
import polars as pl
from ..base import FeatureExtractor, FeatureConfig

class ReplyPatternExtractor(FeatureExtractor):
    """Extract reply behavior patterns"""
    feature_names = ['reply_count', 'unique_replied_to',
                    'reply_depth', 'conversation_length']