from typing import List, Optional

from farcaster_sybil_detection.features.extractors.base import FeatureExtractor


class FeatureRegistry:
    def __init__(self):
        self._extractors = {}
        self._enabled = set()
        self._feature_providers = {}

    def register(self, name: str, extractor: FeatureExtractor):
        self._extractors[name] = extractor
        self._enabled.add(name)

        # Register feature providers
        for feature in extractor.get_feature_names():
            self._feature_providers[feature] = name

    def get_provider_for_feature(self, feature: str) -> Optional[str]:
        """Get name of extractor that provides a feature"""
        return self._feature_providers.get(feature)

    def get_extractor(self, name: str) -> Optional[FeatureExtractor]:
        return self._extractors.get(name)

    def get_enabled_names(self) -> List[str]:
        return sorted(list(self._enabled))
