from typing import Dict, List, Optional
from farcaster_sybil_detection.utils.with_logging import add_logging
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from farcaster_sybil_detection.evaluation.segmentation import UserSegmentation


@add_logging
class LabelingSampler:
    """Handle sampling for labeling with anomaly detection"""

    def __init__(self, anomaly_features: Optional[List[str]] = None):
        self.anomaly_features = anomaly_features or [
            "cast_count",
            "follower_count",
            "following_count",
            "authenticity_score",
            "total_reactions",
            "rapid_actions",
        ]

    def get_unlabeled_samples(
        self,
        matrix: pl.DataFrame,
        labeled_fids: List[int],
        samples_per_segment: int = 50,
    ) -> Dict[str, pl.DataFrame]:
        """Get stratified samples of unlabeled data using anomaly detection"""
        # Get unlabeled data
        unlabeled = matrix.filter(~pl.col("fid").is_in(labeled_fids))

        # Use segmentation to stratify
        segments = UserSegmentation().segment_users(unlabeled)

        samples = {}
        for name, segment in segments.items():
            print(f"\nProcessing {name} segment ({len(segment)} users)")

            if len(segment) == 0:
                continue

            if len(segment) > samples_per_segment:
                # Use anomaly detection for sampling
                samples[name] = self._anomaly_based_sampling(
                    segment, samples_per_segment
                )
            else:
                # Take all samples for small segments
                samples[name] = segment

            print(f"Final sample size: {len(samples[name])}")

        return samples

    def _anomaly_based_sampling(
        self, segment: pl.DataFrame, n_samples: int
    ) -> pl.DataFrame:
        """Use Isolation Forest for anomaly-based sampling"""
        # Prepare features
        valid_features = [f for f in self.anomaly_features if f in segment.columns]

        if len(valid_features) == 0:
            return segment.sample(n=n_samples, seed=42)

        X = segment.select(valid_features).fill_null(0).to_numpy()
        X = StandardScaler().fit_transform(X)

        # Detect anomalies
        iso = IsolationForest(n_estimators=100, contamination="0.1", random_state=42)
        scores = iso.fit_predict(X)

        # Split normal and anomaly indices
        anomaly_idx = np.where(scores == -1)[0]
        normal_idx = np.where(scores == 1)[0]

        # Sample both groups
        n_anomalies = min(n_samples // 4, len(anomaly_idx))
        n_normal = n_samples - n_anomalies

        selected_idx = np.concatenate(
            [
                np.random.choice(anomaly_idx, n_anomalies, replace=False),
                np.random.choice(normal_idx, n_normal, replace=False),
            ]
        )

        # Create filter
        return (
            segment.with_row_count("row_nr")
            .filter(pl.col("row_nr").is_in(selected_idx))
            .drop("row_nr")
        )
