from typing import Dict, List, Tuple
import polars as pl
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class Segment:
    name: str
    filter_expr: str  # Polars expression string
    feature_names: List[str]

class UserSegmentation:
    """Handle user segmentation and segment-specific analysis"""
    
    def __init__(self):
        self.segments = {
            'power_users': Segment(
                name='power_users',
                filter_expr="(cast_count >= 20) AND (reply_count >= 5)",
                feature_names=[
                    'cast_count', 'total_reactions', 'avg_cast_length',
                    'reply_count', 'mentions_count', 'engagement_score',
                    'weekday_diversity', 'hour_diversity'
                ]
            ),
            'casual_users': Segment(
                name='casual_users',
                filter_expr="(cast_count >= 5) AND (cast_count < 20)",
                feature_names=[
                    'cast_count', 'total_reactions', 'engagement_score',
                    'reply_count', 'avg_cast_length'
                ]
            ),
            'lurkers': Segment(
                name='lurkers',
                filter_expr="cast_count = 0",
                feature_names=[
                    'profile_completeness', 'follower_count',
                    'following_count', 'authenticity_score'
                ]
            )
        }
    
    def segment_users(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Segment users based on behavior patterns"""
        segments = {}
        total = len(df)
        
        print("\nUser Segment Distribution:")
        for name, segment in self.segments.items():
            try:
                # Check if required columns exist
                required_cols = self._extract_column_names(segment.filter_expr)
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Skipping {name} segment due to missing columns: {missing_cols}")
                    continue
                
                # Apply filter using Polars expression
                segmented = df.filter(segment.filter_expr)
                segments[name] = segmented
                
                size = len(segmented)
                if size > 0:
                    print(f"{name}: {size:,} users ({size/total*100:.1f}%)")
                    
                    if 'bot' in segmented.columns:
                        bot_pct = (segmented.filter("bot = 1").shape[0] / size) * 100
                        print(f"  Bot percentage: {bot_pct:.1f}%")
                    
                    # Print average metrics for available columns
                    available_metrics = ['cast_count', 'follower_count', 'following_count']
                    metrics_to_print = [col for col in available_metrics if col in df.columns]
                    
                    if metrics_to_print:
                        metrics = segmented.select([
                            pl.col(col).mean() for col in metrics_to_print
                        ]).to_numpy()[0]
                        
                        for col, value in zip(metrics_to_print, metrics):
                            print(f"  Avg {col}: {value:.1f}")
                
            except Exception as e:
                print(f"Error processing segment {name}: {str(e)}")
                continue
        
        return segments
    
    def _extract_column_names(self, expr: str) -> List[str]:
        """Extract column names from a filter expression"""
        # Simple extraction - can be made more robust if needed
        tokens = expr.replace('(', ' ').replace(')', ' ').split()
        return [token for token in tokens 
                if token not in ['AND', 'OR', '>=', '<=', '=', '<', '>', '0', '5', '20']
                and not token.isdigit()]
    
    def prepare_segment_features(self, segment_df: pl.DataFrame, 
                               segment_name: str) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for a specific segment"""
        segment = self.segments[segment_name]
        valid_features = [col for col in segment.feature_names 
                         if col in segment_df.columns]
        
        if not valid_features:
            raise ValueError(f"No valid features found for segment {segment_name}")
        
        X = segment_df.select(valid_features).fill_null(0).to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, valid_features