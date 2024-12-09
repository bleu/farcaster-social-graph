from typing import List
import polars as pl
from ..base import FeatureExtractor, FeatureConfig

class NetworkFeatureExtractor(FeatureExtractor):
    """Extract network-based features"""
    
    def __init__(self):
        self.feature_names = [
            'following_count', 'follower_count', 'follower_ratio',
            'unique_following_count', 'unique_follower_count',
            'follower_ratio_log', 'unique_follower_ratio_log'
        ]
    
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract network features with error handling"""
        try:
            result = df.clone()
            
            # Calculate following patterns
            following = (
                df.group_by('fid')
                .agg([
                    pl.len().alias('following_count'),
                    pl.n_unique('target_fid').alias('unique_following_count')
                ])
                .fill_null(0)
            )
            
            # Calculate follower patterns
            followers = (
                df.group_by('target_fid')
                .agg([
                    pl.len().alias('follower_count'),
                    pl.n_unique('fid').alias('unique_follower_count')
                ])
                .rename({'target_fid': 'fid'})
                .fill_null(0)
            )
            
            # Join and calculate ratios
            result = (
                result.join(following, on='fid', how='left')
                .join(followers, on='fid', how='left')
                .with_columns([
                    (pl.col('follower_count') / (pl.col('following_count') + 1))
                        .alias('follower_ratio'),
                    (pl.col('unique_follower_count') / (pl.col('unique_following_count') + 1))
                        .alias('unique_follower_ratio')
                ])
            )
            
            # Add log transformations
            result = result.with_columns([
                pl.col('follower_ratio').log1p().alias('follower_ratio_log'),
                pl.col('unique_follower_ratio').log1p().alias('unique_follower_ratio_log')
            ])
            
            return result.fill_null(0)
            
        except Exception as e:
            print(f"Error in network features: {str(e)}")
            return df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
    
    def get_dependencies(self) -> List[str]:
        return ['fid', 'target_fid']