from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Union
import hashlib
import json
import logging
import os

import polars as pl
from farcaster_sybil_detection.config.defaults import Config


class DatasetLoader:
    """Handles dataset loading with consistent FID filtering"""

    def __init__(self, config: Config):
        self.config = config
        self._base_fids: Optional[List[int]] = None
        self._cached_datasets: Dict[str, Union[pl.DataFrame, pl.LazyFrame]] = {}
        self._setup_logging()
        self._validate_paths()

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _validate_paths(self):
        """Ensure necessary directories exist"""
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_path(self, source: str, name: str) -> Path:
        """Get path for a dataset"""
        if source == "farcaster":
            return self.config.data_path / f"{source}-{name}-0-1733162400.parquet"
        elif source == "nindexer":
            return self.config.data_path / f"{source}-{name}-0-1733508243.parquet"
        else:
            raise ValueError(f"Unknown dataset source: {source}")

    def _get_cache_key(
        self,
        source: str,
        name: str,
        columns: Optional[List[str]],
        fids: Optional[List[int]],
        lazy: bool,
    ) -> str:
        """Generate unique cache key"""
        key = {
            "source": source,
            "name": name,
            "columns": columns,
            "fids": fids,
            "lazy": lazy,
        }
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()

    def _apply_filters(
        self, lf: pl.LazyFrame, columns: Optional[List[str]], fids: Optional[List[int]]
    ) -> pl.LazyFrame:
        """Apply column selection and FID filtering"""
        if columns:
            lf = lf.select(columns)

        # Apply FID filtering - if base_fids is set, use it unless specific fids are provided
        fids_to_use = fids if fids is not None else self._base_fids
        if fids_to_use is not None:
            lf = lf.filter(pl.col("fid").is_in(fids_to_use))
            self.logger.debug(f"Filtered to {len(fids_to_use)} FIDs")

        return lf

    def load_lazy_dataset(
        self,
        name: str,
        source: str = "farcaster",
        columns: Optional[List[str]] = None,
        fids: Optional[List[int]] = None,
    ) -> pl.LazyFrame:
        """Load dataset with early column and FID filtering"""
        path = self._get_dataset_path(source, name)

        # Use exact columns needed or fall back to provided columns
        cols_to_load = columns or []
        if cols_to_load and "fid" not in cols_to_load:
            cols_to_load = ["fid"] + cols_to_load

        fids_to_use = fids if fids is not None else []

        if len(fids_to_use) > 0:
            lf = (
                pl.scan_parquet(path, low_memory=True)
                .filter(pl.col("fid").is_not_null(), pl.col("fid").is_in(fids_to_use))
                .select(cols_to_load)
            )
        else:
            lf = pl.scan_parquet(path, low_memory=True).select(cols_to_load)

        self.logger.info(f"Loading {name} with columns: {cols_to_load}")

        # Get filtered size without materializing
        stats = lf.select(
            [
                pl.count().alias("total_records"),
                pl.col("fid").n_unique().alias("unique_fids"),
            ]
        ).collect()

        self.logger.info(
            f"Filtered dataset: {stats[0]['total_records'][0]} records, {stats[0]['unique_fids'][0]} unique FIDs"
        )

        return lf

    def load_dataset(
        self,
        name: str,
        source: str = "farcaster",
        columns: Optional[List[str]] = None,
        fids: Optional[List[int]] = None,
    ) -> pl.DataFrame:
        """Load dataset as DataFrame with filters applied"""
        # Use lazy loading and then collect
        lf = self.load_lazy_dataset(name, source, columns, fids)
        df = lf.collect()

        self.logger.info(f"Loaded {source}-{name}: {len(df)} records")
        return df

    def clear_cache(self):
        """Clear the cache"""
        self._cached_datasets = {}
        self.logger.info("Cache cleared")

    def get_columns(self, name: str, source: str = "farcaster") -> List[str]:
        """Get available columns for a dataset"""
        path = self._get_dataset_path(source, name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset {source}-{name} not found at {path}")
        return pl.scan_parquet(path, parallel="prefiltered", low_memory=True).columns
