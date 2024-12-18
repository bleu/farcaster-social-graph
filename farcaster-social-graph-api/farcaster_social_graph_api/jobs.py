from farcaster_social_graph_api.services.farcaster_data_collection import (
    AsyncS3ParquetImporter,
)
from farcaster_social_graph_api.services.farcaster_data_transformation import (
    FarcasterLinksAggregator,
    FarcasterUndirectedLinksBuilder,
)
from farcaster_social_graph_api.services.farcaster_sybil_detection import (
    SybilScarExecutor,
)
import asyncio
import os
import glob
from farcaster_social_graph_api.config import config
import logging
import polars as pl


async def sync_lbp_data():
    """Function to sync the LBP data pipeline."""
    logging.info("Starting LBP data sync...")

    # approx 10 minutes
    s3_importer = AsyncS3ParquetImporter()
    await s3_importer.download_latest_files()

    logging.info(f"Downloaded files")

async def run_sybilscar():
    """Function to execute SybilSCAR pipeline."""

    # approx 1 minute
    farcaster_links_aggregator = FarcasterLinksAggregator()
    data = await farcaster_links_aggregator.execute()
    logging.info("Farcaster links aggregated.")

    # approx 30 seconds
    farcaster_undirected_links_builder = FarcasterUndirectedLinksBuilder()
    data = await farcaster_undirected_links_builder.execute()
    logging.info("Farcaster undirected links built.")

    # approx 5 minutes
    sybil_executor = SybilScarExecutor()
    await sybil_executor.execute()
    logging.info("SybilScar executed.")


async def delete_old_files():
    raw_patterns = [file.split("/")[-1] + "*.parquet" for file in config.FILES_TO_DOWNLOAD]
    processed_patterns = [
        "processed-farcaster-undirected-connections-*.parquet",
        "processed-farcaster-mutual-links-*.parquet",
    ]
    files_patterns = raw_patterns + processed_patterns

    async def delete_pattern(file_pattern: str):
        files = await asyncio.to_thread(
            glob.glob, os.path.join(config.DOWNLOAD_DATA_PATH, file_pattern)
        )
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # if there are more than 1 file, delete the older ones
        if len(files) > 1:
            old_files = files[1:]
            for old_file in old_files:
                try:
                    await asyncio.to_thread(os.remove, old_file)
                    logging.info(f"Deleted old file: {old_file}")
                except OSError as e:
                    logging.error(f"Error deleting file {old_file}: {e}")

    callbacks = [delete_pattern(pattern) for pattern in files_patterns]
    await asyncio.gather(*callbacks)





async def build_ml_model_feature_matrix(detector):
    file_pattern = "farcaster-fids-*.parquet"
    parquet_files = await asyncio.to_thread(
        glob.glob, os.path.join(config.DOWNLOAD_DATA_PATH, file_pattern)
    )
    if not parquet_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    parquet_files.sort()
    farscaster_fids_file = parquet_files[-1]
    
    all_fids = pl.scan_parquet(farscaster_fids_file).select(["fid"]).collect()
    all_fids = all_fids.with_columns(pl.col("fid").sort().alias("fid"))
    n_fids = 50
    sample_size = 30
    # n_fids = len(all_fids) # uncomment this for production
    # sample_size = config.SAMPLE_SIZE
    current_fids = 0

    while current_fids < n_fids:
        start_index = current_fids
        end_index = min(current_fids + sample_size, n_fids)
        
        # Get the batch of FIDs
        fid_batch = all_fids["fid"][start_index:end_index].to_list()
        
        # Build features for this batch
        detector.feature_manager.build_feature_matrix(fid_batch)
        
        # Update progress
        current_fids = end_index
        logging.info(f"Processed {current_fids}/{n_fids} FIDs ({(current_fids/n_fids*100):.2f}%)")
    
    # Retrain model
    # labels_df = pl.read_parquet(f"{config.MODELS_PATH}/labels.parquet")
    # detector.trainer.train(labels_df)

