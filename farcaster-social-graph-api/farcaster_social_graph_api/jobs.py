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


async def sync_lbp_data():
    """Function to sync the LBP data pipeline."""
    logging.info("Starting LBP data sync...")

    # # approx 10 minutes
    # s3_importer = AsyncS3ParquetImporter(
    #     s3_prefix="public-postgres/farcaster/v2/full/farcaster-links-"
    # )
    # file = await s3_importer.download_latest_file()
    # logging.info(f"Downloaded latest file: {file}")

    # # approx 1 minute
    # farcaster_links_aggregator = FarcasterLinksAggregator()
    # data = await farcaster_links_aggregator.execute()
    # logging.info("Farcaster links aggregated.")

    # # approx 30 seconds
    # farcaster_undirected_links_builder = FarcasterUndirectedLinksBuilder()
    # data = await farcaster_undirected_links_builder.execute()
    # logging.info("Farcaster undirected links built.")

    # approx 35 minutes
    sybil_executor = SybilScarExecutor()
    await sybil_executor.execute()
    logging.info("SybilScar executed.")


async def delete_old_files():
    files_patterns = [
        "processed-farcaster-undirected-connections-*.parquet",
        "farcaster-links-*.parquet",
        "processed-farcaster-mutual-links-*.parquet",
    ]

    async def delete_pattern(file_pattern: str):
        files = await asyncio.to_thread(
            glob.glob, os.path.join(config.DOWNLOAD_DATA_PATH, file_pattern)
        )
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # if there are more than 2 files, delete the older ones
        if len(files) > 2:
            old_files = files[2:]
            for old_file in old_files:
                try:
                    await asyncio.to_thread(os.remove, old_file)
                    logging.info(f"Deleted old file: {old_file}")
                except OSError as e:
                    logging.error(f"Error deleting file {old_file}: {e}")

    callbacks = [delete_pattern(pattern) for pattern in files_patterns]
    await asyncio.gather(*callbacks)
