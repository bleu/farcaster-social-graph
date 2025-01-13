import json
import time
import logging
import os
import numpy as np
import polars as pl
from pathlib import Path

from farcaster_social_graph_api.config import config

from farcaster_sybil_detection.config.defaults import Config
from farcaster_sybil_detection.services.detector import DetectorService
from farcaster_sybil_detection.features.registry import FeatureRegistry
from farcaster_sybil_detection.features.extractors.network_analysis_extractor import (
    NetworkAnalysisExtractor,
)
from farcaster_sybil_detection.features.extractors.temporal_behavior_extractor import (
    TemporalBehaviorExtractor,
)
from farcaster_sybil_detection.features.extractors.user_identity_extractor import (
    UserIdentityExtractor,
)
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
import glob
from prisma import Prisma

async def sync_lbp_data():
    """Function to sync the LBP data pipeline."""
    logging.info("Starting LBP data sync...")

    # approx 10 minutes
    s3_importer = AsyncS3ParquetImporter()
    await s3_importer.download_latest_files()

    logging.info("Downloaded files")


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

def remove_files_from_folder(folder_path:str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed old file {file_path}")

def remove_old_files():
    folder_paths = [config.DATA_PATH, config.CHECKPOINTS_PATH]
    for folder in folder_paths:
        remove_files_from_folder(folder)

async def build_ml_model_feature_matrix(detector):
    fids_path = get_latest_file_pattern("farcaster-fids")
    all_fids = pl.scan_parquet(fids_path).select(["fid"]).collect()
    all_fids = all_fids.with_columns(pl.col("fid").sort().alias("fid"))
    # n_fids = 50
    # sample_size = 30
    n_fids = len(all_fids)  # uncomment this for production
    sample_size = config.SAMPLE_SIZE
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
        logging.info(
            f"Processed {current_fids}/{n_fids} FIDs ({(current_fids/n_fids*100):.2f}%)"
        )

    # Retrain model
    labels_df = pl.read_parquet(f"{config.MODELS_PATH}/labels.parquet")
    detector.trainer.train(labels_df)


detector_config = Config(
    data_path=Path(config.DOWNLOAD_DATA_PATH),
    checkpoint_dir=Path(config.CHECKPOINTS_PATH),
    model_dir=Path(config.MODELS_PATH),
)

def get_latest_file_pattern(name:str)->str:
    file_pattern = f"{name}-*.parquet"
    parquet_files = glob.glob(os.path.join(config.DATA_PATH, file_pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    parquet_files.sort()
    path = parquet_files[-1]
    return path

def get_last_fnames()->pl.DataFrame:
    fnames_path = get_latest_file_pattern("farcaster-fnames")
    fnames = pl.read_parquet(fnames_path)
    last_fnames = fnames[["fid","updated_at"]].group_by("fid").max()
    last_fnames = last_fnames.join(
        fnames,
        on=["fid","updated_at"],
        how="left",
        coalesce=True
    )[["fid","fname"]]

    return last_fnames

async def post_outputs_to_db(
    sybil_threshold: float = 0.8,
    benign_threshold: float = 0.2,
    db_batch_size: int = 30_000,
    db_super_batch_size: int = 300_000
) -> None:
    """
    Process sybil detection data and post results to database.
    
    Args:
        sybil_threshold: Threshold for classifying as sybil
        benign_threshold: Threshold for classifying as benign
        db_batch_size: Batch size for database operations
        db_super_batch_size: Super batch size for database operations
    """
    
    # Process checkpoints
    checkpoints_fids = []
    for file in os.listdir(config.CHECKPOINTS_PATH):
        if "network_analysis_filtered_" not in file:
            continue
        df = pl.read_parquet(f"{config.CHECKPOINTS_PATH}/{file}")
        checkpoints_fids.extend(df["fid"].to_list())
        
    # Build feature matrix
    X = detector.feature_manager.build_feature_matrix(checkpoints_fids)
    
    # Filter out infinites
    with_inf = X.filter(pl.any_horizontal(pl.all().is_infinite()))
    without_inf = X.filter(~pl.col("fid").is_in(with_inf["fid"]))
    
    # Get model predictions
    model = detector.predictor.model
    y = model.predict_proba(without_inf[model.feature_names])

    print(f"computed {len(y[:,1])} ML predictions.")
    
    # Create results DataFrame
    results = pl.DataFrame({
        "fid": without_inf["fid"].to_list(),
        "ml_proba": y[:,1]
    }, {
        "fid": pl.Int64,
        "ml_proba": pl.Float64
    })
    
    last_fnames = get_last_fnames()
    
    # Join results with fnames
    results = results.join(last_fnames, how="left", on="fid", coalesce=True)
    
    # Process SybilScar results
    sybilscar_results = pl.read_parquet(f"{config.DATA_PATH}/interim/sybil_scar_results.parquet")
    sybilscar_results = sybilscar_results.with_columns(
        (np.ones(len(sybilscar_results)) - pl.col("posterior")).alias("sybilscar_proba")
    )
    
    # Combine all results
    all_results = results.join(
        sybilscar_results[["fid","sybilscar_proba"]],
        how="left",
        on="fid",
        coalesce=True
    )

    print(f"combined ML predictions and SybilSCAR predictions.")
    
    # Calculate ensemble predictions
    ensemble_predictions = all_results.with_columns(
        pl.when(pl.col('sybilscar_proba').is_null() & pl.col('ml_proba').is_null())
            .then(None)
            .when(pl.col('ml_proba').is_null())
            .then(pl.col('sybilscar_proba'))
            .when(pl.col('sybilscar_proba').is_null())
            .then(pl.col('ml_proba'))
            .otherwise((pl.col('sybilscar_proba') + pl.col('ml_proba')) / 2)
            .alias('sybil_probability')
    )
    
    ensemble_predictions = ensemble_predictions.with_columns(
        pl.when(pl.col('sybil_probability').is_null())
            .then(None)
            .when(pl.col('sybil_probability') > sybil_threshold)
            .then(True)
            .when(pl.col('sybil_probability') <= benign_threshold)
            .then(False)
            .otherwise(None)
            .alias('sybil_diagnosis')
    )

    print(f"Ensembled ML and SybilSCAR predictions.")
    
    # Get all FIDs and join with predictions
    
    fids_path = get_latest_file_pattern("farcaster-fids")
    all_fids = pl.read_parquet(fids_path)
    df = all_fids[["fid"]].join(
        ensemble_predictions[["fid","fname","sybil_probability","sybil_diagnosis"]],
        on="fid",
        how="left",
        coalesce=True
    )

    print(f"Built a final dataframe with all fids.")
    
    # Update database
    async def bulk_update_sybil_probabilities(df: pl.DataFrame, batch_size: int, super_batch_size: int):
        try:
            fids = [int(fid) for fid in df['fid']]
            records = [
                {
                    "fid": int(row["fid"]),
                    "fname": row["fname"],
                    "sybilProbability": float(row["sybil_probability"]) if row.get("sybil_probability") is not None else None,
                    "sybilDiagnosis": bool(row["sybil_diagnosis"]) if row.get("sybil_diagnosis") is not None else None,
                }
                for row in df.to_dicts()
            ]
            
            for super_i in range(0, len(fids), super_batch_size):
                prisma = Prisma()
                await prisma.connect()
                
                try:
                    super_batch_fids = fids[super_i:super_i + super_batch_size]
                    super_batch_records = records[super_i:super_i + super_batch_size]
                    
                    for i in range(0, len(super_batch_fids), batch_size):
                        batch_fids = super_batch_fids[i:i + batch_size]
                        await prisma.sybilprobability.delete_many(
                            where={
                                'fid': {'in': batch_fids}
                            }
                        )
                        print(f"Deleted batch {(super_i + i)//batch_size + 1} of {(len(fids)-1)//batch_size + 1}")
                    
                    for i in range(0, len(super_batch_records), batch_size):
                        batch_records = super_batch_records[i:i + batch_size]
                        await prisma.sybilprobability.create_many(
                            data=batch_records
                        )
                        print(f"Inserted batch {(super_i + i)//batch_size + 1} of {(len(records)-1)//batch_size + 1}")
                
                finally:
                    await prisma.disconnect()
                    print(f"Completed super batch {super_i//super_batch_size + 1} of {(len(fids)-1)//super_batch_size + 1}")
            print("Posted data to DB!")
        
        except Exception as e:
            print(f"Error occurred: {e}")
            raise e
    
    await bulk_update_sybil_probabilities(df, batch_size=db_batch_size, super_batch_size=db_super_batch_size)

print(detector_config.data_path)

# Initialize feature registry
registry = FeatureRegistry()
registry.register("user_identity", UserIdentityExtractor)
registry.register("network_analysis", NetworkAnalysisExtractor)
registry.register("temporal_behavior", TemporalBehaviorExtractor)
# registry.register("content_engagement", ContentEngagementExtractor)
# registry.register("reputation_meta", ReputationMetaExtractor)

# Initialize detector
detector = DetectorService(detector_config, registry)


async def main_routine():
    if os.path.exists(f"{config.CHECKPOINTS_PATH}/checkpoint.json"):
        with open(f"{config.CHECKPOINTS_PATH}/checkpoint.json", "r") as f:
            last_time_processed = json.load(f)["last_timestamp"]
        current_timestamp = round(time.time())
        six_days = 60 * 60 * 24 * 6  # Almost job rerun time
        if current_timestamp < last_time_processed + six_days:
            return

    await remove_old_files()
    await sync_lbp_data()
    await run_sybilscar()
    await build_ml_model_feature_matrix(detector)
    await post_outputs_to_db()

    # Log last time processed
    current_timestamp = round(time.time())
    with open(f"{config.CHECKPOINTS_PATH}/checkpoint.json", "w") as f:
        json.dump({"last_timestamp": current_timestamp}, f)


if __name__ == "__main__":
    asyncio.run(main_routine())
