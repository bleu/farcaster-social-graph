from fastapi import FastAPI, HTTPException, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
from farcaster_social_graph_api.jobs import sync_lbp_data, delete_old_files
from farcaster_social_graph_api.config import config
import polars as pl

from fastapi import FastAPI
# from honeybadger import honeybadger, contrib

from apscheduler.triggers.cron import CronTrigger

import asyncio
import logging

from pathlib import Path
import polars as pl
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

pl.Config.set_streaming_chunk_size(1_000_000)
pl.Config.set_fmt_str_lengths(50)

detector_config = Config(
    data_path=Path(f"{config.DOWNLOAD_DATA_PATH}/../../../data/raw"),
    checkpoint_dir=Path(f"{config.DOWNLOAD_DATA_PATH}/../../../notebooks/checkpoints"),
    model_dir=Path(f"{config.DOWNLOAD_DATA_PATH}/../../../notebooks/models"),
)

registry = FeatureRegistry()

registry.register("user_identity", UserIdentityExtractor)
registry.register("network_analysis", NetworkAnalysisExtractor)
registry.register("temporal_behavior", TemporalBehaviorExtractor)
#registry.register("content_engagement", ContentEngagementExtractor)
# registry.register("reputation_meta", ReputationMetaExtractor)

detector = DetectorService(detector_config, registry)

app = FastAPI(title="Optimism Farcaster Social Graph API", version="0.1.0")

# honeybadger.configure(api_key=config.HONEYBADGER_API_KEY, enviroment=config.ENVIRONMENT)
# app.add_middleware(contrib.ASGIHoneybadger)
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def startup_event():
    """Schedule the cron jobs and start sync_lbo_data when the app starts."""
    try:
        # asyncio.create_task(sync_lbp_data())
        # asyncio.create_task(delete_old_files())

        # scheduler.add_job(
        #     sync_lbp_data,
        #     CronTrigger(day_of_week="tue", hour="0", minute="0"),
        #     id="sync_lbp_data",
        #     replace_existing=True,
        # )

        # scheduler.add_job(
        #     delete_old_files,
        #     CronTrigger(hour="0", minute="0"),
        #     id="delete_old_files",
        #     replace_existing=True,
        # )

        # logging.info(f"Scheduler started. Jobs: {scheduler.get_jobs()}")
        # scheduler.start()
        pass
    except Exception as e:
        # honeybadger.notify(e)
        logging.error(f"Error in scheduling jobs: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler."""
    scheduler.shutdown()
    logging.info("Scheduler shut down.")


@app.get("/check-sybil/{fid}")
async def check_sybil(fid: int):
    try:
        result_path = config.DOWNLOAD_DATA_PATH + "/sybil_scar_results.parquet"
        lazy_df = pl.scan_parquet(result_path)

        sybil_df = (
            lazy_df.filter(pl.col("fid") == fid)
            .select("posterior")
            .collect(streaming=True)
        )

        if sybil_df.is_empty():
            return {"fid": fid, "result": "Not Found"}

        posterior_value = sybil_df[0, "posterior"]
        
        detector_prediction = detector.predict(identifier=fid)["probability"]
        mean = ((1-posterior_value) + detector_prediction)/2
        is_benign = mean < 0.5

        return {"fid": fid, "result": "benign" if is_benign else "sybil", "numeric_result":mean}

    except Exception as e:
        # honeybadger.notify(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/test")
async def test():
    return {"test": "api-is-working"}
