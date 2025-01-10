from typing import Union
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
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

# Global configurations

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

for logger_name in ["farcaster_social_graph_api", "farcaster_sybil_detection"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

pl.Config.set_streaming_chunk_size(1_000_000)
pl.Config.set_fmt_str_lengths(50)

detector_config = Config(
    data_path=Path(config.DOWNLOAD_DATA_PATH),
    checkpoint_dir=Path(config.CHECKPOINTS_PATH),
    model_dir=Path(config.MODELS_PATH),
)

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

# Initialize scheduler
scheduler = AsyncIOScheduler()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup: Schedule jobs and run initial routine
#     try:
#         asyncio.create_task(main_routine())

#         scheduler.add_job(
#             main_routine,
#             CronTrigger(day_of_week="tue", hour="0", minute="0"),
#             id="sync_lbp_data",
#             replace_existing=True,
#         )

#         logging.info(f"Scheduler started. Jobs: {scheduler.get_jobs()}")
#         scheduler.start()
#     except Exception as e:
#         logging.error(f"Error in scheduling jobs: {str(e)}")
#         raise

#     yield  # Application runs here

#     # Shutdown: Clean up resources
#     try:
#         scheduler.shutdown()
#         logging.info("Scheduler shut down.")
#     except Exception as e:
#         logging.error(f"Error shutting down scheduler: {str(e)}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Optimism Farcaster Social Graph API",
    version="0.1.0",
    # lifespan=lifespan
)


def get_sybil_probability_sybilscar(fid: int) -> Union[float, None]:
    try:
        result_path = config.DOWNLOAD_DATA_PATH + "/sybil_scar_results.parquet"
        lazy_df = pl.scan_parquet(result_path)

        sybil_df = (
            lazy_df.filter(pl.col("fid") == fid)
            .select("posterior")
            .collect(streaming=True)
        )

        if sybil_df.is_empty():
            return None

        posterior_value = sybil_df[0, "posterior"]
        if posterior_value is None:
            return None
        return 1 - posterior_value
    except Exception as e:
        logging.error(f"Error getting SybilSCAR result for fid = {fid}: {e}")
        return None


def get_sybil_probability_ml_model(fid: int) -> Union[float, None]:
    try:
        return detector.predict(identifier=fid)["probability"]
    except Exception as e:
        logging.error(f"Error getting ML model result for fid = {fid}: {e}")
        return None


@app.get("/check-sybil/{fid}")
async def check_sybil(fid: int):
    try:
        sybilscar_result = get_sybil_probability_sybilscar(fid)
        ml_model_result = get_sybil_probability_ml_model(fid)

        if (sybilscar_result is None) and (ml_model_result is None):
            return {
                "fid": fid,
                "result": None,
                "numeric_result": None,
                "message": "Couldn't compute sybil probability for this user",
            }

        if sybilscar_result is None:
            mean = ml_model_result
        elif ml_model_result is None:
            mean = sybilscar_result
        else:
            mean = (sybilscar_result + ml_model_result) / 2

        if mean is None:
            return {
                "fid": fid,
                "result": None,
                "numeric_result": None,
                "message": "Couldn't compute sybil probability for this user",
            }

        is_benign = mean < 0.5

        return {
            "fid": fid,
            "result": "benign" if is_benign else "sybil",
            "numeric_result": round(mean, 4),
            "message": "",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    # TODO: ideally also return what pct of the job is done
    return {"status": "ok"}
