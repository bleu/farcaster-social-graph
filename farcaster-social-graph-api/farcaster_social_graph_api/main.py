from fastapi import FastAPI, HTTPException, BackgroundTasks

from farcaster_social_graph_api.services.farcaster_data_collection import (
    AsyncS3ParquetImporter,
)
from farcaster_social_graph_api.services.farcaster_data_transformation import (
    FarcasterLinksAggregator,
    FarcasterUndirectedLinksBuilder,
)
from farcaster_social_graph_api.services.farcaster_sybil_detection import SybilScarExecutor
from farcaster_social_graph_api.config import config
import polars as pl

from fastapi import FastAPI
from honeybadger import honeybadger, contrib

import os
import glob
import asyncio
import psutil


app = FastAPI(title="Optimism Farcaster Social Graph API", version="0.1.0")

honeybadger.configure(
    api_key=config.HONEYBADGER_API_KEY, enviroment=config.ENVIRONMENT
)
app.add_middleware(contrib.ASGIHoneybadger)


async def sync_predictions():
    try:
        ## aprox 10 minutes
        # s3_importer = AsyncS3ParquetImporter(
        #     s3_prefix="public-postgres/farcaster/v2/full/farcaster-links-"
        # )
        # file = await s3_importer.download_latest_file()

        ## aprox 1 minute
        # farcaster_links_aggregator = FarcasterLinksAggregator()
        # data = await farcaster_links_aggregator.execute()

        ## aprox 30 seconds
        # farcaster_undirected_links_builder = FarcasterUndirectedLinksBuilder()
        # data = await farcaster_undirected_links_builder.execute()

        # aprox 35 minutes
        # sybil_executor = SybilScarExecutor()
        # sybil_executor.execute()

        return {"status": "ok"}
    except Exception as e:
        honeybadger.notify(e)
        raise e


@app.get("/sync")
async def sync_datasets(background_tasks: BackgroundTasks):
    try:
        # background_tasks.add_task(sync_predictions)
        await sync_predictions()
    except Exception as e:
        honeybadger.notify(e)
        raise HTTPException(status_code=500, detail=str(e))


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
        is_sybil = posterior_value > 0.5  # what is the threshold?

        return {"fid": fid, "result": "sybil" if is_sybil else "benign"}

    except Exception as e:
        honeybadger.notify(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory-usage")
async def memory_usage():
    process = psutil.Process()

    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss
        / (1024 * 1024),  # Resident Set Size: Memory currently used
        "vms": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        "memory_percent": process.memory_percent(),  # % of RAM used by the process
        # "shared": psutil.virtual_memory().shared / (1024 * 1024),  # Shared memory
        "total_memory": psutil.virtual_memory().total
        / (1024 * 1024),  # Total system memory in MB
        "available_memory": psutil.virtual_memory().available
        / (1024 * 1024),  # Available system memory
        "used_memory": psutil.virtual_memory().used
        / (1024 * 1024),  # Used system memory
        "free_memory": psutil.virtual_memory().free
        / (1024 * 1024),  # Free system memory
    }
    return memory_stats


@app.get("/cpu-usage")
async def cpu_usage():
    cpu_stats = {
        "cpu_percent": psutil.cpu_percent(
            interval=1
        ),  # CPU utilization across all cores
        "cpu_count": psutil.cpu_count(logical=True),  # Number of logical CPUs
        "cpu_times": psutil.cpu_times()._asdict(),  # CPU times spent in different states
        "cpu_per_core": psutil.cpu_percent(
            interval=1, percpu=True
        ),  # CPU usage per core
    }
    return cpu_stats


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
                    print(f"Deleted old file: {old_file}")
                except OSError as e:
                    print(f"Error deleting file {old_file}: {e}")

    callbacks = [delete_pattern(pattern) for pattern in files_patterns]
    await asyncio.gather(*callbacks)
