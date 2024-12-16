import aioboto3
import os
import aiofiles
from typing import List
import asyncio
from botocore.exceptions import NoCredentialsError
from farcaster_social_graph_api.config import config
import logging


class AsyncS3ParquetImporter:
    def __init__(self, s3_prefix: str = "public-postgres/farcaster/v2/full/"):
        self.bucket_name = config.S3_FARCASTER_PARQUET_BUCKET_NAME
        self.s3_prefix = s3_prefix
        self.local_download_path = os.getenv("DOWNLOAD_PATH", "/data")
        self.session = aioboto3.Session(
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

    async def list_files(self, file_name: str = None):
        try:
            async with self.session.client("s3") as s3:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=file_name if file_name else self.s3_prefix,
                )
                if "Contents" not in response:
                    raise ValueError(
                        f"No files found in S3 path: s3://{self.bucket_name}/{file_name if file_name else self.s3_prefix}"
                    )

                files = response["Contents"]
                files_sorted = sorted(
                    files, key=lambda x: x["LastModified"], reverse=True
                )

                return files_sorted
        except NoCredentialsError:
            logging.error("AWS credentials not found. Please check your configuration.")

    async def get_latest_file(self, prefix: str = None):
        files_sorted = await self.list_files(prefix)
        latest_file = files_sorted[0]
        return latest_file["Key"]

    async def download_latest_file(self, file_name: str = None) -> str:
        """Download the latest file for a given prefix."""
        latest_file_key = await self.get_latest_file(file_name)
        file_name = latest_file_key.split("/")[-1]
        local_file_path = os.path.join(config.DOWNLOAD_DATA_PATH, file_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if os.path.exists(local_file_path):
            logging.info(f"File {local_file_path} already exists! Skipping download...")
            return local_file_path

        logging.info(f"Downloading {file_name} to {local_file_path}...")

        async with self.session.client("s3") as s3:
            async with aiofiles.open(local_file_path, "wb") as f:
                await s3.download_fileobj(self.bucket_name, latest_file_key, f)

        return local_file_path

    async def download_latest_files(self) -> List[str]:
        """Download the latest files for multiple prefixes concurrently."""
        tasks = [
            self.download_latest_file(file_name)
            for file_name in config.FILES_TO_DOWNLOAD
        ]
        await asyncio.gather(*tasks)
