import aioboto3
import os
import aiofiles
from botocore.exceptions import NoCredentialsError
from farcaster_social_graph_api.logger import get_logger
from farcaster_social_graph_api.config import config

logger = get_logger(__name__)


class AsyncS3ParquetImporter:
    def __init__(self, s3_prefix):
        self.bucket_name = config.S3_FARCASTER_PARQUET_BUCKET_NAME
        self.s3_prefix = s3_prefix
        self.local_download_path = os.getenv("DOWNLOAD_PATH", "/data")
        self.session = aioboto3.Session(
            profile_name=config.AWS_PROFILE,
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

    async def list_files(self):
        try:
            async with self.session.client("s3") as s3:
                response = await s3.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=self.s3_prefix
                )
                if "Contents" not in response:
                    raise ValueError(
                        f"No files found in S3 path: s3://{self.bucket_name}/{self.s3_prefix}"
                    )

                files = response["Contents"]
                files_sorted = sorted(
                    files, key=lambda x: x["LastModified"], reverse=True
                )

                return files_sorted
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please check your configuration.")

    async def get_latest_file(self):
        files_sorted = await self.list_files()
        latest_file = files_sorted[0]
        return latest_file["Key"]

    async def download_latest_file(self):
        latest_file_key = await self.get_latest_file()
        file_name = latest_file_key.split("/")[-1]
        local_file_path = os.path.join(config.DOWNLOAD_DATA_PATH, file_name)

        logger.info(f"Downloading {file_name} to {local_file_path}...")

        async with self.session.client("s3") as s3:
            async with aiofiles.open(local_file_path, "wb") as f:
                await s3.download_fileobj(self.bucket_name, latest_file_key, f)

        return local_file_path
