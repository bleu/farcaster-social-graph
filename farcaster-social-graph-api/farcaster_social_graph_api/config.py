import os


class Config:
    # AWS configurations
    AWS_PROFILE = os.getenv("AWS_PROFILE")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_FARCASTER_PARQUET_BUCKET_NAME = os.getenv("S3_FARCASTER_PARQUET_BUCKET_NAME")

    # Path configurations
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DOWNLOAD_DATA_PATH = os.path.join(BASE_PATH, "data")
    PERSISTED_DATA_PATH = os.path.join(BASE_PATH, "persisted_data")

    # Environment configurations
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Honeybadger configurations
    HONEYBADGER_API_KEY = os.getenv("HONEYBADGER_API_KEY")


config = Config()
