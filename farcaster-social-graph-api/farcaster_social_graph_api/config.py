import os

class Config:
    # AWS configurations
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_FARCASTER_PARQUET_BUCKET_NAME = os.getenv("S3_FARCASTER_PARQUET_BUCKET_NAME")

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, "../../data")
    DOWNLOAD_DATA_PATH = os.path.join(DATA_PATH, "raw")
    PERSISTED_DATA_PATH = os.path.join(BASE_PATH, "persisted_data")
    CHECKPOINTS_PATH = os.path.join(DATA_PATH, "checkpoints")
    MODELS_PATH = os.path.join(BASE_PATH, "persisted_data")

    # Honeybadger configurations
    HONEYBADGER_API_KEY = os.getenv("HONEYBADGER_API_KEY")

    # Feature matrix processing sample size
    SAMPLE_SIZE = 100_000

    FILES_TO_DOWNLOAD = [
        "public-postgres/farcaster/v2/full/farcaster-account_verifications-",
        "public-postgres/farcaster/v2/full/farcaster-blocks-",
        "public-postgres/farcaster/v2/full/farcaster-casts-",
        "public-postgres/farcaster/v2/full/farcaster-channel_follows-",
        "public-postgres/farcaster/v2/full/farcaster-channel_members-",
        "public-postgres/farcaster/v2/full/farcaster-fids-",
        "public-postgres/farcaster/v2/full/farcaster-fnames-",
        "public-postgres/farcaster/v2/full/farcaster-links-",
        "public-postgres/farcaster/v2/full/farcaster-power_users-",
        "public-postgres/farcaster/v2/full/farcaster-profile_with_addresses-",
        "public-postgres/farcaster/v2/full/farcaster-reactions-",
        "public-postgres/farcaster/v2/full/farcaster-signers-",
        "public-postgres/farcaster/v2/full/farcaster-storage-",
        "public-postgres/farcaster/v2/full/farcaster-user_data-",
        "public-postgres/farcaster/v2/full/farcaster-verifications-",
        "public-postgres/farcaster/v2/full/farcaster-warpcast_power_users-",
        "public-postgres/nindexer/v3/1/full/nindexer-follow_counts-",
        "public-postgres/nindexer/v3/1/full/nindexer-follows-",
        "public-postgres/nindexer/v3/1/full/nindexer-neynar_user_scores-",
        "public-postgres/nindexer/v3/1/full/nindexer-profiles-",
        "public-postgres/nindexer/v3/1/full/nindexer-verifications-",
    ]

config = Config()
