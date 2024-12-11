# Farcaster Sybil Detection

Package to handle Sybil Detection using Machine Learning

### Main components:

**FeatureRegistry** (features/feature_registry.py) : all features that will be used must be registered first. This class handles this process

**FeatureManager** (features/feature_manager.py): a class to handle feature processing from raw datasets

**DetectorService** (services/detector.py): handles model training and predictions

### Example usage

The notebook 17-framework-usage shows an example of how to use the library

### Data needed

If AWS CLI is not configured, check [how to ingest](https://docs.neynar.com/docs/how-to-ingest).

**Farcaster**

To check most updated timestamps

```bash
aws s3 ls s3://tf-premium-parquet/public-postgres/farcaster/v2/full/"
```

To download it

```bash
aws s3 cp s3://tf-premium-parquet/public-postgres/farcaster/v2/full/ <output_dir> --recursive --exclude "*" --include "*-<end_timestamp>.parquet"
```

**Nindexer**

To check most updated timestamps

```bash
aws s3 ls s3://tf-premium-parquet/public-postgres/nindexer/v3/1/full/
```

To download it

```bash
aws s3 cp s3://tf-premium-parquet/public-postgres/nindexer/v3/1/full/  <output_dir> --recursive  --exclude "*"  --include "*-<end_timestamp>.parquet" --profile neynar_parquet_exports
```
