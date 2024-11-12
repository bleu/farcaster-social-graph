# farcaster-social-graph

### Download data

```bash
aws s3 cp s3://tf-premium-parquet/public-postgres/farcaster/v2/full/ <output_folder> \
    --recursive \
    --exclude "*" \
    --include "*-<end_timestamp>.parquet"
```
