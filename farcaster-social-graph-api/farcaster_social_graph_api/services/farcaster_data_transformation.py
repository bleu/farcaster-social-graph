import os
import glob
import asyncio
import aiofiles
import polars as pl
import time
from farcaster_social_graph_api.config import config
import logging


class FarcasterBaseProcessor:
    def __init__(self):
        self.data_path = config.DOWNLOAD_DATA_PATH
        self.persisted_data_path = config.PERSISTED_DATA_PATH

    async def get_latest_parquet_file(self, file_pattern):
        """Gets the latest parquet file matching a pattern."""
        parquet_files = await asyncio.to_thread(
            glob.glob, os.path.join(self.data_path, file_pattern)
        )
        if not parquet_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        parquet_files.sort()
        return parquet_files[-1]

    def get_links_lazy_df(self, file_path):
        """Returns a lazy DataFrame for the given parquet file."""
        return pl.scan_parquet(file_path)

    def write_links_to_parquet(self, df, filename_suffix):
        """Writes the DataFrame to a parquet file with a unique timestamp."""
        filename = f"/{filename_suffix}-{int(time.time())}.parquet"
        df.sink_parquet(self.data_path + filename)

    def execute(self):
        """Template method to be overridden by subclasses."""
        raise NotImplementedError("Subclasses should implement the `execute` method.")


class FarcasterLinksAggregator(FarcasterBaseProcessor):
    async def execute(self):
        logging.info("Aggregating links...")
        start = time.time()
        latest_file = await self.get_latest_parquet_file("farcaster-links-0-*.parquet")
        links_lazy_df = self.get_links_lazy_df(latest_file)
        mutual_links = self.get_mutual_links(links_lazy_df)
        self.write_links_to_parquet(mutual_links, "processed-farcaster-mutual-links")
        logging.info(f"Execution time: {time.time() - start} seconds")
        return mutual_links

    def get_mutual_links(self, links_df):
        df_filtered = links_df.filter(
            (pl.col("deleted_at").is_null())
            & (pl.col("fid") != pl.col("target_fid"))
            & (pl.col("type") == "follow")
        ).select(["fid", "target_fid"])

        df_sorted = df_filtered.with_columns(
            [
                pl.min_horizontal(["fid", "target_fid"]).alias("sorted_fid"),
                pl.max_horizontal(["fid", "target_fid"]).alias("sorted_target_fid"),
            ]
        )

        df_grouped = df_sorted.group_by(["sorted_fid", "sorted_target_fid"]).agg(
            pl.count().alias("count")
        )

        return df_grouped.filter(pl.col("count") == 2).select(
            [
                pl.col("sorted_fid").alias("fid"),
                pl.col("sorted_target_fid").alias("target_fid"),
            ]
        )


class FarcasterUndirectedLinksBuilder(FarcasterBaseProcessor):
    async def execute(self):
        logging.info("Building undirected links...")
        start = time.time()
        latest_file = await self.get_latest_parquet_file(
            "processed-farcaster-mutual-links-*.parquet"
        )
        links_lazy_df = self.get_links_lazy_df(latest_file)
        undirected_links = self.get_undirected_links(links_lazy_df)
        self.write_links_to_parquet(
            undirected_links, "processed-farcaster-undirected-connections"
        )
        logging.info(f"Execution time: {time.time() - start} seconds")
        return undirected_links

    def get_undirected_links(self, links_df):
        fids = links_df.select("fid").unique()
        target_fids = (
            links_df.select("target_fid").unique().rename({"target_fid": "fid"})
        )
        all_fids = (
            pl.concat([fids, target_fids]).unique().collect()
        )  # test streaming mode

        # Use the collected DataFrame's shape to get the height
        mutual_reindex = all_fids.with_columns(
            pl.arange(0, all_fids.shape[0]).alias("index")
        )

        mutual_links_with_index = links_df.join(
            mutual_reindex.select(
                [pl.col("fid"), pl.col("index").alias("fid_index")]
            ).lazy(),
            on="fid",
            how="left",
        ).join(
            mutual_reindex.select(
                [pl.col("fid"), pl.col("index").alias("target_fid_index")]
            ).lazy(),
            left_on="target_fid",
            right_on="fid",
            how="left",
        )

        df_reversed = mutual_links_with_index.select(
            [
                pl.col("target_fid").alias("fid"),
                pl.col("fid").alias("target_fid"),
                pl.col("target_fid_index").alias("fid_index"),
                pl.col("fid_index").alias("target_fid_index"),
            ]
        )

        order = ["fid", "target_fid", "fid_index", "target_fid_index"]
        mutual_links_with_index_concatenated = pl.concat(
            [mutual_links_with_index.select(order), df_reversed.select(order)]
        )

        # mutual_links_with_index_concatenated = mutual_links_with_index_concatenated.with_columns(
        #     (pl.col("fid_index").cast(pl.Utf8) + " " + pl.col("target_fid_index").cast(pl.Utf8)).alias("connection")
        # )

        labels_df = pl.scan_parquet(
            f"/{self.persisted_data_path}/labels.parquet"
        )

        return mutual_links_with_index_concatenated.join(
            labels_df, how="left", on="fid"
        ).select("fid", "fid_index", "target_fid_index", "bot")


# class FarcasterUndirectedLinksToLPBEntries(FarcasterBaseProcessor):
#     async def execute(self):
#         start = time.time()
#         latest_file = await self.get_latest_parquet_file("processed-farcaster-undirected-connections-*.parquet")
#         links_lazy_df = self.get_links_lazy_df(latest_file)
#         mutual_links = await self.process_links_to_lpb(links_lazy_df)
#         logging.info(f"Execution time: {time.time() - start} seconds")
#         return mutual_links

#     async def process_links_to_lpb(self, links_df):
#         connections_list = links_df.select("connection").collect(streaming=True).to_series().to_list()
#         train_sybils = links_df.filter(pl.col("bot") == True).select("fid_index").collect(streaming=True).to_series().to_list()
#         train_benigns = links_df.filter(pl.col("bot") == False).select("fid_index").collect(streaming=True).to_series().to_list()

#         current_time = int(time.time())
#         await asyncio.gather(
#             self.write_to_txt(connections_list, f"connections-{current_time}.txt"),
#             self.write_to_txt(train_sybils, f"train_sybils-{current_time}.txt"),
#             self.write_to_txt(train_benigns, f"train_benigns-{current_time}.txt")
#         )

#     async def write_to_txt(self, data_list, filename):
#         async with aiofiles.open(os.path.join(self.data_path, filename), 'w') as f:
#             await f.write("\n".join(map(str, data_list)))
