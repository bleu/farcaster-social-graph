import random
import os
import glob
from collections import defaultdict
import polars as pl
import time
from farcaster_social_graph_api.config import config
import asyncio
import numpy as np
import logging


class SybilScar:
    def __init__(self):
        self.network_map = defaultdict(list)
        self.weighted_graph = 0
        self.prior = None
        self.post = None
        self.post_pre = None
        self.theta_pos = 0.6
        self.theta_neg = 0.4
        self.theta_unl = 0.5
        self.weight = 0.6
        self.max_iter = 10
        self.N = 0
        self.ordering_array = []
        self.semaphore = asyncio.Semaphore(4)

    def add_edge(self, node1, node2, w):
        if node1 != node2:  # Avoid self-loops
            self.network_map[node1].append((node2, w))

    # Refactored to read from in-memory connections data
    def read_network(self, connections):
        for node1, node2 in connections:
            self.add_edge(node1, node2, self.weight - 0.5)

        self.N = len(self.network_map)
        self.post = np.zeros(self.N)
        self.post_pre = np.zeros(self.N)
        self.prior = np.zeros(self.N)

    # Refactored to read from in-memory sybil and benigns sets
    def read_prior(self, train_sybils, train_benigns):
        self.prior.fill(self.theta_unl - 0.5)

        for benign in train_benigns:
            self.prior[benign] = self.theta_pos - 0.5

        for sybil in train_sybils:
            self.prior[sybil] = self.theta_neg - 0.5

    ## Write final posterior probabilities of nodes to the output file
    ## The final posterior probability is changed from p (in the residual form) to p + 0.5.
    def get_posterior(self, post_file):
        # with open(post_file, 'w') as f:
        #     for i in range(self.N):
        #         f.write(f"{i} {self.post[i] + 0.5:.10f}\n")

        data = [
            {"fid_index": i, "posterior": self.post[i] + 0.5} for i in range(self.N)
        ]
        df_lazy = pl.LazyFrame(data)
        return df_lazy

    async def lbp_thread(self, start, end):
        async with self.semaphore:
            for index in range(start, end):
                node = self.ordering_array[index]
                # update the the post for node
                for neighbor, weight in self.network_map[node]:
                    self.post[node] += 2 * self.post_pre[neighbor] * weight
                self.post[node] += self.prior[node]
                self.post[node] = max(min(self.post[node], 0.5), -0.5)

    # Async version of the LBP algorithm
    async def lbp_async(self):
        self.ordering_array = list(range(self.N))

        # initialize posts
        np.copyto(self.post, self.prior)
        iter_count = 1

        while iter_count <= self.max_iter:
            random.shuffle(self.ordering_array)
            np.copyto(self.post_pre, self.post)

            tasks = []
            num_nodes = int(
                np.ceil(self.N / self.semaphore._value)
            )  # Divide tasks by semaphore limit
            for current_thread in range(self.semaphore._value):
                start = current_thread * num_nodes
                end = min(start + num_nodes, self.N)
                task = asyncio.create_task(self.lbp_thread(start, end))
                tasks.append(task)

            await asyncio.gather(*tasks)
            iter_count += 1


class SybilScarExecutor:
    def __init__(self):
        self.data_path = config.DOWNLOAD_DATA_PATH
        self.sybil_scar = SybilScar()

    async def aget_latest_parquet_file(self):
        """Gets the latest parquet file matching a pattern."""
        file_pattern = "processed-farcaster-undirected-connections-*.parquet"
        parquet_files = await asyncio.to_thread(
            glob.glob, os.path.join(self.data_path, file_pattern)
        )
        if not parquet_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        parquet_files.sort()
        return parquet_files[-1]

    def get_latest_parquet_file(self):
        """Gets the latest parquet file matching a pattern."""
        file_pattern = "processed-farcaster-undirected-connections-*.parquet"
        parquet_files = glob.glob(os.path.join(self.data_path, file_pattern))
        if not parquet_files:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        parquet_files.sort()
        return parquet_files[-1]

    def load_data(self, parquet_path):
        """Load data from the Parquet file and process connections, sybils, and benigns."""
        links_df = pl.scan_parquet(parquet_path)

        connections_df = links_df.select(["fid_index", "target_fid_index"]).collect(
            streaming=True
        )
        self.connections = ((row[0], row[1]) for row in connections_df.iter_rows())

        sybils_df = (
            links_df.filter(pl.col("bot") == True)
            .select("fid_index")
            .unique()
            .collect(streaming=True)
            # .sample(300, seed=42)
        )
        self.sybils = (row[0] for row in sybils_df.iter_rows())

        benigns_df = (
            links_df.filter(pl.col("bot") == False)
            .select("fid_index")
            .unique()
            .collect(streaming=True)
            # .sample(300, seed=42)
        )
        self.benigns = (row[0] for row in benigns_df.iter_rows())

    async def arun_sybil_scar(self):
        """Execute the SybilScar algorithm asynchronously on the loaded data."""
        self.sybil_scar.read_network(self.connections)
        self.sybil_scar.read_prior(self.sybils, self.benigns)
        await self.sybil_scar.lbp_async()

    def save_results(self, output_file: str, latest_undirected_links_path: str):
        """Write the SybilScar post results to a file."""
        posterior_df = self.sybil_scar.get_posterior(output_file)
        latest_undirected_links_df = (
            pl.scan_parquet(latest_undirected_links_path)
            .select(["fid", "fid_index"])
            .unique()
        )
        df_lazy = posterior_df.join(latest_undirected_links_df, on="fid_index")
        df_lazy.sink_parquet(output_file)

    async def execute(self):
        """Load data, run the algorithm, and save the results."""
        logging.info("Running SybilScar...")
        start = time.time()

        parquet_path = await self.aget_latest_parquet_file()

        logging.info(f"Loading data from: {parquet_path}")
        self.load_data(parquet_path)

        logging.info("Data loaded. Running SybilScar algorithm...")
        await self.arun_sybil_scar()

        logging.info("SybilScar algorithm executed. Saving results...")
        self.save_results(
            self.data_path + "/sybil_scar_results.parquet",
            latest_undirected_links_path=parquet_path,
        )
        end = time.time()

        logging.info(f"SybilScar execution time: {end - start:.2f} seconds")
