import random
import math
import threading
from collections import defaultdict
from typing import Set, Dict, Tuple

class Data:
    def __init__(self):
        self.network_map: Dict[int, Set[Tuple[int, float]]] = defaultdict(set)
        self.pos_train_set: Set[int] = set()
        self.N: int = 0
        self.weighted_graph: int = 0
        self.weight: float = 1.0
        self.alpha: float = 0.15
        self.max_iter: int = 10
        self.num_threads: int = 1
        self.prior: Dict[int, float] = {}
        self.post: Dict[int, float] = {}
        self.post_pre: Dict[int, float] = {}
        self.ordering_array: list = []

    def add_edge(self, node1: int, node2: int, w: float):
        if node1 == node2:
            return
        self.network_map[node1].add((node2, w))
        self.network_map[node2].add((node1, w))

    def read_network(self, network_file: str):
        with open(network_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                node1 = int(parts[0])
                node2 = int(parts[1])
                w = float(parts[2]) if self.weighted_graph == 1 else self.weight
                self.add_edge(node1, node2, w)
        self.N = len(self.network_map)
        self.post = {node: 0.0 for node in range(self.N)}
        self.post_pre = {node: 0.0 for node in range(self.N)}
        self.prior = {node: 0.0 for node in range(self.N)}

    def read_prior(self, prior_file: str = "", train_set_file: str = ""):
        for node in range(self.N):
            self.prior[node] = 0.0

        if prior_file:
            with open(prior_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    node = int(parts[0])
                    score = float(parts[1])
                    self.prior[node] = score

        if train_set_file:
            with open(train_set_file, 'r') as file:
                line = file.readline().strip()
                nodes = map(int, line.split())
                for node in nodes:
                    self.pos_train_set.add(node)
                    self.prior[node] = 1.0

    def rw_thread(self, start: int, end: int):
        for index in range(start, end):
            node = self.ordering_array[index]
            message = 0.0
            if node in self.network_map:
                for neighbor, weight in self.network_map[node]:
                    sum_weights = sum(w for _, w in self.network_map[neighbor])
                    message += self.post_pre[neighbor] * weight / sum_weights if sum_weights != 0 else 0
            self.post[node] = (1 - self.alpha) * message + self.alpha * self.prior[node]

    def rw(self):
        self.ordering_array = list(range(self.N))
        for node in range(self.N):
            self.post[node] = self.prior[node]

        iter_num = 1
        self.max_iter = int(math.log(self.N))

        while iter_num <= self.max_iter:
            self.post_pre = self.post.copy()
            random.shuffle(self.ordering_array)
            threads = []
            num_nodes = math.ceil(self.N / self.num_threads)

            for current_thread in range(self.num_threads):
                start = current_thread * num_nodes
                end = min(start + num_nodes, self.N)
                thread = threading.Thread(target=self.rw_thread, args=(start, end))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            iter_num += 1

    def normalize_post(self):
        for node in range(self.N):
            self.post[node] /= len(self.network_map[node]) if len(self.network_map[node]) != 0 else 1

    def write_posterior(self, post_file: str):
        with open(post_file, 'w') as file:
            for node in range(self.N):
                file.write(f"{node} {self.post[node]:.10f}\n")

    def parse_parameters(self, params: dict):
        self.network_file = params.get('network_file', "")
        self.train_set_file = params.get('train_set_file', "")
        self.prior_file = params.get('prior_file', "")
        self.post_file = params.get('post_file', "")
        self.alpha = params.get('alpha', 0.15)
        self.max_iter = params.get('max_iter', 10)
        self.num_threads = params.get('num_threads', 1)
        self.weighted_graph = params.get('weighted_graph', 0)
        self.weight = params.get('weight', 1.0)
    


if __name__ == "__main__":
    parameters = {
        'network_file': 'Undirected_Farcaster/graph.txt',
        'train_set_file': 'Undirected_Farcaster/train.txt',
        'prior_file': '',
        'post_file': 'Undirected_Farcaster/post_SybilSCAR.txt',
        'alpha': 0.15,
        'max_iter': 10,
        'num_threads': 4,
        'weighted_graph': 0,
        'weight': 1
    }
    data = Data()
    data.parse_parameters(parameters)
    data.read_network(parameters['network_file'])
    data.read_prior(parameters.get('prior_file', ""), parameters.get('train_set_file', ""))
    data.rw()
    data.normalize_post()
    data.write_posterior(parameters['post_file'])