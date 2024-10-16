import random
import numpy as np
from collections import defaultdict
import threading

class Data:
    def __init__(self):
        self.network_map = defaultdict(list)
        self.weighted_graph = 0
        self.messages_map = {}
        self.N = 0
        self.prior = None
        self.post = None
        self.theta_pos = 0.9
        self.theta_neg = 0.1
        self.theta_unl = 0.5
        self.weight = 0.9
        self.max_iter = 5
        self.num_threads = 6
        self.ordering_array = None

    def add_edge(self, node1, node2, w):
        if node1 == node2:
            return
        self.network_map[node1].append((node2, w))

    def read_network(self, network_file):
        with open(network_file, 'r') as f:
            for line in f:
                parts = line.split()
                node1, node2 = int(parts[0]), int(parts[1])
                w = float(parts[2]) if self.weighted_graph == 1 else self.weight
                self.add_edge(node1, node2, w)

        self.N = len(self.network_map)
        self.post = np.full(self.N, 0.5)
        self.prior = np.full(self.N, self.theta_unl)

    def read_prior(self, prior_file=None, train_set_file=None):
        if prior_file:
            with open(prior_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    node, score = int(parts[0]), float(parts[1])
                    self.prior[node] = score

        if train_set_file:
            with open(train_set_file, 'r') as f:
                pos_nodes = f.readline().split()
                neg_nodes = f.readline().split()

                for node in pos_nodes:
                    self.prior[int(node)] = self.theta_pos
                for node in neg_nodes:
                    self.prior[int(node)] = self.theta_neg

    def construct_id(self, sender, receiver):
        return f"{sender} {receiver}"

    def initialize_messages(self):
        for node in range(self.N):
            for nei, _ in self.network_map[node]:
                id = self.construct_id(nei, node)
                self.messages_map[id] = 0.5

    def send_message(self, sender, receiver, w):
        other_message = [1 - self.post[sender], self.post[sender]]
        id_receiver_sender = self.construct_id(receiver, sender)
        message_receiver_sender_pos = self.messages_map[id_receiver_sender]

        other_message[0] /= 1 - message_receiver_sender_pos
        other_message[1] /= message_receiver_sender_pos

        message = [0.0, 0.0]
        for i in range(2):
            for j in range(2):
                node_potential = self.prior[sender] if j == 1 else (1 - self.prior[sender])
                edge_potential = w if i == j else 1 - w
                message[i] += node_potential * edge_potential * other_message[j]

        sum_val = message[0] + message[1]
        message[1] = message[1] / sum_val

        return min(max(message[1], 1e-10), 1 - 1e-10)

    def update_post(self):
        for node in range(self.N):
            value = [1.0, 1.0]
            for nei, _ in self.network_map[node]:
                id = self.construct_id(nei, node)
                value[0] *= 1 - self.messages_map[id]
                value[1] *= self.messages_map[id]

                sum_val = value[0] + value[1]
                value[0] /= sum_val
                value[1] /= sum_val

            self.post[node] = value[1] / (value[1] + value[0])

    def posterior(self):
        for node in range(self.N):
            value_0 = (1 - self.prior[node]) * (1 - self.post[node])
            value_1 = self.prior[node] * self.post[node]
            self.post[node] = value_1 / (value_0 + value_1)

    def lbp_thread(self, start, end):
        for index in range(start, end):
            node = self.ordering_array[index]
            for nei, w in self.network_map[node]:
                message = self.send_message(nei, node, w)
                id = self.construct_id(nei, node)
                self.messages_map[id] = message
        self.update_post()

    def lbp(self):
        self.ordering_array = list(range(self.N))
        self.initialize_messages()

        for _ in range(self.max_iter):
            random.shuffle(self.ordering_array)
            threads = []
            num_nodes = (self.N + self.num_threads - 1) // self.num_threads

            for current_thread in range(self.num_threads):
                start = current_thread * num_nodes
                end = min(start + num_nodes, self.N)
                thread = threading.Thread(target=self.lbp_thread, args=(start, end))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        self.posterior()

    def write_posterior(self, post_file):
        with open(post_file, 'w') as f:
            for i in range(self.N):
                f.write(f"{i} {self.post[i]:.10f}\n")

if __name__ == "__main__":
    data = Data()
    data.read_network("Undirected_Farcaster/graph.txt")
    data.read_prior(train_set_file="Undirected_Farcaster/train.txt")
    data.lbp()
    data.write_posterior("Undirected_Farcaster/post_SybilSCAR.txt")