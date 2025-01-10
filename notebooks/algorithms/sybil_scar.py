import random
import threading
from collections import defaultdict
import numpy as np

class Data:
    def __init__(self):
        self.network_map = defaultdict(list)
        self.weighted_graph = 0
        self.prior = None
        self.post = None
        self.post_pre = None
        self.train_set_file = None
        self.theta_pos = 0.6
        self.theta_neg = 0.4
        self.theta_unl = 0.5
        self.weight = 0.6
        self.max_iter = 10
        self.num_threads = 1
        self.N = 0
        self.ordering_array = []

    # Store edges and weights in the network_map
    def add_edge(self, node1, node2, w):
        # no self loops
        if node1 == node2:
            return
        # add node2 to the adjacency list of node1
        self.network_map[node1].append((node2, w))

    # Read the social graph
    # the format for the social graph is
    # each line corresponds to an edge, e.g, 3 2 0.8
    # each edge in the graph appears twice, e.g.,
    # 3 2 0.8
    # 2 3 0.9
    def read_network(self, network_file):
        with open(network_file, 'r') as f:
            for line in f:
                parts = line.split()
                node1 = int(parts[0])
                node2 = int(parts[1])
                w = float(parts[2]) - 0.5 if self.weighted_graph else self.weight - 0.5
                self.add_edge(node1, node2, w)
        
        self.N = len(self.network_map)
        self.post = np.zeros(self.N)
        self.post_pre = np.zeros(self.N)
        self.prior = np.zeros(self.N)

    def read_prior(self, prior_file=None, train_set_file=None):
        self.prior.fill(self.theta_unl - 0.5)
        if prior_file:
            with open(prior_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    node = int(parts[0])
                    score = float(parts[1])
                    self.prior[node] = score - 0.5

        # reassign node priors for labeled benign (positive) nodes or/and Sybils (negative nodes) 
        if train_set_file:
            with open(train_set_file, 'r') as f:
                # reading labeled benign nodes.
                pos_line = f.readline()
                for sub in map(int, pos_line.split()):
                    self.prior[sub] = self.theta_pos - 0.5
                
                # reading labeled Sybils
                neg_line = f.readline()
                for sub in map(int, neg_line.split()):
                    self.prior[sub] = self.theta_neg - 0.5

    ## Write final posterior probabilities of nodes to the output file
    ## The final posterior probability is changed from p (in the residual form) to p + 0.5.
    def write_posterior(self, post_file):
        with open(post_file, 'w') as f:
            for i in range(self.N):
                f.write(f"{i} {self.post[i] + 0.5:.10f}\n")

    # Mainloop of SybilSCAR
    def lbp_thread(self, start, end):
        for index in range(start, end):
            node = self.ordering_array[index]
            # update the the post for node
            for neighbor, weight in self.network_map[node]:
                self.post[node] += 2 * self.post_pre[neighbor] * weight
            self.post[node] += self.prior[node]
            self.post[node] = max(min(self.post[node], 0.5), -0.5)

    # Multithread to speed up the calculation
    def lbp(self):
        self.ordering_array = list(range(self.N))
        
        # initialize posts
        np.copyto(self.post, self.prior)
        iter_count = 1

        while iter_count <= self.max_iter:
          
            random.shuffle(self.ordering_array)
            np.copyto(self.post_pre, self.post)

            threads = []
            num_nodes = int(np.ceil(self.N / self.num_threads))
            for current_thread in range(self.num_threads):
                start = current_thread * num_nodes
                end = min(start + num_nodes, self.N)
                thread = threading.Thread(target=self.lbp_thread, args=(start, end))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            iter_count += 1
            
param_dist = {
    'theta_pos': [0.7, 0.8, 0.9, 1.0],
    'theta_neg': [0.1, 0.2, 0.3, 0.4],
    'weight': [0.4, 0.5, 0.6, 0.7, 0.8],
    'max_iter': [5, 10, 15, 20],
    # 'num_threads': [1, ]
}

if __name__ == "__main__":
    data = Data()
    # args = data.parse_args()

    data.weighted_graph = 0
    data.theta_pos = 0.9
    data.theta_neg = 0.1
    # data.theta_unl = args.tu
    data.weight = 0.6
    data.max_iter = 6
    data.num_threads = 1

    # data.read_network("Undirected_Facebook/graph.txt")
    # data.read_prior(train_set_file="Undirected_Facebook/train.txt")
    # data.lbp()
    # data.write_posterior("Undirected_Facebook/post_SybilSCAR.txt")
    
    data.read_network("Undirected_Farcaster/graph.txt")
    data.read_prior(train_set_file="Undirected_Farcaster/train.txt")
    data.lbp()
    data.write_posterior("Undirected_Farcaster/post_SybilSCAR.txt")


# AUC: 0.702386250831366
# {'weight': 0.4789473684210527, 'theta_pos': 0.95, 'theta_neg': 0.07125, 'max_iter': 6}


# AUC: 0.7019575605575767
# {'weight': 0.4789473684210527, 'theta_pos': 0.7, 'theta_neg': 0.27541666666666664, 'max_iter': 6}


# AUC: 0.7001894110405398
# {'weight': 0.43157894736842106, 'theta_pos': 0.65, 'theta_neg': 0.23458333333333334, 'max_iter': 6}

