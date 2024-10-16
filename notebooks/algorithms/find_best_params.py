import random
from sklearn.model_selection import ParameterSampler
from sybil_scar import Data
from metrics import Metrics
import numpy as np

param_dist = {
    'theta_pos': np.linspace(0.3, 1.0, 11).tolist(),  # Values between 0.5 and 1.0 with finer steps (0.05 increment)
    'theta_neg': np.linspace(0.01, 0.5, 25).tolist(),  # Values between 0.01 and 0.5 with finer steps (0.02 increment)
    'weight': np.linspace(0.01, 1.0, 20).tolist(),  # Values between 0.1 and 1.0 with finer steps (0.05 increment)
    'max_iter': [5, 6, 7],
}

# Number of parameter samples to test
n_iter_search = 500

# Randomized search
random_search = list(ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42))

if __name__ == "__main__":
  for params in random_search:
      # Set parameters for the data instance
      data = Data()
      data.theta_pos = params['theta_pos']
      data.theta_neg = params['theta_neg']
      data.weight = params['weight']
      data.max_iter = params['max_iter']
      data.num_threads = 6 # params['num_threads']

      # Run the SybilSCAR algorithm
      data.read_network("Undirected_Farcaster/graph.txt")
      data.read_prior(train_set_file="Undirected_Farcaster/train.txt")
      data.lbp()
      data.write_posterior("Undirected_Farcaster/post_SybilSCAR.txt")

      metric = Metrics(
          post_file="Undirected_Farcaster/post_SybilSCAR.txt",
          test_set_file="Undirected_Farcaster/test.txt",
          thresh=0.5
      )
      
      metric.read_scores()
      metric.read_test_data()
      result = metric.test_error()

      for key, value in result.items():
          if value > 0.7:
            print(f"{key}: {value}")
            print(params, "\n\n")
        