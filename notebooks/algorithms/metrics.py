import random
import numpy as np
from collections import defaultdict


def rankdata(predictions):
    """
    Rank data with tied ranks.
    """
    sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1])
    ranks = np.zeros(len(predictions))

    sumranks, dupcount = 0, 0
    for i, (_, value) in enumerate(sorted_preds):
        sumranks += i
        dupcount += 1

        if i == len(predictions) - 1 or value != sorted_preds[i + 1][1]:
            averank = sumranks / dupcount + 1
            for j in range(i - dupcount + 1, i + 1):
                ranks[sorted_preds[j][0]] = averank
            sumranks = 0
            dupcount = 0

    return ranks


def auc(predictions, labels):
    """
    Calculate AUC.
    """
    num_p, num_n = sum(labels), len(labels) - sum(labels)
    tied_rank = rankdata(predictions)
    sum_rank = np.sum([tied_rank[i] for i in range(len(labels)) if labels[i] == 1])
    auc = (sum_rank - (num_p * num_p + num_p) / 2.0) / (num_p * num_n)
    return auc


def randint(min_val, max_val):
    return random.randint(min_val, max_val)


class Metrics:
    def __init__(self, post_file="Undirected_Facebook/post_SybilSCAR.txt", test_set_file="Undirected_Facebook/test.txt", thresh=0.5):
        self.post = defaultdict(float)
        self.pos_test_set = set()
        self.neg_test_set = set()
        self.post_file = post_file
        self.test_set_file = test_set_file
        self.thresh = thresh

    def read_scores(self):
        with open(self.post_file, "r") as f:
            for line in f:
                parts = line.split()
                node, score = int(parts[0]), float(parts[1])
                self.post[node] = score

    def read_test_data(self):
        with open(self.test_set_file, "r") as f:
            # First line: positive test nodes
            pos_line = f.readline().strip()
            self.pos_test_set = set(map(int, pos_line.split()))
            # Second line: negative test nodes
            neg_line = f.readline().strip()
            self.neg_test_set = set(map(int, neg_line.split()))

    def test_error(self):
        result = {}
        labels = [1] * len(self.pos_test_set) + [0] * len(self.neg_test_set)
        predictions = [self.post[node] for node in self.pos_test_set] + [self.post[node] for node in self.neg_test_set]
        result["AUC"] = auc(predictions, labels)

        # # Confusion matrix
        # tp, fp, tn, fn, eq_pos, eq_neg = 0, 0, 0, 0, 0, 0

        # for node in self.pos_test_set:
        #     if self.post[node] > self.thresh:
        #         tp += 1
        #     elif self.post[node] == self.thresh:
        #         eq_pos += 1
        #     else:
        #         fn += 1

        # tp += eq_pos // 2
        # fn += eq_pos - eq_pos // 2

        # for node in self.neg_test_set:
        #     if self.post[node] > self.thresh:
        #         fp += 1
        #     elif self.post[node] == self.thresh:
        #         eq_neg += 1
        #     else:
        #         tn += 1

        # tn += eq_neg // 2
        # fp += eq_neg - eq_neg // 2

        # result["ACC"] = (tp + tn) / (len(self.pos_test_set) + len(self.neg_test_set))
        # result["FPR"] = fp / (fp + tn)
        # result["TPR"] = tp / (fn + tp)
        # result["FNR"] = fn / (fn + tp)
        # result["TNR"] = tn / (fp + tn)

        return result

if __name__ == "__main__":
    # data = Data()    
    data = Metrics(
        post_file="Undirected_Farcaster/post_SybilSCAR.txt",
        test_set_file="Undirected_Farcaster/test.txt",
        thresh=0.5
    )
    
    data.read_scores()
    data.read_test_data()
    result = data.test_error()

    for key, value in result.items():
        print(f"{key}: {value}")