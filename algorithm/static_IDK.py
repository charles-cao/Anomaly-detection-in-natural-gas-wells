import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time

class IDK:
    def __init__(self, X, psi=2, t=100, sample=None) -> None:
        self.X = X
        self.psi = psi
        self.t = t
        self.sample = sample

        self.center_list = self.get_center_radius()
        self.true_center_list = self.X[self.center_list.reshape(-1)]
        self.radius_list, self.feature_map = self.IK_inne()

        self.feature_mean_map = np.mean(self.feature_map, axis=0)

    def get_center_radius(self):
        center_list = []
        if self.sample is None:
            for _ in range(self.t):
                center_list.append(np.random.choice(
                    self.X.shape[0], self.psi, replace=False))
        else:
            center_list = self.sample
        return np.array(center_list).reshape(self.t, -1)

    def IK_inne(self):
        radius_list = []
        output = np.zeros((self.X.shape[0], self.psi*self.t))
        for i in range(self.t):
            sample = self.X[self.center_list[i]]

            tem1 = np.dot(np.square(self.X), np.ones(sample.T.shape))
            tem2 = np.dot(np.ones(self.X.shape), np.square(sample.T))
            p2s = tem1 + tem2 - 2 * np.dot(self.X, sample.T)
            s2s = p2s[self.center_list[i], :]

            row, col = np.diag_indices_from(s2s)
            s2s[row, col] = np.inf
            temp_radius_list = np.min(s2s, axis=0)
            radius_list.append(temp_radius_list)

            p2ns_index = np.argmin(p2s, axis=1)
            p2ns = p2s[range(self.X.shape[0]), p2ns_index]
            ind = p2ns < temp_radius_list[p2ns_index]
            output[ind, (p2ns_index+i*self.psi)[ind]] = 1
        return np.array(radius_list), output

    def IDK_score(self):
        return np.dot(self.feature_map, self.feature_mean_map) / self.t

    def get_fm(self, data):
        data = data.reshape(1, -1)
        output = np.zeros((data.shape[0], self.psi*self.t))

        tem1 = np.dot(np.square(data), np.ones(self.true_center_list.T.shape))
        tem2 = np.dot(np.ones(data.shape), np.square(self.true_center_list.T))
        p2s = tem1 + tem2 - 2 * np.dot(data, self.true_center_list.T)

        p2ns_index = np.argmin(p2s, axis=1)
        p2ns = p2s[range(data.shape[0]), p2ns_index]
        ind = p2ns < self.radius_list.reshape(-1)[p2ns_index]
        output[ind, (p2ns_index)[ind]] = 1
        return output

    def get_given_score(self, index):
        index = [x + self.X.shape[0] if x < 0 else x for x in index]
        given_fm = self.feature_map[index]
        given_score = np.dot(given_fm, self.feature_mean_map) / self.t
        output = dict(zip(index, given_score))
        return output

def main(config):
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    detector = IDK(data, psi=config['argument']['psi'], t=config['argument']['t'])
    scores = detector.IDK_score()
    np.savetxt(
        f'{config["output path"]}/{config["name"]}_{config["input file"]}_{roc_auc_score(1-label, scores):.8f}_{time.time():.8f}.score', scores)
