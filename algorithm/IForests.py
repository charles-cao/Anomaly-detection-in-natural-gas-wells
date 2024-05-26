from pyod.models.iforest import IForest
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import trange
import os
import time
import warnings

class IForests:
    def __init__(self,
                 X: np.ndarray,
                 psi=2,
                 t=100,
                 W=10,
                 output_index=[-1, 0]) -> None:
        self.X = X
        self.psi = psi
        self.t = t
        self.W = W
        self.output_index = np.array(
            [x + self.W if x < 0 else x for x in output_index])

        self.score_dict = defaultdict(list)

        self.n, self.dim = X.shape[0], X.shape[1]

        self.main()

    def main(self):
        for now in trange(0, self.n-self.W+1):
            detector = IForest(n_estimators=self.t, max_samples=self.psi)
            detector.fit(self.X[now:now+self.W, :])
            scores = detector.decision_function(self.X[self.output_index+now])
            for i, idx in enumerate(self.output_index):
                self.score_dict[idx+now].append(scores[i])

def main(config):
    warnings.filterwarnings("ignore")
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    t = time.time()
    detector = IForests(data,
                    psi=config['argument']['psi'],
                    t=config['argument']['t'],
                    W=config['argument']['window_size'])
    keys = list(detector.score_dict.keys())
    keys.sort()
    scores = []
    for key in keys:
        scores.append(min(detector.score_dict[key]))
    total_time = time.time() - t
    np.savetxt(f'{config["output path"]}/{config["name"]}_{config["input file"]}_{roc_auc_score(label, scores):.8f}_{time.time():.8f}.score', scores)
    # with open(f'{config["output path"]}', mode='a+') as f:
    #     print(f'{config["name"]},{config["input file"]},{roc_auc_score(label, scores)},',
    #           end='\n',
    #           file=f)
    # pass


