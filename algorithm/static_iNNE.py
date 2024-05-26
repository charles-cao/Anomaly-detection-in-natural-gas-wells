from pyod.models.inne import INNE
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time


def main(config):
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    detector = INNE(n_estimators=config['argument']['n_estimators'],
                    max_samples=config['argument']['max_samples'])
    detector.fit(data)
    scores = detector.decision_function(data)
    np.savetxt(
        f'{config["output path"]}/{config["name"]}_{config["input file"]}_{roc_auc_score(label, scores):.8f}_{time.time():.8f}.score', scores)
