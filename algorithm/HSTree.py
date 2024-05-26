# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import HalfSpaceTrees
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time
from sklearn.metrics import roc_auc_score


def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = HalfSpaceTrees(**dict({'feature_mins': np.array(np.min(data)).reshape(-1),
                                   'feature_maxes': np.array(np.max(data)).reshape(-1)}, **config['argument']))
    auroc = AUROCMetric()
    scores = []
    t = time.time()
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        scores.append(score)
    total_time = time.time() - t
    np.savetxt(f'{config["output path"]}/{config["name"]}_{config["input file"]}_{roc_auc_score(label, scores):.8f}_{time.time():.8f}.score', scores)
    # with open(f'{config["output path"]}', mode='a+') as f:
    #     print(f'{config["name"]},{config["input file"]},{roc_auc_score(label, scores)},',
    #           end='\n',
    #           file=f)
