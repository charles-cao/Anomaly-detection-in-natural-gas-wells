# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np
import os
import time


def main(config):
    dl = np.loadtxt(os.path.join(
        config['input path'], 'csv', config['input file']+".csv"), delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    iterator = ArrayStreamer(shuffle=False)
    model = xStream(**config['argument'])
    auroc = AUROCMetric()
    t = time.time()
    scores = []
    for x, y in tqdm(iterator.iter(data, label)):
        score = model.fit_score_partial(x)
        auroc.update(y, score)
        scores.append(score)
    np.savetxt(f'{config["output path"]}/{config["name"]}_{config["input file"]}_{auroc.get():.8f}_{time.time():.8f}.score', scores)
    total_time = time.time() - t
    # with open(f'{config["output path"]}', mode='a+') as f:
    #     print(f'{config["name"]},{config["input file"]},{auroc.get()},',
    #           end='\n',
    #           file=f)
