import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import torch
import numpy as np
from utils.visualize.animation import animate_nomodel_lyon


if __name__ == '__main__':

    
    z0 = torch.load('/home/augustsemrau/drive/bachelor/TGML/src/wandb/run-20211116_102215-3j9hw6j1/files/final_z0.pt', map_location=torch.device('cpu'))
    
    v0 = torch.load('/home/augustsemrau/drive/bachelor/TGML/src/wandb/run-20211116_102215-3j9hw6j1/files/final_v0.pt', map_location=torch.device('cpu'))

    metadata = pd.read_csv('/home/augustsemrau/drive/bachelor/TGML/src/data/real/datasets/metadata_LyonSchool.csv', delimiter=',', header=None)
    metadata = metadata.values.tolist()
    # color_dict = {'cpa': 'red',
    #                 'cpb': 'cyan',
    #                 'ce1a': 'yellow',
    #                 'ce1b': 'cadetblue',
    #                 'ce2a': 'green',
    #                 'ce2b': 'darkblue',
    #                 'cm1a': 'chartreuse',
    #                 'cm1b': 'cornsilk',
    #                 'cm2a': 'burlywood',
    #                 'cm2b': 'darkviolet',
    #                 'teachers': 'black'}
    
    metadata_dict = {}

    for tup in metadata:
        # metadata_dict[str(tup[0])] = str(color_dict[str(tup[1])])
        metadata_dict[str(tup[0])] = str(tup[1])


    z0_xmean = np.mean(z0.numpy()[:,0])
    z0_ymean = np.mean(z0.numpy()[:,1])
    z0[:,0], z0[:,1] = z0[:,0] - z0_xmean, z0[:,1] -z0_ymean
    for i in range(len(v0[0,0])):
      sumx = 0
      sumy = 0
      for node in range(len(v0)):
        sumx += v0[node,0][i]
        sumy += v0[node,1][i]
      meanx = sumx / len(v0)
      meany = sumy / len(v0)
      for node in range(len(v0)):
        v0[node,0][i] = v0[node,0][i] - meanx
        v0[node,1][i] = v0[node,1][i] - meany

    time_intervals = torch.linspace(0, 84.25, v0.shape[2] + 1)
    start_times = time_intervals[:-1]
    end_times = time_intervals[1:]
    time_intervals = list(zip(start_times.tolist(), end_times.tolist()))
    time_deltas = (end_times-start_times)
    # All deltas should be equal do to linspace, so we can take the first
    step_size = time_deltas[0]
    animate_nomodel_lyon(z0=z0, v0=v0, time_deltas=time_deltas, step_size=step_size, num_of_steps=v0.shape[2], t_start=0, t_end=84.25, num_of_time_points=5000, device=None, metadata=metadata_dict)
