import math
import matplotlib.pyplot as plt

def metrics(metrics:dict) -> None:
    '''
    Plots the logloss for each entry in the 
    given dictionary of loglosses as different subplots.

    :param metrics:   A dictionary with key = name of metric, and
                        val = list of metric values for each epoch
    '''
    metric_keys = list(metrics.keys())
    rows = int(len(metric_keys)**(1/2))
    cols = math.ceil(len(metric_keys)/rows)

    fig = plt.figure()
    for i, metric in enumerate(metric_keys):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log Loss')
        ax.plot(metrics[metric], label=metric)
        ax.legend()

    plt.show()