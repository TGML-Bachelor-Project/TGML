import math
import matplotlib.pyplot as plt

def logloss(loglosses:dict) -> None:
    '''
    Plots the logloss for each entry in the 
    given dictionary of loglosses as different subplots.

    :param loglosses:   A dictionary with key = name of logloss, and
                        val = list of loglosses for each epoch
    '''
    metric_keys = list(loglosses.keys())
    rows = int(len(metric_keys)**(1/2))
    cols = math.ceil(len(metric_keys)/rows)

    fig = plt.figure()
    for i, metric in enumerate(metric_keys):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log Loss')
        ax.plot(loglosses[metric], label=metric)
        ax.legend()

    plt.show()