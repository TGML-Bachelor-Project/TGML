import matplotlib.pyplot as plt
import numpy as np


def plot_event_dist(dataset):
    event_times = dataset[:,2].tolist()
    plt.hist(event_times, bins = 100)
    # plt.show()
    return



