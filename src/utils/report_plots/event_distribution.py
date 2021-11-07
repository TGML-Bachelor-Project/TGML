import matplotlib.pyplot as plt
import numpy as np


def plot_event_dist(dataset, wandb_handler):
    event_times = dataset[:,2].tolist()
    fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
    ax.hist(event_times, bins = 100)
    # plt.show()
    wandb_handler.log({'distribution_plot': wandb_handler.Image(fig)})
    return



