import matplotlib.pyplot as plt


def plot_event_dist(dataset, wandb_handler):
    event_times = dataset[:,2].tolist()
    fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
    ax.hist(event_times, bins = 100)

    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    wandb_handler.log({'distribution_plot': wandb_handler.Image(fig)})
    return


def plot_event_dist_eu_data(dataset):
    event_times = dataset[:,2].tolist()
    fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
    ax.hist(event_times, bins = 100)
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Frequency')
    ax.set_title('Interaction Distribution: EU Research Institution Email Correspondences')
    # plt.show()
    return


def plot_event_dist_resistance_data(dataset):
    event_times = dataset[:,2].tolist()
    fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
    ax.hist(event_times, bins = 100)
    ax.set_xlabel('Time in seconds')
    ax.set_ylabel('Frequency')
    ax.set_title('Interaction Distribution: Resistance Game 4 ')
    plt.show()
    return
