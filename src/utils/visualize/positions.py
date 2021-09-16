import math
import matplotlib.pyplot as plt

def node_positions(z:list, title:str) -> None:
    '''
    Visualizes the 2D points in a list z as
    dots in a plot.

    :param z:   A list of 2D coordinates to be plotted
    "param title:   Title of the plot
    '''

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    for i, zi in enumerate(z):
        plt.scatter(x=zi[0], y=zi[1], label=f'Node: {i}')

    plt.legend()
    plt.show()

def compare_positions(zs:list, titles:list) -> None:
    '''
    Visualizes the predicted node positions in a
    separate subplot for each z vector in zs. 
    Subplot i will plot z_i with title t_i
    from the list of titles.

    :param zs:      A list of latent space vectors
    "param titles:  A list of string titles
    '''
    rows = int(len(zs)**(1/2))
    cols = math.ceil(len(zs)/rows)

    fig = plt.figure()
    fig.suptitle('Initial Node Positions - Predicted vs Actual')
    for i, zi in enumerate(zs):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(titles[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log Loss')

        # Plot each node in zi in the subplot ax
        for j, val in enumerate(zi): 
            ax.scatter(val[0], val[1], label=f'Node {j}')
            
        ax.legend()

    plt.tight_layout()
    plt.show()