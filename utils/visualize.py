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

def compare_positions(z1:list, z2:list, title:str) -> None:
    '''
    Visualizes the predicted node positions in z1 
    against the actual node positions in z2.
    z1 and z2 should be equal in length.

    :param z1:      First set of node positions in 2D latent space
    :param z2:      Second set of node positions in 2D latent space
    "param title:   Title of the plot
    '''

    if len(z1) != len(z2):
        raise Exception('z1 and z2 represents positions of the same node. \
                        So they should be equal length.')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(len(z1)):
        zi1 = z1[i]
        zi2 = z2[i]
        plt.scatter(x=zi1[0], y=zi1[1], label=f'Predicted Node: {i}')
        plt.scatter(x=zi2[0], y=zi2[1], label=f'Actual Node: {i}')

    plt.legend()
    plt.show()