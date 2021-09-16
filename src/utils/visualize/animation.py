import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def node_movements(node_positions:list, title:str, trail:bool) -> None:
    '''
    Creates a matplotlib animation of node movements based on
    the collection of time dependent node_positions.

    :param node_positions:  List of collections of node positions for 
                            each recorded time step
    :param title:           Title of the animation
    :param trail:           If true keeps old node_positions in the animation
                            and there by creates a "trail" of the node movements.
                            Otherwise the node movements only show the latest 
                            node positions, so only a sinle dot per node throughout
                            the animation.
    '''
    xs, ys = [], []
    for time_step in node_positions:
        for positions in time_step:
            xs.append(positions[0])
            ys.append(positions[1])

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'ro')
    
    def init():
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), min(ys))
        return ln,
    
    def update(frame):
        xs = [n[0] for n in frame]
        ys = [n[1] for n in frame]

        # Clear data to only plot new node position without trail
        if not trail:
            xdata = []
            ydata = []

        xdata.append(xs)
        ydata.append(ys)
        ln.set_data(xdata, ydata)
        return ln,
    
    ani = FuncAnimation(fig, update, frames=node_positions,
                                    init_func=init, blit=True)
    plt.title(title)
    plt.show()