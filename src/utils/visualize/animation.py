import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    # starting from index 1 to get rid of bright color, which is hard to see
    shift = 10
    node_colors = list(mcolors.CSS4_COLORS.keys())[shift:(len(node_positions[0])+shift)]
    xs, ys = [], []
    for time_step in node_positions:
        for positions in time_step:
            xs.append(positions[0])
            ys.append(positions[1])

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln = [plt.plot([], [], 'o', label=f'Node {i}', color=node_colors[i])[0] for i in range(len(node_positions[0]))]
    
    def init():
        # Setting the limit a little larger to not cut of nodes at the edge of plot
        ax.set_xlim(min(xs)-0.1*abs(min(xs)), max(xs)+0.1*abs(max(xs)))
        ax.set_ylim(min(ys)-0.1*abs(min(ys)), max(ys)+0.1*abs(max(ys)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return ln
    
    def update(frame):
        xs = [n[0] for n in frame]
        ys = [n[1] for n in frame]

        # Clear data to only plot new node position without trail
        if trail:
            for i in len(xdata):
                for j in range(len(ln)):
                    ln[j].set_data(xdata[i][j], ydata[i][j])

            # Added current x and y to trail history
            xdata.append(xs)
            ydata.append(ys)
        
        # Add current x and y pos to plot data
        for i in range(len(xs)):
            ln[i].set_data(xs[i], ys[i])

        return ln 
    
    ani = FuncAnimation(fig, update, frames=node_positions,
                                    init_func=init, blit=True)
    plt.title(title)
    plt.legend()
    plt.show()