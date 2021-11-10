import torch
import pandas as pd
import plotly.express as px
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
    # starting from index 10 to get rid of bright color, which is hard to see
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

def animate(model, t_start, t_end, num_of_time_points, device):
    z0 = model.z0.clone().detach()
    v0 = model.v0.clone().detach()
    time_deltas = model.time_deltas.clone().detach()
    step_size = model.step_size.clone().detach()
    num_of_steps = model.num_of_steps

    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
        
    times = torch.linspace(t_start, t_end, num_of_time_points)

    #Calculate how many steps each time point corresponds to
    time_step_ratio = times/step_size
    #Make round down time_step_ratio to find the index of the step which each time fits into
    time_step_indices = torch.floor(time_step_ratio)
    #Make sure times that lands on tn is put into the last time step by subtracting 1 from their step index
    time_step_indices = torch.tensor([t if t < num_of_steps else t-1 for t in time_step_indices])
    #Calculate the remainding time that will be inside the matching step for each time
    remainding_time = (times-time_step_indices*step_size)
    time_step_indices = time_step_indices.tolist()
    #The step positions we will start from for each time point and then use to find their actual position
    z_step_starting_positions = steps_z0[:,:,time_step_indices]
    #Latent Z positions for all times
    step_zt = z_step_starting_positions + v0[:,:,time_step_indices]*remainding_time    

    df = pd.DataFrame({
        'node': [*list(range(step_zt.shape[0]))]*len(times),
        'x': step_zt[:,0,:].T.flatten().tolist(),
        'y': step_zt[:,1,:].T.flatten().tolist(),
        't': [t for t in times.tolist() for _ in list(range(step_zt.shape[0]))]
    })

    fig = px.scatter(df, x='x', y='y', animation_frame='t', animation_group='node', color="node", hover_name="node",
               log_x=False, size_max=20, range_x=[torch.min(step_zt[:,0,:]).item(), torch.max(step_zt[:,0,:]).item()], 
               range_y=[torch.min(step_zt[:,1,:]).item(), torch.max(step_zt[:,1,:]).item()])
    fig.show()


def animate(z0, v0, time_deltas, step_size, num_of_steps, t_start, t_end, num_of_time_points, device):
    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
        
    times = torch.linspace(t_start, t_end, num_of_time_points)

    #Calculate how many steps each time point corresponds to
    time_step_ratio = times/step_size
    #Make round down time_step_ratio to find the index of the step which each time fits into
    time_step_indices = torch.floor(time_step_ratio)
    #Make sure times that lands on tn is put into the last time step by subtracting 1 from their step index
    time_step_indices = torch.tensor([t if t < num_of_steps else t-1 for t in time_step_indices])
    #Calculate the remainding time that will be inside the matching step for each time
    remainding_time = (times-time_step_indices*step_size)
    time_step_indices = time_step_indices.tolist()
    #The step positions we will start from for each time point and then use to find their actual position
    z_step_starting_positions = steps_z0[:,:,time_step_indices]
    #Latent Z positions for all times
    step_zt = z_step_starting_positions + v0[:,:,time_step_indices]*remainding_time    

    df = pd.DataFrame({
        'node': [*list(range(step_zt.shape[0]))]*len(times),
        'x': step_zt[:,0,:].T.flatten().tolist(),
        'y': step_zt[:,1,:].T.flatten().tolist(),
        't': [t for t in times.tolist() for _ in list(range(step_zt.shape[0]))]
    })

    fig = px.scatter(df, x='x', y='y', animation_frame='t', animation_group='node', color="node", hover_name="node",
               log_x=False, size_max=20, range_x=[torch.min(step_zt[:,0,:]).item(), torch.max(step_zt[:,0,:]).item()], 
               range_y=[torch.min(step_zt[:,1,:]).item(), torch.max(step_zt[:,1,:]).item()])
    fig.show()
