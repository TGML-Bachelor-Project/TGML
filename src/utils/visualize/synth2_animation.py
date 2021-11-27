
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import torch
import numpy as np

import plotly.express as px


def animate_nomodel_lyon(z0, v0, time_deltas, step_size, num_of_steps, t_start, t_end, num_of_time_points, device):
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
        'node': [str(n) for n in [*list(range(step_zt.shape[0]))]*len(times)],
        'x': step_zt[:,0,:].T.flatten().tolist(),
        'y': step_zt[:,1,:].T.flatten().tolist(),
        't': [t for t in times.tolist() for _ in list(range(step_zt.shape[0]))]
    })

    fig = px.scatter(df, x='x', y='y', animation_frame='t', animation_group='node', color="node",
               log_x=False, size_max=20,
               range_x=[torch.min(step_zt[:,0,:]).item(), torch.max(step_zt[:,0,:]).item()], 
               range_y=[torch.min(step_zt[:,1,:]).item(), torch.max(step_zt[:,1,:]).item()])
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_traces(marker=dict(size=20,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

    fig.show()
    # fig.write_html(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'animations', 'latest_animation.html'), auto_play=False)



if __name__ == '__main__':

    z0 = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.], [0., 2.]])
    v0 = torch.tensor([
                    [[-0.15, 0.15, -0.3, 0.3, -0.01, -0.3, 0.3, -0.2, 0.1, -0.15], 
                    [-0.15, 0.15, -0.3, 0.3, -0.5, 0.01, 0.8, -0.4, -0.1, 0.05]
                    ],
                    [[-0.15, 0.15, -0.3, 0.3, -0.01, -0.3, 0.3, -0.2, 0.1, -0.15],
                    [0.15, -0.15, 0.3, -0.3, 0.5, -0.01, -0.8, 0.4, 0.1, -0.05] 
                    ],
                    [[0.15, -0.15, 0.3, -0.3, 0.01, 0.3, -0.3, 0.2, -0.1, 0.15],
                    [-0.15, 0.15, -0.3, 0.3, -0.5, 0.01, 0.8, -0.4, -0.1, -0.05] 
                    ],
                    [[0.15, -0.15, 0.3, -0.3, 0.01, 0.3, -0.3, 0.2, -0.1, 0.15], 
                    [0.15, -0.15, 0.3, -0.3, 0.5, -0.01, -0.8, 0.4, 0.1, 0.05] 
                    ],
                    [[0.05, -0.1, 0.1, -0.05, 0.2, -0.2, 0.01, 0.01, -0.1, 0.1],
                    [-0.3, -0.05, 0.4, -0.3, -0.05, -0.05, 0.01, 0.01, 0.01, 0.05] 
                    ]])
    max_time = 50


    time_intervals = torch.linspace(0, max_time, v0.shape[2] + 1)
    start_times = time_intervals[:-1]
    end_times = time_intervals[1:]
    time_intervals = list(zip(start_times.tolist(), end_times.tolist()))
    time_deltas = (end_times-start_times)
    # All deltas should be equal do to linspace, so we can take the first
    step_size = time_deltas[0]
    animate_nomodel_lyon(z0=z0, v0=v0, time_deltas=time_deltas, step_size=step_size, num_of_steps=v0.shape[2], t_start=0, t_end=max_time, num_of_time_points=500, device=None)

