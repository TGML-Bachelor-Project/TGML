import torch 
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def create_coordinate_system(fig_dict, frame_duration, transition_duration):
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": frame_duration, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": transition_duration,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]


def create_time_slider(prefix, transition_duration):
    sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": prefix,
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": transition_duration, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
    }
    
    return sliders_dict


def create_animation_frames(fig_dict, sliders_dict, data, nodes, node1_col, node2_col, time_col, x_pos, y_pos, frame_duration, transition_duration):
    for time in data[time_col]:
        frame = {"data": [], "name": str(time)}
        for node in nodes:
            data_by_time = data[data[time_col] == int(time)]
            data_by_time_and_node = data_by_time[(data_by_time[node1_col] == node) 
                                                | (data_by_time[node2_col] == node)]

            data_dict = {
                "x": list(data_by_time_and_node[x_pos]),
                "y": list(data_by_time_and_node[y_pos]),
                "mode": "markers",
                "text": list(node),
                "color": node,
                "marker": {
                    "sizemode": "area",
                    "sizeref": 200000,
                    "size": [20]*data_by_time_and_node.shape[0]
                },
                "name": node
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [time],
            {"frame": {"duration": frame_duration, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": transition_duration}}
        ],
            "label": time,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    return fig_dict


def update_fig_layout(fig):
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_traces(marker=dict(size=20,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return fig

def animate(model, interaction_data, device, wandb_handler):
    z0 = model.z0.detach().to(device)
    v0 = model.v0.detach().to(device)
    time_deltas = model.time_deltas.detach().to(device)
    step_size = model.step_size.detach().to(device)
    start_times = model.start_times.detach().to(device)

    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
    #times = torch.linspace(t_start, t_end, num_of_time_points).to(device)
        
    x_positions = []
    y_positions = []
    for data in torch.split(interaction_data, 10000):
        times = data[:,2]
        unique_times, unique_time_indices = torch.unique(times, return_inverse=True)

        step_mask = ((unique_times.unsqueeze(1) > start_times) | (start_times == 0).unsqueeze(0))
        step_end_times = step_mask*torch.cumsum(step_mask*step_size, axis=1)
        time_mask = unique_times.unsqueeze(1) <= step_end_times
        time_deltas = (step_size - (step_end_times - unique_times.unsqueeze(1))*time_mask)*step_mask
        movement = torch.sum(v0.unsqueeze(2)*time_deltas, dim=3)
        step_zt = z0.unsqueeze(2) + movement

        positions = step_zt[data[:,0].long(), data[:,1].long(), unique_time_indices]
        x_positions.extend(positions[:,0].tolist())
        y_positions.extend(positions[:,1].tolist())

    node_col, time_col, x, y = 'node', 'interaction_time', 'pos x', 'pos y'
    interactions =  pd.DataFrame({
        node_col: torch.unique(interaction_data[:,:2].long()).detach().numpy(),
        x: x_positions, 
        y: y_positions,
        time_col: interaction_data[:,2]
    })
    nodes = pd.unique(interactions[['node1', 'node2']].values.ravel())

    # Figure where the animation will take place
    fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
    }

    # Creating main layout
    frame_duration = 100
    transition_duration = 1
    create_coordinate_system(fig_dict=fig_dict, frame_duration=frame_duration, transition_duration=transition_duration)
    sliders_dict = create_time_slider(prefix='Interaction Time: ', transition_duration=transition_duration)
    
    # Create animation frames
    fig_dict = create_animation_frames(fig_dict=fig_dict, sliders_dict=sliders_dict, data=interactions, nodes=nodes,
                                        node1_col=node1_col, node2_col=node_col, time_col=time_col, 
                                        frame_duration=frame_duration, transition_duration=transition_duration)

    # Create figure
    fig = go.Figure(fig_dict)
    fig = update_fig_layout(fig)
    fig.show()

    #wandb_handler.log({'animation': wandb_handler.Html(plotly.io.to_html(fig, auto_play=False))})
