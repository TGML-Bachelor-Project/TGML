import torch 
import plotly
import pandas as pd
from tqdm import tqdm
from itertools import repeat
import plotly.graph_objects as go

def create_animation_data(model, interaction_data, device):
    nodes = []    
    x_positions = []
    y_positions = []
    interaction_times = []
    interacts_with = []
    print('Preprocessing animation data...')
    for data in tqdm(torch.split(interaction_data, 10000)):
        times = torch.unique(data[:,2])
        step_zt = model.steps(times)
        nodes.extend(list(range(step_zt.shape[0]))*len(times))
        x_positions.extend(step_zt[:,0,:].flatten().tolist())
        y_positions.extend(step_zt[:,1,:].flatten().tolist())
        # Adding time of node positions
        interaction_times.extend([t for time in times.tolist() for t in repeat(time, step_zt.shape[0])])


    return nodes, x_positions, y_positions, interaction_times


def create_coordinate_system(fig_dict, xrange, yrange, frame_duration, transition_duration):
    print('Creating animation main layout and coordinate system...')
    fig_dict["layout"]["xaxis"] = {"range": xrange}
    fig_dict["layout"]["yaxis"] = {"range": yrange}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["plot_bgcolor"] = 'rgba(0,0,0,0)'
    fig_dict["layout"]["xaxis"] = { 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'LightGray'}
    fig_dict["yaxis"]= {'showgrid': True, 'gridwidth': 1, 'gridcolor': 'LightGray'}
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


def create_frame(grp, node_col, time, x_pos_col, y_pos_col):
    frame = {"data": [], "name": str(time)}
    for node, node_grp in grp.groupby(node_col):
        frame["data"].append(
            {
                "x": list(node_grp[x_pos_col]),
                "y": list(node_grp[y_pos_col]),
                "mode": "markers",
                "text": list(node_grp[node_col]),
                "marker": {
                    "size": [20]*node_grp.shape[0],
                    'line': {'width': 2,
                            'color': 'DarkSlateGrey'}
                },
                "name": str(node)
            }
        )
    return frame


def create_animation_frames(fig_dict, sliders_dict, data, node_col, time_col, x_pos_col, y_pos_col, frame_duration, transition_duration):
    print('Creating animation time frames...')

    grp_data_by_time = data.sort_values(time_col, ascending=True).groupby(time_col)
    for time, grp in tqdm(grp_data_by_time):
        frame = create_frame(grp, node_col, time, x_pos_col, y_pos_col)
        slider_step = {"args": [
            [str(time)],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": str(time),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        fig_dict["frames"].append(frame)

    fig_dict["data"].extend(fig_dict["frames"][0]["data"])
    fig_dict["layout"]["sliders"] = [sliders_dict]
    return fig_dict

def animate(model, interaction_data, device, wandb_handler):

    nodes, x_positions, y_positions, interaction_times = create_animation_data(model=model, 
                                                            interaction_data=interaction_data, device=device)

    node_col, time_col, x, y = 'node', 'interaction_time', 'pos x', 'pos y'
    interactions =  pd.DataFrame({
        node_col: nodes,
        x: x_positions, 
        y: y_positions,
        time_col: interaction_times
    })
    # Figure where the animation will take place
    fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
    }

    # Creating main layout
    xrange = [interactions[x].min(), interactions[x].max()]
    yrange = [interactions[y].min(), interactions[y].max()]
    frame_duration = 100
    transition_duration = 1 
    create_coordinate_system(fig_dict=fig_dict, xrange=xrange, yrange=yrange, 
                            frame_duration=frame_duration, transition_duration=transition_duration)
    sliders_dict = create_time_slider(prefix='Interaction Time: ', transition_duration=transition_duration)

    # Create animation frames
    fig_dict = create_animation_frames(fig_dict=fig_dict, sliders_dict=sliders_dict, data=interactions, 
                                        node_col=node_col, time_col=time_col, x_pos_col=x, y_pos_col=y, 
                                        frame_duration=frame_duration, transition_duration=transition_duration)

    # Create figure
    print('Preparing animation...')
    fig = go.Figure(fig_dict)
    fig.show()

    #wandb_handler.log({'animation': wandb_handler.Html(plotly.io.to_html(fig, auto_play=False))})
