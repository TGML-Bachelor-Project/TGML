import enum
import torch 
import plotly
import pandas as pd
from tqdm import tqdm
from itertools import repeat
import plotly.graph_objects as go


def create_animation_node_data(model, interaction_data):
    nodes = []    
    x_positions = []
    y_positions = []
    interaction_times = []
    print('Preprocessing animation node data...')
    for data in tqdm(torch.split(interaction_data, 10000)):
        times = torch.unique(data[:,2])
        step_zt = model.steps(times)
        nodes.extend([str(n) for n in [*list(range(step_zt.shape[0]))]*len(times)])
        x_positions.extend(step_zt[:,0,:].T.flatten().tolist())
        y_positions.extend(step_zt[:,1,:].T.flatten().tolist())
        # Adding time of node positions
        interaction_times.extend([t for t in times.tolist() for _ in list(range(step_zt.shape[0]))])
        

    return nodes, x_positions, y_positions, interaction_times


def create_animation_edge_data(model, interaction_data):
    node1 = []    
    node2 = []    
    node1_x = []
    node1_y = []
    node2_x = []
    node2_y = []
    interaction_times = []
    print('Preprocessing animation edge data...')
    for data in tqdm(torch.split(interaction_data, 10000)):
        times = torch.unique(data[:,2])
        step_zt = model.steps(times)

        for ti, t in enumerate(times):
            t_data = data[data[:,2] == t]
            node1.extend(t_data[:,0].tolist())
            node2.extend(t_data[:,0].tolist())
            node1_x.extend(step_zt[t_data[:,0].long(),0,ti].tolist())
            node1_y.extend(step_zt[t_data[:,0].long(),1,ti].tolist())
            node2_x.extend(step_zt[t_data[:,1].long(),0,ti].tolist())
            node2_y.extend(step_zt[t_data[:,1].long(),1,ti].tolist())
            interaction_times.extend([t.item() for _ in list(range(t_data.shape[0]))])
        

    return node1, node1_x, node1_y, node2, node2_x, node2_y, interaction_times


def create_coordinate_system(fig_dict, xrange, yrange, frame_duration, transition_duration):
    print('Creating animation main layout and coordinate system...')
    fig_dict["layout"]["xaxis"] = {"range": xrange, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'LightGray'}
    fig_dict["layout"]["yaxis"] = {"range": yrange, 'showgrid': True, 'gridwidth': 1, 'gridcolor': 'LightGray'}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["plot_bgcolor"] = 'rgba(0,0,0,0)'
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


def create_edge(edge):
    return {
                "x": [edge['node1_x'], edge['node2_x']],
                "y": [edge['node1_y'], edge['node2_y']],
                "mode": "lines",
                "line": {"width": 2, "color": "cyan"},
                "text": f'({edge["node1"]}, {edge["node2"]})',
                "name": f'({edge["node1"]}, {edge["node2"]})'
            }


def create_frame(grp, node_col, time, x_pos_col, y_pos_col, edge_data):
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
    
    for i, e in edge_data.iterrows():
        frame["data"].append(create_edge(e))
    return frame


def create_animation_frames(fig_dict, sliders_dict, data, node_col, time_col, x_pos_col, y_pos_col, 
                            edge_data, frame_duration, transition_duration):
    print('Creating animation time frames...')

    grp_data_by_time = data.sort_values(time_col, ascending=True).groupby(time_col)
    for time, grp in tqdm(grp_data_by_time):
        t_edges = edge_data[edge_data['time'] == time]
        frame = create_frame(grp, node_col, time, x_pos_col, y_pos_col, t_edges)
        slider_step = {"args": [
            [str(time)],
            {"frame": {"duration": frame_duration, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": transition_duration}}
        ],
            "label": str(round(time,2)),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        fig_dict["frames"].append(frame)

    fig_dict["data"].extend(fig_dict["frames"][0]["data"])
    fig_dict["layout"]["sliders"] = [sliders_dict]
    return fig_dict

def animate(model, interaction_data, wandb_handler):

    nodes, x_positions, y_positions, interaction_times = create_animation_node_data(model=model, interaction_data=interaction_data)

    node_col, time_col, x, y = 'node', 'interaction_time', 'pos_x', 'pos_y' 
    interactions =  pd.DataFrame({
        node_col: nodes,
        x: x_positions, 
        y: y_positions,
        time_col: interaction_times
    })

    # Edges
    node1, node1_x, node1_y, node2, node2_x, node2_y, edge_times = create_animation_edge_data(model, interaction_data)
    edges = pd.DataFrame({
        'node1': node1,
        'node1_x': node1_x,
        'node1_y': node1_y,
        'node2': node2,
        'node2_x': node2_x,
        'node2_y': node2_y,
        'time': edge_times
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
                                        node_col=node_col, time_col=time_col, x_pos_col=x, y_pos_col=y, edge_data=edges,
                                        frame_duration=frame_duration, transition_duration=transition_duration)

    # Create figure
    print('Preparing animation...')
    fig = go.Figure(fig_dict)
    fig.show()

    #wandb_handler.log({'animation': wandb_handler.Html(plotly.io.to_html(fig, auto_play=False))})
