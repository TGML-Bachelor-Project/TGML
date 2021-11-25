import torch 
import plotly
import pandas as pd
from tqdm import tqdm
from itertools import repeat
import plotly.graph_objects as go

def create_animation_data(model, interaction_data, device):
    z0 = model.z0.detach().to(device)
    v0 = model.v0.detach().to(device)
    time_deltas = model.time_deltas.detach().to(device)
    step_size = model.step_size.detach().to(device)
    start_times = model.start_times.detach().to(device)

    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
    #times = torch.linspace(t_start, t_end, num_of_time_points).to(device)

    nodes = []    
    x_positions = []
    y_positions = []
    interaction_times = []
    interacts_with = []
    print('Preprocessing animation data...')
    for data in tqdm(torch.split(interaction_data, 10000)):
        times = data[:,2]
        unique_times, unique_time_indices = torch.unique(times, return_inverse=True)

        step_mask = ((unique_times.unsqueeze(1) > start_times) | (start_times == 0).unsqueeze(0))
        step_end_times = step_mask*torch.cumsum(step_mask*step_size, axis=1)
        time_mask = unique_times.unsqueeze(1) <= step_end_times
        time_deltas = (step_size - (step_end_times - unique_times.unsqueeze(1))*time_mask)*step_mask
        movement = torch.sum(v0.unsqueeze(2)*time_deltas, dim=3)
        step_zt = z0.unsqueeze(2) + movement

        # Adding nodes
        all_nodes = list(range(step_zt.shape[0]))
        nodes.extend(all_nodes*unique_time_indices.shape[0])
        # Adding node positions
        x_positions.extend(step_zt[:,0, unique_time_indices].flatten().tolist())
        y_positions.extend(step_zt[:,1,unique_time_indices].flatten().tolist())
        # Adding time of node positions
        interaction_times.extend([t for ut in unique_times[unique_time_indices].tolist() for t in repeat(ut, step_zt.shape[0])])


    return nodes, x_positions, y_positions, interaction_times


def animate(model, interaction_data, device, wandb_handler):

    nodes, x_positions, y_positions, interaction_times = create_animation_data(model=model, 
                                                            interaction_data=interaction_data, device=device)

    node_col, time_col, x, y, size_col = 'node', 'interaction_time', 'pos x', 'pos y', 'size'
    dataset =  pd.DataFrame({
        node_col: nodes,
        x: x_positions, 
        y: y_positions,
        time_col: interaction_times,
        size_col: [20]*len(x_positions)
    })

    xrange = [dataset[x].min(), dataset[x].max()]
    yrange = [dataset[y].min(), dataset[y].max()]

    # make list of continents
    continents = []
    for continent in dataset["node"]:
        if continent not in continents:
            continents.append(continent)
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": xrange, "title": "x"}
    fig_dict["layout"]["yaxis"] = {"range": yrange, "title": "y"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
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

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 1, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    time = dataset[time_col].iloc[0]
    for continent in continents:
        dataset_by_year = dataset[dataset[time_col] == time]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["node"] == continent]

        data_dict = {
            "x": list(dataset_by_year_and_cont[x]),
            "y": list(dataset_by_year_and_cont[y]),
            "mode": "markers",
            "text": list(dataset_by_year_and_cont[node_col]),
            "marker": {
                "sizemode": "area",
                "sizeref": 200000,
                "size": dataset_by_year_and_cont[size_col]
            },
            "name": continent
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for time in dataset[time_col].unique():
        frame = {"data": [], "name": str(time)}
        for continent in continents:
            dataset_by_year = dataset[dataset[time_col] == int(time)]
            dataset_by_year_and_cont = dataset_by_year[
                dataset_by_year[node_col] == continent]

            data_dict = {
                "x": list(dataset_by_year_and_cont[x]),
                "y": list(dataset_by_year_and_cont[y]),
                "mode": "markers",
                "text": list(dataset_by_year_and_cont[node_col]),
                "marker": {
                    "sizemode": "area",
                    "sizeref": 200000,
                    "size": dataset_by_year_and_cont[size_col]
                },
                "name": continent
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [time],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": time,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.show()


