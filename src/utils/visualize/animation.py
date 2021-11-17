import os
import torch
import plotly
import pandas as pd
import plotly.express as px

def animate(model, t_start, t_end, num_of_time_points, device, wandb_handler):
    z0 = model.z0.clone().detach()
    v0 = model.v0.clone().detach()
    time_deltas = model.time_deltas.clone().detach()
    step_size = model.step_size.clone().detach()
    start_times = model.start_times

    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
        
    times = torch.linspace(t_start, t_end, num_of_time_points).to(device)

    step_mask = ((times.unsqueeze(1) > start_times) | (start_times == 0).unsqueeze(0))
    step_end_times = step_mask*torch.cumsum(step_mask*step_size, axis=1)
    time_mask = times.unsqueeze(1) <= step_end_times
    time_deltas = (step_size - (step_end_times - times.unsqueeze(1))*time_mask)*step_mask
    movement = torch.sum(v0.unsqueeze(2)*time_deltas, dim=3)
    step_zt = z0.unsqueeze(2) + movement

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

    wandb_handler.log({'animation': wandb_handler.Html(plotly.io.to_html(fig, auto_play=False))})


def animate_nomodel(z0, v0, time_deltas, step_size, start_times, t_start, t_end, num_of_time_points, device):
    # Starting positions for each model step
    steps_z0 = z0.unsqueeze(2) + torch.cumsum(v0*time_deltas, dim=2)
    steps_z0 = torch.cat((z0.unsqueeze(2), steps_z0), dim=2)
        
    times = torch.linspace(t_start, t_end, num_of_time_points)

    step_mask = ((times.unsqueeze(1) > start_times) | (start_times == 0).unsqueeze(0))
    step_end_times = step_mask*torch.cumsum(step_mask*step_size, axis=1)
    time_mask = times.unsqueeze(1) <= step_end_times
    time_deltas = (step_size - (step_end_times - times.unsqueeze(1))*time_mask)*step_mask
    movement = torch.sum(v0.unsqueeze(2)*time_deltas, dim=3)
    step_zt = z0.unsqueeze(2) + movement

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
    fig.write_html(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'animations', 'latest_animation.html'), auto_play=False)

def animate_nomodel_lyon(z0, v0, time_deltas, step_size, num_of_steps, t_start, t_end, num_of_time_points, device, metadata):
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
        'class': [str(metadata[str(n)]) for n in [*list(range(step_zt.shape[0]))]*len(times)],
        'x': step_zt[:,0,:].T.flatten().tolist(),
        'y': step_zt[:,1,:].T.flatten().tolist(),
        't': [t for t in times.tolist() for _ in list(range(step_zt.shape[0]))]
    })

    fig = px.scatter(df, x='x', y='y', animation_frame='t', animation_group='node', color="class", color_discrete_sequence=px.colors.qualitative.Light24,
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
    fig.write_html(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'animations', 'latest_animation.html', auto_play=False))
    # fig.write_html('/home/augustsemrau/drive/bachelor/TGML/animations_rl_data/lyon_full_lr0025_steps240.html', auto_play=False)
