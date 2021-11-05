import os
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
from itertools import cycle
import plotly
import requests
import plotly.express as px


def source_API_output(path):

    # Determine if path is api path or data file path
    # Case 1: API path
    if 'http://' in path:
        data = requests.get('http://localhost:5000/locations/20/inputs/17/outputs').json()
    # Case 2: Data file path
    else:
        with open(path, 'r') as data_file:
            data = json.load(data_file)

    input_batch_id = (data.get('input_batch_id'))
    location_id = (data.get('location_id'))
    positions = (data.get('positions'))

    positions_df = pd.DataFrame(positions)

    return(input_batch_id, location_id, positions_df)

def deprec_visualize_3D_movement_data(API_endpoint_path, positions_df, input_batch_id, location_id):

    # TODO: potentially integrate location data to be able to scale plot correctly with location (x,y,z) boundaries
    #location_data = requests.get(API_endpoint_path + '/locations/' + str(location_id)).json()

    x = positions_df['x']
    y = positions_df['y']
    z = positions_df['z']

    colors = cycle(plotly.colors.sequential.Viridis)

    fig = go.Figure()
    for s in positions_df['object_identifier'].unique():
        df = positions_df[positions_df['object_identifier'] == s]
        fig.add_trace(go.Scatter3d(x=[0], y = [0], z = [0],
                                mode = 'markers',
                                name = s,
                                marker_color = next(colors)))
    
    frames = [go.Frame(data= [go.Scatter3d(
                                       x=x[[k]], 
                                       y=y[[k]],
                                       z=z[[k]])],
                   
                   traces= [0],
                   name=f'frame{k}'      
                  )for k  in  range(len(x)-1)]

    fig.update(frames = frames)

    fig.update_layout(updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, dict(frame=dict(redraw=True,fromcurrent=True, mode='immediate'))
                                        ])])])

    fig.update_layout(
        
        scene = dict(
        
        xaxis=dict(range=[min(x)-0.2*min(x), max(x)+0.2*max(x)], autorange=False),
        yaxis=dict(range=[min(y)-0.2*min(y), max(y)+0.2*max(y)], autorange=False),
        zaxis=dict(range=[min(z)-0.2*min(z), max(z)+0.2*max(z)], autorange=False),
        ),
        title='Visualization of object movement over time'
        )

    # sliders = [dict(
    # active=10,
    # currentvalue={"prefix": "Frequency: "},
    # pad={"t": 50},
    # steps=positions_df['timestamp']
    # )]

    # fig.update_layout(
    #     sliders=sliders
    # )

    
    fig.write_html("file.html")

    return None


    

def visualize_3D_movement_data(dataset):

    fig = px.scatter_3d(dataset, x = 'x', y = 'y', z = 'z', animation_frame='timestamp',
           color="object_identifier", hover_name="object_identifier",
           range_x=[min(dataset['x'])*0.9,max(dataset['x'])*1.1], 
           range_y=[min(dataset['y'])*0.9,max(dataset['y'])*1.1],
           range_z=[min(dataset['z'])*0.9,max(dataset['z'])*1.1])

    fig.write_html('scripts/data_visualization/assets/generated_graphs/visualization_3D_movement_data.html')  


def visualize_2D_movement_data(dataset):

    fig = px.scatter(dataset, x = 'x', y = 'y', animation_frame='timestamp',
           color="object_identifier", hover_name="object_identifier",
           range_x=[min(dataset['x'])*0.9,max(dataset['x'])*1.1], 
           range_y=[min(dataset['y'])*0.9,max(dataset['y'])*1.1])

    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

    fig.write_html('scripts/data_visualization/assets/generated_graphs/visualization_2D_movement_data.html')  



testfile_path = str(os.getcwd()) + '/scripts/data_visualization/assets/test_data/test_data.json'

API_path = 'http://localhost:5000/locations/20/inputs/17/outputs'


visualize_3D_movement_data(source_API_output(testfile_path)[2])


#deprec_visualize_3D_movement_data('http://localhost:5000/', API_output_df[2], API_output_df[0], API_output_df[1])
