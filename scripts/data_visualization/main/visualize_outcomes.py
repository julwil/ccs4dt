import os
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
from itertools import cycle
import plotly
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import datetime

# TODO: Write documentation
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

# TODO: Remove, as function is now split into two functions below
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

    
# TODO: Write documentation
def visualize_3D_movement_data(dataset):

    fig = px.scatter_3d(dataset, x = 'x', y = 'y', z = 'z', animation_frame='timestamp',
           color="object_identifier", hover_name="object_identifier",
           range_x=[min(dataset['x'])*0.9,max(dataset['x'])*1.1], ## +/- 10% in every direction
           range_y=[min(dataset['y'])*0.9,max(dataset['y'])*1.1], ## +/- 10% in every direction
           range_z=[min(dataset['z'])*0.9,max(dataset['z'])*1.1]) ## +/- 10% in every direction

    fig.write_html('scripts/data_visualization/assets/generated_graphs/visualization_3D_movement_data.html')  

# TODO: Write documentation
def visualize_2D_movement_data(dataset):

    dataset['true_id'] = dataset['occupant_id']

    fig = px.scatter(dataset, x = 'x_original', y = 'y_original', animation_frame = 'timestamp', animation_group = 'true_id', 
           color = 'occupant_id', hover_name = 'true_id',
           range_x = [min(dataset['x_original'])*0.9,max(dataset['x_original'])*1.1], ## +/- 10% in every direction
           range_y = [min(dataset['y_original'])*0.9,max(dataset['y_original'])*1.1]) ## +/- 10% in every direction
    
    fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

    fig.write_html('scripts/data_visualization/assets/generated_graphs/visualization_2D_movement_data_new.html')  
 
    predicted_elements_only_position_dataframe = dataset.copy()

    
    predicted_elements_only_position_dataframe['x_original'] = predicted_elements_only_position_dataframe['predicted_x']
    predicted_elements_only_position_dataframe['y_original'] = predicted_elements_only_position_dataframe['predicted_y']

    predicted_elements_only_position_dataframe['true_id'] = predicted_elements_only_position_dataframe['pred_obj_id']

    full_dataset = dataset.append(predicted_elements_only_position_dataframe)

    full_dataset.to_excel('testmax3.xlsx')

    fig2 = px.scatter(full_dataset, x = 'x_original', y = 'y_original', animation_frame = 'timestamp', animation_group = 'true_id', 
           color = 'occupant_id', hover_name = 'true_id',
           range_x = [min(full_dataset['x_original'])*0.9,max(full_dataset['x_original'])*1.1], ## +/- 10% in every direction
           range_y = [min(full_dataset['y_original'])*0.9,max(full_dataset['y_original'])*1.1]) ## +/- 10% in every direction
    
    fig2.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

    fig2.write_html('scripts/data_visualization/assets/generated_graphs/visualization_2D_movement_data_new_true_and_predicted.html')  


# TODO: Write documentation
def visualize_all_movement_data_static(filepath):

    dataset = pd.read_csv(filepath, nrows = 20000, header = 0)

    fig = px.scatter(dataset, x = 'x', y = 'y', opacity = 0.2, color_discrete_sequence=['grey'],
           range_x = [min(dataset['x'])*0.5,max(dataset['x'])*2.0], ## +/- 10% in every direction
           range_y = [min(dataset['y'])*0.5,max(dataset['y'])*2.0]) ## +/- 10% in every direction

    fig.update_traces(marker=dict(size=12,
                            line=dict(width=2,
                                    color='DarkSlateGrey')),
                selector=dict(mode='markers'))
    
    # # Sensor 1 - WiFi 2.4GHz 20m range
    s1_x, s1_y, s1_reach = (200, 200, 2000)
    fig.add_trace(go.Scatter(x= [s1_x], y=[s1_y], name = 'Sensor 1 WiFi 2.4GHz', mode='markers', marker=dict(color='rgb(238,234,252)', size=[20], line=dict(width= 4, color='red'))))

    fig.add_shape(type="circle",
        xref="x", yref="y",
        fillcolor='rgb(238,234,252)',
        x0 = s1_x-0.5*s1_reach, y0 = s1_y-0.5*s1_reach, x1 = s1_x+0.5*s1_reach, y1 = s1_y+0.5*s1_reach,
        line=dict(color='black', width = 5, dash= 'dashdot'), opacity = 0.1
    )

    # # Sensor 2 - RFID 1m range
    # s2_x, s2_y, s2_reach = (330, 260, 100)
    # fig.add_trace(go.Scatter(x= [s2_x], y=[s2_y], name = 'Sensor 2 RFID', mode='markers', marker=dict(color='rgb(165,188,212)', size=[20], line=dict(width= 4, color='red'))))


    # fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     fillcolor='rgb(165,188,212)',
    #     x0 = s2_x-0.5*s2_reach, y0 = s2_y-0.5*s2_reach, x1 = s2_x+0.5*s2_reach, y1 = s2_y+0.5*s2_reach,
    #     line=dict(color='black', width = 5, dash= 'dashdot'), opacity = 0.3)

    # # Sensor 3 - Camera 3m range
    # s3_x, s3_y, s3_reach = (400, 100, 300)
    # fig.add_trace(go.Scatter(x= [s3_x], y=[s3_y], name = 'Sensor 3 camera', mode='markers', marker=dict(color='rgb(0,109,219)', size=[20], line=dict(width= 4, color='red'))))

    # fig.add_shape(type="circle", 
    #     xref="x", yref="y",
    #     fillcolor='rgb(0,109,219)',
    #     x0 = s3_x-0.5*s3_reach, y0 = s3_y-0.5*s3_reach, x1 = s3_x+0.5*s3_reach, y1 = s3_y+0.5*s3_reach,
    #     line=dict(color='black', width = 5, dash= 'dashdot'), opacity = 0.1
    # )


    # # Sensor 4 - WiFi 5GHz 4.5m range
    # s4_x, s4_y, s4_reach = (400, 300, 450)
    # fig.add_trace(go.Scatter(x= [s4_x], y=[s4_y], name = 'Sensor 4 WiFi 5GHz', mode='markers', marker=dict(color='rgb(0,73,73)', size=[20], line=dict(width= 4, color='red'))))

    # fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     fillcolor='rgb(0,73,73)',
    #     x0 = s4_x-0.5*s4_reach, y0 = s4_y-0.5*s4_reach, x1 = s4_x+0.5*s4_reach, y1 = s4_y+0.5*s4_reach,
    #     line=dict(color='black', width = 5, dash= 'dashdot'), opacity = 0.1
    # )



    # # Sensor 5 - Bluetooth 17m range
    # s5_x, s5_y, s5_reach = (500, 500, 1700)
    # fig.add_trace(go.Scatter(x= [s5_x], y=[s5_y], name = 'Sensor 5 Bluetooth', mode='markers', marker=dict(color='rgb(0,146,146)', size=[20], line=dict(width= 4, color='red'))))

    # fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     fillcolor='rgb(0,146,146)',
    #     x0 = s5_x-0.5*s5_reach, y0 = s5_y-0.5*s5_reach, x1 = s5_x+0.5*s5_reach, y1 = s5_y+0.5*s5_reach,
    #     line=dict(color='black', width = 5, dash= 'dashdot'), opacity = 0.1
    # )



    fig.write_html('scripts/data_visualization/assets/generated_graphs/visualization_2D_all_movement_data_static.html')

       


testfile_path = str(os.getcwd()) + '/scripts/synthetic_data_generation/assets/sampledata/occupancy_presence_and_trajectories.csv'

API_path = 'http://localhost:5000/locations/20/inputs/17/outputs'


visualize_all_movement_data_static(testfile_path)

#visualize_2D_movement_data(source_API_output(testfile_path)[2])


#deprec_visualize_3D_movement_data('http://localhost:5000/', API_output_df[2], API_output_df[0], API_output_df[1])
