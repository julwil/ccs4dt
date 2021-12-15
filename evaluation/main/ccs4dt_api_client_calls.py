from os import close
import requests
import json
import pandas as pd
import openpyxl
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

# Project internal imports
from requests import api
from scripts.synthetic_data_generation.main.transform_test_data_set import CoordinateSystem, Sensor, Location, simulate_sensor_measurement_for_multiple_sensors

# TODO: Write documentation
class APIClient(object):

    def __init__(self) -> None:
        super().__init__()

    # TODO: Write documentation
    # TODO: Request not correct, validate with Julius
    def API_post_input_batch_call(API_endpoint_path, payload, location_id):
        
        response = requests.post(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs', json = json.loads(payload))

        # 202 Code with successful delivery
        response = response.json() if response and response.status_code == 202 else None

        return (response, response['id'], response['status'], response['location_id'])

    # TODO: Write documentation
    def API_get_input_batch_by_id_call(API_endpoint_path, location_id, input_batch_id):
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs/' + str(input_batch_id))

        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    # TODO: Write documentation
    def API_get_all_input_batches_call(API_endpoint_path, location_id):
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs')

        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    # TODO: Write documentation
    def API_get_location_by_id_call(API_endpoint_path, location_id):
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id))
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    # TODO: Write documentation
    def API_get_all_locations_call(API_endpoint_path):
        
        response = requests.get(API_endpoint_path + '/locations')
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    # TODO: Write documentation
    def API_post_new_location_call(API_endpoint_path, payload):
        
        response = requests.post(API_endpoint_path + '/locations', json = json.loads(payload))

        # 201 Code with successful delivery
        response = response.json() if response and response.status_code == 201 else None

        return (response, response['id'], response['name'])

    # TODO: Write documentation
    def API_get_output_batch_call(API_endpoint_path, location_id, batch_id):
        
        response = requests.get(API_endpoint_path + '/locations/' + str(location_id) + '/inputs/' + str(batch_id) + '/outputs')
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    # TODO: Write documentation
    def convert_sensor_measurements_to_api_conform_payload(dataframe, additional_file_generation = False): 

        dataframe = dataframe[['object_id','x_measured_rel_pos','y_measured_rel_pos','z_measured_rel_pos','sensor_type','sensor_id','date_time']]

        dataframe = dataframe.rename(columns = {'object_id':'object_identifier', 'x_measured_rel_pos':'x', 'y_measured_rel_pos':'y', 'z_measured_rel_pos':'z',
        'sensor_id':'sensor_identifier', 'sensor_type':'sensor_type', 'date_time':'timestamp'})

        if additional_file_generation == True:
            json_data = dataframe.to_json(path_or_buf= 'evaluation/assets/generated_files/synthetic_measurements.json', default_handler=str, orient='records')
            json_data = dataframe.to_json(default_handler=str, orient='records')
        elif additional_file_generation == False:
            json_data = dataframe.to_json(default_handler=str, orient='records')

        return(json_data)

    def end_to_end_API_test(location, sensors, api_endpoint_path, measurement_points = 250, return_simulation_output = True, print_progress = True, store_measurement_data_in_json_file = True, debugger_files = False, identifier_randomization_method = 'random', identifier_type = 'mac-address'):

        # API request: GET all locations
        APIClient.API_get_all_locations_call(api_endpoint_path)

        # Construct test location payload
        location_payload = location.construct_json_payload()

        # If debugger attribute is set, store API data in file
        if(debugger_files):
            with open('evaluation/assets/generated_files/location_payload.json', 'w') as text_file:
                text_file.write(location_payload)
            

        # API request: POST new location
        post_location_response, location_id, test_location_name = (APIClient.API_post_new_location_call(api_endpoint_path, location_payload))

        # Simulate sensor data measurement
        measurement_data = simulate_sensor_measurement_for_multiple_sensors(sensors, measurement_points, identifier_randomization_method = identifier_randomization_method, identifier_type = identifier_type)

        # Generate synthetic measurement data payload
        API_payload = APIClient.convert_sensor_measurements_to_api_conform_payload(measurement_data, additional_file_generation = store_measurement_data_in_json_file)

        if(debugger_files):
            with open('evaluation/assets/generated_files/synthetic_measurement_payload.json', 'w') as text_file:
                text_file.write(API_payload)

        # API request: POST new input batch
        post_input_batch_response, input_batch_id, input_batch_status, location_id_for_input_batch = APIClient.API_post_input_batch_call(api_endpoint_path, API_payload, location_id)

        if print_progress == True:
            # API request: GET input batch status
            print('Get input batch by id')

            print('\n')

            print(APIClient.API_get_input_batch_by_id_call(api_endpoint_path, location_id_for_input_batch, input_batch_id))

            print('\n')
            print('\n')

        # API request: GET output batch based on generated id
        # Pause to let API process / check status before proceeding
        import time

        # Pause for 1 second before requesting result again
        while True:
            time.sleep(1)

            # Check batch status
            status = (APIClient.API_get_input_batch_by_id_call(api_endpoint_path, location_id_for_input_batch, input_batch_id).get('status'))

        
        # Proceed with end-to-end test if status is set to 'finished'
            if status == 'finished':
                break

        
        
        output_batch_response = (APIClient.API_get_output_batch_call(api_endpoint_path, location_id_for_input_batch, input_batch_id))

        if print_progress == True:
            print('\n')
            print('Get output batch by id')
            print(output_batch_response)

            print('\n')
            print('\n')

        if return_simulation_output == True:
            return output_batch_response, measurement_data
        else:
            return output_batch_response                               

# TODO: Write documentation
class PredictionEvaluator(object):


    def __init__(self, api_output, measurement_data):
        
        self.api_output = api_output
        self.measurement_data =  measurement_data
        self.location_id = self.api_output.pop('location_id')
        self.input_batch_id = self.api_output.pop('input_batch_id')
        self.predicted_positions = self.api_output

        # Remove object identifier mappings from prediction results and store in seperate attribute
        self.object_identifier_mappings = self.api_output.pop('object_identifier_mappings')

    # TODO: Write documentation
    def get_prediction_results(self):
        return(self.predicted_positions)

    # TODO: Write documentation
    def get_object_identifier_mappings(self):
        return(self.object_identifier_mappings)

    # TODO: Write documentation
    def add_object_identifier_mapping_to_measurement_dataframe(self):

        pred_obj_id_list = []
        sensor_id_list = []
        pred_initial_obj_id_list = []

        for pred_obj_id in self.object_identifier_mappings:
            for combined_sensor_id_and_initial_object_id in self.object_identifier_mappings[pred_obj_id]:
                input_namings_array = combined_sensor_id_and_initial_object_id.split("___", 1)
                pred_obj_id_list.append(pred_obj_id)
                sensor_id_list.append(input_namings_array[1])
                pred_initial_obj_id_list.append(input_namings_array[0])

        object_identifier_dataframe = pd.DataFrame({'sensor_id_from_api_output': sensor_id_list,
                   'pred_obj_id': pred_obj_id_list,
                   'pred_initial_obj_id': pred_initial_obj_id_list})
        
        measurement_df = pd.DataFrame(self.measurement_data)

        # Type conversions to ensure that merging works correctly
        measurement_df['sensor_id'] = measurement_df['sensor_id'].astype(str)
        measurement_df['object_id'] = measurement_df['object_id'].astype(str)
        object_identifier_dataframe['sensor_id_from_api_output'] = object_identifier_dataframe['sensor_id_from_api_output'].astype(str)
        object_identifier_dataframe['pred_initial_obj_id'] = object_identifier_dataframe['pred_initial_obj_id'].astype(str)

        merged_identifier_df = pd.merge(measurement_df, object_identifier_dataframe, left_on = ['sensor_id','object_id'], right_on = ['sensor_id_from_api_output','pred_initial_obj_id'], how= 'left')

        # Error check for correct mapping
        if(merged_identifier_df.empty):
            raise ValueError('Mapping dataframe is empty')

        return merged_identifier_df

    # TODO: Write documentation
    def add_prediction_data_to_merged_identifier_dataframe(self, debugger_files = False):

        merged_identifier_dataframe = self.add_object_identifier_mapping_to_measurement_dataframe()

        

        pred_obj_id_list = []
        pred_timestamp_list = []
        pred_confidence_list = []
        pred_x_list = []
        pred_y_list = []
        pred_z_list = []


        for i in self.predicted_positions['positions']:
            pred_obj_id_list.append(i['object_identifier'].split("___", 1)[0])
            pred_timestamp_list.append(i['timestamp'])
            pred_confidence_list.append(i['confidence'])
            pred_x_list.append(i['x'])
            pred_y_list.append(i['y'])
            pred_z_list.append(i['z'])

    
        prediction_df = pd.DataFrame({'temp_pred_obj_identifier': pred_obj_id_list,
                   'prediction_timestamp': pred_timestamp_list,
                   'prediction_confidence': pred_confidence_list,
                   'predicted_x': pred_x_list,
                   'predicted_y': pred_y_list,
                   'predicted_z': pred_z_list,
                   })

        # Type conversions to ensure merge is working
        prediction_df['prediction_timestamp'] = prediction_df['prediction_timestamp'].astype(float)
        prediction_df['temp_pred_obj_identifier'] = prediction_df['temp_pred_obj_identifier'].astype(str)
        merged_identifier_dataframe['pred_obj_id'] = merged_identifier_dataframe['pred_obj_id'].astype(str)



        # Sort dataframes
        merged_identifier_dataframe = merged_identifier_dataframe.sort_values(by='timestamp')
        prediction_df = prediction_df.sort_values(by='prediction_timestamp')
    

        if(debugger_files):
            merged_identifier_dataframe.to_excel('evaluation/assets/generated_files/merged_identifier_dataframe.xlsx')
            prediction_df.to_excel('evaluation/assets/generated_files/prediction_df.xlsx')


        merged_prediction_df = pd.merge_asof(merged_identifier_dataframe, prediction_df, left_on=['timestamp'], right_on = ['prediction_timestamp'], left_by=['object_id'], right_by=['temp_pred_obj_identifier'],direction='nearest')

        return merged_prediction_df

    # TODO: Write documentation
    def calculate_object_matching_accuracy(self):

        dataframe_with_matched_object_ids = self.add_object_identifier_mapping_to_measurement_dataframe()

        #dataframe_with_matched_object_ids['object_id_matched_correctly'] = [(dataframe_with_matched_object_ids['pred_initial_obj_id'] == dataframe_with_matched_object_ids['object_identifier']) for x in dataframe_with_matched_object_ids['pred_initial_obj_id']]

        dataframe_with_matched_object_ids['object_id_matched_correctly'] = np.where(dataframe_with_matched_object_ids['object_id'] == dataframe_with_matched_object_ids['pred_initial_obj_id'], 'True', 'False')

        object_matching_accuarcy = (sum(dataframe_with_matched_object_ids['object_id_matched_correctly'] == 'True')) / dataframe_with_matched_object_ids.shape[0]

        dataframe_with_matched_object_ids.to_excel('evaluation/assets/generated_files/dataframe_with_matched_object_ids.xlsx')

        ## TODO: Check count of persons -> was this correctly identified?
        
        return object_matching_accuarcy

    # TODO: Write documentation
    # TODO: Test when sufficient change in API was provided
    def calculate_prediction_accuracy(self, accuracy_estimation_method = 'euclidean-distance', debugger_files = False, output_include_dataframe = False):

        prediction_dataframe = self.add_prediction_data_to_merged_identifier_dataframe(debugger_files=debugger_files)
 

        def calculate_euclidean_distance_between_two_points(true_x, true_y, true_z, pred_x, pred_y, pred_z):

            distance = ((true_x - pred_x)**2 + (true_y - pred_y)**2 + (true_z - pred_z)**2)**(0.5)

            return distance


        if accuracy_estimation_method == 'euclidean-distance':

            prediction_dataframe['prediction_error'] = prediction_dataframe.apply(lambda row : calculate_euclidean_distance_between_two_points(row['x_original'], row['y_original'], row['z_original'],
             row['predicted_x'], row['predicted_y'], row['predicted_z']), axis = 1)
                  
        else:
            raise ValueError('Please select valid accuracy estimation method')

        prediction_accuracy_sum = prediction_dataframe['prediction_error'].sum()
        prediction_accuracy_mean = prediction_dataframe['prediction_error'].mean()
        prediction_accuracy_min = prediction_dataframe['prediction_error'].min()
        prediction_accuracy_max = prediction_dataframe['prediction_error'].max()
        prediction_accuracy_median = prediction_dataframe['prediction_error'].median()

        prediction_accuracy = np.round([prediction_accuracy_sum, prediction_accuracy_mean, prediction_accuracy_min, prediction_accuracy_max, prediction_accuracy_median] ,2)

        if output_include_dataframe:
            return prediction_accuracy, prediction_dataframe

        else:
            return prediction_accuracy


        ## TODO: Implement average deviation
        ## TODO: Plot verteilung -> 8


# Test setup parameters 
endpoint_path = 'http://localhost:5000'

test_coord_sys = CoordinateSystem(6,-2,4, 0,0,0)
test_coord_sys2 = CoordinateSystem(0,-1,1, 2,3,4)
test_sensor = Sensor('RFID', test_coord_sys, 40, 20, 1000) # 20ms seems reasonable (https://electronics.stackexchange.com/questions/511278/low-latency-passive-rfid-solution)
test_sensor3 = Sensor('camera', test_coord_sys, 10, 17, 5000) # 16.666 ms is equal to 60fps
test_sensor2 = Sensor('WiFi 2.4GHz', test_coord_sys2, 30, 3, 4000) # 3ms is average wifi latency, source?

# Generate test_location
test_location = Location('test_name', 'test_id_ext', [test_sensor,test_sensor2, test_sensor3])


api_output, measurement_data = APIClient.end_to_end_API_test(test_location,[test_sensor, test_sensor2, test_sensor3], \
                                                             endpoint_path, measurement_points = 10000, print_progress = False, debugger_files = True,
                                                             identifier_randomization_method = 'sensor_and_object_based', identifier_type = 'mac-address')


prediction_outcome = PredictionEvaluator(api_output, measurement_data)

print('---- Object matching accuracy ----')
print(prediction_outcome.calculate_object_matching_accuracy())

print('---- Position prediction accuracy ----')
position_prediction_accuracy, prediction_outcome_dataframe = prediction_outcome.calculate_prediction_accuracy(debugger_files = True, output_include_dataframe = True)
print(position_prediction_accuracy)

def plot_position_accuracy_distribution(dataframe, analysis_dimension = 'total', bin_size = 5):
    
    if analysis_dimension == 'total':
        # Group data together
        hist_data = [dataframe['prediction_error']]

        group_labels = ['prediction_error - whole dataset']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size = bin_size)

        fig.write_html('evaluation/assets/generated_files/position_prediction_accuracy_plot_whole_dataset.html')
        return None

    elif analysis_dimension == 'sensor':
        hist_data = []
       
        group_labels = []



        for sensor in dataframe['sensor_id_from_api_output'].unique():
            temp_sensor_type_label = dataframe[dataframe['sensor_id_from_api_output'] == sensor]['sensor_type'].unique()[0]

            group_labels.append(temp_sensor_type_label + '___' + sensor)

            temp_sensor_df = dataframe[dataframe['sensor_id_from_api_output'] == sensor]

            hist_data.append(temp_sensor_df['prediction_error'])


        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size = bin_size)

        fig.write_html('evaluation/assets/generated_files/position_prediction_accuracy_plot_by_sensor.html')

        return None

    elif analysis_dimension == 'sensor_type':

        hist_data = []
       
        group_labels = []


        for sensor_type in dataframe['sensor_type'].unique():
           
            group_labels.append(sensor_type)

            temp_sensor_df = dataframe[dataframe['sensor_type'] == sensor_type]

            hist_data.append(temp_sensor_df['prediction_error'])


        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size = bin_size)

        fig.write_html('evaluation/assets/generated_files/position_prediction_accuracy_plot_by_sensor_type.html')

        return None

    else:
        raise ValueError('Please input an available analysis_dimension. Currently supported are total, sensor and sensor_type.')

plot_position_accuracy_distribution(prediction_outcome_dataframe, analysis_dimension = 'sensor_type')




## TODO: If API crash -> log data and share data with Julius
## TODO: visualize sensors in location -> Plot
## TODO: Evaluation plots -> see how performance changes with additional sensors, different sensors, more data points, etc
## TODO: Performance of API (CPU, RAM load) & Time based on measurement points
