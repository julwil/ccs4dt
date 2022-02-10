from os import close
import requests
import json
import pandas as pd
import openpyxl
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from collections import defaultdict

from sklearn.metrics.cluster import contingency_matrix

# Clustering evaluation with bcubed metrics
import bcubed

# Multi object tracking evaluation with motmetrics
import motmetrics as mm

# Project internal imports
from requests import api
from scripts.synthetic_data_generation.main.transform_test_data_set import CoordinateSystem, Sensor, Location, simulate_sensor_measurement_for_multiple_sensors
from scripts.data_visualization.main.visualize_outcomes import visualize_2D_movement_data, visualize_3D_movement_data

class APIClient(object):
    """This class represents the API client to call the CCS4DT API.

    """

    def __init__(self) -> None:
        super().__init__()

    def API_post_input_batch_call(API_endpoint_path, payload, location_id):
        """CCS4DT POST API call for input batch

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param payload: Sensor measurements json payload
        :type payload: dict(json)
        :param location_id: location id for which the measurements were taken
        :type location_id: integer

        :return: Returns API response and reponse parameters (id, status, location_id)
        :rtype: dict(json), string, string, string
        """
        
        response = requests.post(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs', json = json.loads(payload))

        # 202 Code with successful delivery
        response = response.json() if response and response.status_code == 202 else None

        return (response, response['id'], response['status'], response['location_id'])

    def API_get_input_batch_by_id_call(API_endpoint_path, location_id, input_batch_id):
        """CCS4DT GET API call for input batch by id

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param location_id: location id for which the input batch was posted
        :type location_id: integer
        :param input_batch_id: input batch id for which the results should be returned
        :type input_batch_id: integer

        :return: Returns API response 
        :rtype: dict(json)
        """
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs/' + str(input_batch_id))

        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    def API_get_all_input_batches_call(API_endpoint_path, location_id):
        """CCS4DT GET API call for all input batches of a specific location

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param location_id: location id for which the input batches should be retrieved
        :type location_id: integer


        :return: Returns API response 
        :rtype: dict(json)
        """
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs')

        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    def API_get_location_by_id_call(API_endpoint_path, location_id):
        """CCS4DT GET API call for location by id

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param location_id: location id for which the input batch was posted
        :type location_id: integer

        :return: Returns API response 
        :rtype: dict(json)
        """
        
        response = requests.get(API_endpoint_path + '/locations/'+ str(location_id))
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    def API_get_all_locations_call(API_endpoint_path):
        """CCS4DT GET API call for all locations

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string

        :return: Returns API response 
        :rtype: dict(json)
        """
        
        response = requests.get(API_endpoint_path + '/locations')
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    def API_post_new_location_call(API_endpoint_path, payload):
        """CCS4DT POST API call for new location

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param payload: Location json payload
        :type payload: dict(json)

        :return: Returns API response and reponse parameters (location id, name)
        :rtype: dict(json), string, string
        """


        response = requests.post(API_endpoint_path + '/locations', json = json.loads(payload))

        # 201 Code with successful delivery
        response = response.json() if response and response.status_code == 201 else None

        return (response, response['id'], response['name'])


    def API_get_output_batch_call(API_endpoint_path, location_id, batch_id):
        """CCS4DT GET API call for all output batch by id

        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param location_id: location id for which the input batch was posted
        :type location_id: integer
        :param batch_id: input batch id for which the results should be returned
        :type batch_id: integer


        :return: Returns API response 
        :rtype: dict(json)
        """
        
        response = requests.get(API_endpoint_path + '/locations/' + str(location_id) + '/inputs/' + str(batch_id) + '/outputs')
        json_data = response.json() if response and response.status_code == 200 else None

        return json_data

    def convert_sensor_measurements_to_api_conform_payload(dataframe, additional_file_generation = False): 
        """Take existing sensor measurement dataframe and convert it to API confirm input batch payload

        :param dataframe: Dataframe containing the sensor measurements
        :type dataframe: pandas dataframe
        :param additional_file_generation: Flag that states if the json payload should also be stored as a file in the generated_files directory
        :type additional_file_generation: boolean

        :return: Returns measurement JSON payload for CCS4DT API
        :rtype: dict(json)
        """

        dataframe = dataframe[['object_id','x_measured_rel_pos','y_measured_rel_pos','z_measured_rel_pos','sensor_type','sensor_id','date_time']]

        dataframe = dataframe.rename(columns = {'object_id':'object_identifier', 'x_measured_rel_pos':'x', 'y_measured_rel_pos':'y', 'z_measured_rel_pos':'z',
        'sensor_id':'sensor_identifier', 'sensor_type':'sensor_type', 'date_time':'timestamp'})

        if additional_file_generation == True:
            json_data = dataframe.to_json(path_or_buf= 'evaluation/assets/generated_files/synthetic_measurements.json', default_handler=str, orient='records')
            json_data = dataframe.to_json(default_handler=str, orient='records')
        elif additional_file_generation == False:
            json_data = dataframe.to_json(default_handler=str, orient='records')

        return(json_data)

    def end_to_end_API_test(location, sensors, api_endpoint_path, measurement_points = 250, return_simulation_output = True, print_progress = True, store_measurement_data_in_json_file = True, debugger_files = False, identifier_randomization_method = 'random', identifier_type = 'mac-address', transform_to_3D_data = False, synthetic_data_generation = True, measurement_file_path = '/'):
        """Wrapper function that combines all necessary API calls to run an end to end API test

        :param location: Location object for which API request should be run
        :type location: integer
        :param sensors: List of sensors that should be simulated / included in the API request
        :type sensors: List of Sensor
        :param API_endpoint_path: CCS4DT endpoint path (includes port if necessary)
        :type API_endpoint_path: string
        :param measurement_points: Amount of measurement points that should be taken into consideration for synthetic data generation
        :type measurement_points: integer
        :param return_simulation_output: Flag that indicates if measurement data (i.e. synthetic sensor measurements) should be included in function return, default = True
        :type return_simulation_output: boolean
        :param print_progress: Flag that indicates if progress should be printed to terminal, default = True
        :type print_progress: boolean
        :param store_measurement_data_in_json_file: Flag that indicates if input_batch should be stored as persistent .json in generated_files folder, default = True
        :type store_measurement_data_in_json_file: boolean
        :param debugger_files: Flag that indicates if intermediate results (, output_batch) should be stored as persistent .json in generated_files folder, default = False
        :type debugger_files: boolean
        :param identifier_randomization_type: For synthetic data generation: Specifies the randomization approach for object identifiers, default: 'none'
        :type identifier_randomization_type: string
        :param identifier_type: For synthetic data generation: Specifies type of identifier that should be used if randomization takes place, default: mac-address
        :type identifier_type: string
        :param transform_to_3D_data: For synthetic data generation: Specifies if data should either consider the true position coordinate as 0 (if == False) or set the z-coordinate of the true positions as 'height' (if == True), default is True
        :type transform_to_3D_data: boolean
        :param synthetic_data_generation: Flag that identifies if synthetic data needs to be generated (True) or if real-world data file can be taken (False), default = True
        :type synthetic_data_generation: boolean
        :param measurement_file_path: Filepath to real-world sensor measurement file (only relevant if synthetic_data_generation == False)
        :type measurement_file_path: string


        :return: Returns output batch (and optionally measurement)
        :rtype: dict(json), dict(json)
        """

        
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

        if synthetic_data_generation:
            # Simulate sensor data measurement
            measurement_data = simulate_sensor_measurement_for_multiple_sensors(sensors, measurement_points, identifier_randomization_method = identifier_randomization_method, identifier_type = identifier_type, transform_to_3D_data = transform_to_3D_data)

            # Generate synthetic measurement data payload
            API_payload = APIClient.convert_sensor_measurements_to_api_conform_payload(measurement_data, additional_file_generation = store_measurement_data_in_json_file)

            if(debugger_files):
                with open('evaluation/assets/generated_files/synthetic_measurement_payload.json', 'w') as text_file:
                    text_file.write(API_payload)

        elif not(synthetic_data_generation):
            API_payload = measurement_file_path

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

        # If debugger attribute is set, store API data in file
        if(debugger_files):
            with open('evaluation/assets/generated_files/output_batch_response.json', 'w') as text_file:
                text_file.write(str(output_batch_response))

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

class PredictionEvaluator(object):
    """This class represents the evaluator for a API request / synthetic measurement combination
 

    :param api_output: Output batch of the CCS4DT API
    :type api_output: dict
    :param measurement_data: Synthetic sensor measurement data
    :type measurement_data: dict
    """

    def __init__(self, api_output, measurement_data):
        
        self.api_output = api_output
        self.measurement_data =  measurement_data
        self.location_id = self.api_output.pop('location_id')
        self.input_batch_id = self.api_output.pop('input_batch_id')
        self.predicted_positions = self.api_output

        # Remove object identifier mappings from prediction results and store in seperate attribute
        self.object_identifier_mappings = self.api_output.pop('object_identifier_mappings')

    def get_prediction_results(self):
        """Getter function for (position) prediction results

        :return: Returns predicted positions
        :rtype: dict
        """
        return(self.predicted_positions)

    def get_object_identifier_mappings(self):
        """Getter function for object identifer mappings (i.e. mappings between predicted individuals and object ids from sensor measurements) results

        :return: Returns object identifier mapping
        :rtype: dict
        """
        return(self.object_identifier_mappings)

    def add_object_identifier_mapping_to_measurement_dataframe(self):
        """Function that adds object identifier mapping to measurement data frame

        :return: Returns dataframe that includes object identifier mapping
        :rtype: pandas dataframe
        """

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

    def generate_prediction_dataframe(self, add_matching_initial_identifiers = False):
        """Function that generates a prediction dataframe

        :param add_matching_initial_identifiers: Flag that indicates if initial (true) object mappings should be added too, default = False
        :type add_matching_initial_identifiers: boolean

        :return: Returns dataframe of predictions
        :rtype: pandas dataframe
        """

        pred_obj_id_list = []
        pred_timestamp_list = []
        pred_confidence_list = []
        pred_x_list = []
        pred_y_list = []
        pred_z_list = []


        for i in self.predicted_positions['positions']:
            pred_obj_id_list.append(i['object_identifier'])
            pred_timestamp_list.append(i['timestamp'])
            pred_confidence_list.append(i['confidence'])
            pred_x_list.append(i['x'])
            pred_y_list.append(i['y'])
            pred_z_list.append(i['z'])

    
        prediction_df = pd.DataFrame({'prediction_obj_identifier': pred_obj_id_list,
                   'prediction_timestamp': pred_timestamp_list,
                   'prediction_confidence': pred_confidence_list,
                   'predicted_x': pred_x_list,
                   'predicted_y': pred_y_list,
                   'predicted_z': pred_z_list,
                   })

        # Necessary transformation because prediction timestamps have lenght of 13 digits and true data timestamps are 10 digits only
        def reduce_to_10_digits(input):
            return (int(input/1000))
        prediction_df['prediction_timestamp'] = prediction_df['prediction_timestamp'].apply(reduce_to_10_digits)

        prediction_df = prediction_df.sort_values(by='prediction_timestamp')

        # Adds original object identifier as column in prediction dataframe
        if add_matching_initial_identifiers:
            prediction_df['mapping_keys'] =  prediction_df['prediction_obj_identifier'].map(self.get_object_identifier_mappings())
            
        return prediction_df

    def add_prediction_data_to_merged_identifier_dataframe(self, debugger_files = False):
        """Function that adds prediction data (object ids, coordinates) to the original measurement dataframe which was enriched with identifiers of objects, true individuals and predictions

        :param debugger_files: Flag that indicates if dataframes should be persisted in separate files for debugging purposes, default = False
        :type debugger_files: boolean

        :return: Returns dataframe of true data, sensor measurements and predictions
        :rtype: pandas dataframe
        """

        merged_identifier_dataframe = self.add_object_identifier_mapping_to_measurement_dataframe()

        prediction_df = self.generate_prediction_dataframe()

        # Type conversions to ensure merge is working
        prediction_df['prediction_timestamp'] = prediction_df['prediction_timestamp'].astype(float)
        prediction_df['prediction_obj_identifier'] = prediction_df['prediction_obj_identifier'].astype(str)
        merged_identifier_dataframe['pred_obj_id'] = merged_identifier_dataframe['pred_obj_id'].astype(str)

        # Sort dataframes (prediction dataframe already sorted beforehand with self.generate_prediction_dataframe())
        merged_identifier_dataframe = merged_identifier_dataframe.sort_values(by='timestamp')
          
        merged_prediction_df = pd.merge_asof(merged_identifier_dataframe, prediction_df, left_on=['timestamp'], right_on = ['prediction_timestamp'], left_by=['pred_obj_id'], right_by=['prediction_obj_identifier'], direction='nearest')

        if(debugger_files):
            merged_identifier_dataframe.to_excel('evaluation/assets/generated_files/merged_identifier_dataframe.xlsx')
            merged_prediction_df.to_excel('evaluation/assets/generated_files/merged_prediction_dataframe.xlsx')
            prediction_df.to_excel('evaluation/assets/generated_files/prediction_df.xlsx')

        return merged_prediction_df


    def calculate_object_matching_accuracy(self, clustering_evaluation_method = 'bcubed_fscore', debugger_files = True):
        """Function that calculates object matching accuracy score for the requested evaluation method

        :param clustering_evaluation_method: Clustering evaluation method, default = bcubed_fscore
        :type clustering_evaluation_method: string
        :param debugger_files: Flag that indicates if results should be persisted in separate files for debugging purposes, default = True
        :type debugger_files: boolean

        :return: Returns object matching accuracy score for the evaluation method
        :rtype: float
        """


        dataframe_with_matched_object_ids = self.add_object_identifier_mapping_to_measurement_dataframe()

        true_occupant_id_list = dataframe_with_matched_object_ids['occupant_id'].values
        predicted_object_id_list = dataframe_with_matched_object_ids['pred_obj_id'].values

        unique_true_occupant_id_list =  np.unique(true_occupant_id_list)
        unique_predicted_object_id_list = np.unique(predicted_object_id_list)

        # Generate contingency matrix for true occupant id and predicted object id
        contingency_matrix_true_occupant_vs_predicted_object = contingency_matrix(true_occupant_id_list, predicted_object_id_list)

        identified_mapping_dictionary = {}


        # Iterate over contingency matrix
        for i, row in enumerate(contingency_matrix_true_occupant_vs_predicted_object):

            # Find best mapping result for true occupant id <-> predicted object id, necessary to be able to define mapping found by matching algorithm
            best_mapping_result = np.where(row == np.amax(row))

            # Update mapping dictionary with best match key is true id, value is predicted id [0][0] here takes the highest value
            identified_mapping_dictionary.update( { unique_true_occupant_id_list[i] : unique_predicted_object_id_list[best_mapping_result[0][0]] } )

            # Inverse dictionary
            inversed_identified_mapping_dictionary = {v: k for k, v in identified_mapping_dictionary.items()}
            # Add mapping logic based on contingency matrix identifed mapping dictionary
            dataframe_with_matched_object_ids['mapped_true_id_based_on_contingency_matrix'] = dataframe_with_matched_object_ids['pred_obj_id'].map(inversed_identified_mapping_dictionary)

        if clustering_evaluation_method == 'naive':
            # Add boolean variable if object was correctly matched or not
            dataframe_with_matched_object_ids['object_id_matched_correctly_naive'] = np.where(dataframe_with_matched_object_ids['occupant_id'] == dataframe_with_matched_object_ids['mapped_true_id_based_on_contingency_matrix'], 'True', 'False')
           
            object_matching_accuracy = (sum(dataframe_with_matched_object_ids['object_id_matched_correctly_naive'] == 'True')) / dataframe_with_matched_object_ids.shape[0]
        
       # Check if evaluation method is based on bcubed metrics
        elif 'bcubed' in clustering_evaluation_method:

            # If yes, set up necessary ground_truth and prediction dictionaries
            dataframe_with_matched_object_ids['ground_truth_combined_sensor_object_id'] = dataframe_with_matched_object_ids['sensor_id'] + '___' +  dataframe_with_matched_object_ids['object_id']

            dataframe_with_matched_object_ids['predicition_combined_sensor_object_id'] = dataframe_with_matched_object_ids['sensor_id_from_api_output'] + '___' +  dataframe_with_matched_object_ids['pred_initial_obj_id']
            
            # TODO: Write documentation
            def generate_ground_truth_mapping_dict(dataframe_with_matched_object_ids):

                # Dropping duplicates here ensures that no duplicates are later added to the sets
                dataframe_for_analysis = dataframe_with_matched_object_ids[['occupant_id', 'ground_truth_combined_sensor_object_id']].drop_duplicates()

                gt_dict = defaultdict(set)
               
               # Generates ground truth dataset dictonary
                for idx,row in dataframe_for_analysis.iterrows():
                    gt_dict[row['occupant_id']].add(row['ground_truth_combined_sensor_object_id'])

                return gt_dict

            # TODO: Write documentation
            def generate_prediction_mapping_dict(dataframe_with_matched_object_ids):

                # Dropping duplicates here ensures that no duplicates are later added to the sets
                dataframe_for_analysis = dataframe_with_matched_object_ids[['mapped_true_id_based_on_contingency_matrix', 'predicition_combined_sensor_object_id']].drop_duplicates()

                gt_dict = defaultdict(set)
               
               # Generates ground truth dataset dictonary
                for idx,row in dataframe_for_analysis.iterrows():
                    gt_dict[row['mapped_true_id_based_on_contingency_matrix']].add(row['predicition_combined_sensor_object_id'])

                return gt_dict

            ground_truth_mapping_dict = generate_ground_truth_mapping_dict(dataframe_with_matched_object_ids)

            prediction_mapping_dict = generate_prediction_mapping_dict(dataframe_with_matched_object_ids)

            # Check now which exact bcubed metric should be used 
            # bcubed precision
            if clustering_evaluation_method == 'bcubed_precision':
                
                object_matching_accuracy = bcubed.precision(ground_truth_mapping_dict, prediction_mapping_dict)
            
            # bcubed recall
            elif clustering_evaluation_method == 'bcubed_recall':
                
                object_matching_accuracy = bcubed.recall(ground_truth_mapping_dict, prediction_mapping_dict)

            # bcubed fscore
            elif clustering_evaluation_method == 'bcubed_fscore':
                
                bcubed_precision = bcubed.precision(ground_truth_mapping_dict, prediction_mapping_dict)

                bcubed_recall = bcubed.recall(ground_truth_mapping_dict, prediction_mapping_dict)

                object_matching_accuracy = bcubed.fscore(bcubed_precision, bcubed_recall)

            # Catch errors
            else: 
                raise ValueError('Please select one of the available clustering evaluation methods')
            

        else:
            raise ValueError('Please select one of the available clustering evaluation methods')

        if(debugger_files):
            dataframe_with_matched_object_ids.to_excel('evaluation/assets/generated_files/dataframe_with_matched_object_ids.xlsx')

        return object_matching_accuracy


    def calculate_prediction_accuracy(self, accuracy_estimation_method = 'euclidean-distance', debugger_files = False, output_include_dataframe = False, clear_mot_threshold = 1000000, output_precision_euclidean_distance = 2, remove_upsamples = False):
        """Function that calculates position prediction score for the requested evaluation method

        :param accuracy_estimation_method: Position rpediction evaluation method, default = euclidean-distance
        :type accuracy_estimation_method: string
        :param debugger_files: Flag that indicates if results should be persisted in separate files for debugging purposes, default = False
        :type debugger_files: boolean
        :param output_include_dataframe: Flag that indicates if function should return prediction dataframe too, default = False
        :type output_include_dataframe: boolean
        :param clear_mot_threshold: For CLEAR-MOT metrics: Squared cut-off value until which a point should be considered "correctly identifier", default = 1000000
        :type clear_mot_threshold: integer
        :param output_precision_euclidean_distance: For naive metric: Precision of metrics (i.e. decimal digits), default = 2
        :type output_precision_euclidean_distance: integer
        :param remove_upsamples: Relevant for CLEAR-MOT metrics: Flag that indicates if upsampled datapoints from API response should be removed, default = False
        :type remove_upsamples: boolean

        :return: Returns position prediction accuracy (and optionally the dataframe containing the predictions)
        :rtype: float or list of floats, pandas dataframe
        """

        def calculate_euclidean_distance_between_two_points(true_x, true_y, true_z, pred_x, pred_y, pred_z):
            """Function that calculates euclidean distance between two points, based on elementary coordinates

            :param true_x: x coordinate of true position
            :type true_x: numerical
            :param true_y: y coordinate of true position
            :type true_y: numerical
            :param true_z: z coordinate of true position
            :type true_z: numerical
            :param pred_x: x coordinate of predicted position
            :type pred_x: numerical
            :param pred_y: y coordinate of predicted position
            :type pred_y: numerical
            :param pred_z: z coordinate of predicted position
            :type pred_z: numerical

            :return: Returns euclidean distance between two points
            :rtype: numerical
            """

            distance = ((true_x - pred_x)**2 + (true_y - pred_y)**2 + (true_z - pred_z)**2)**(0.5)

            return distance

        if accuracy_estimation_method == 'euclidean-distance':

            # Add prediction data to existing ground truth dataframe
            prediction_dataframe = self.add_prediction_data_to_merged_identifier_dataframe(debugger_files=debugger_files)

            prediction_dataframe['prediction_error'] = prediction_dataframe.apply(lambda row : calculate_euclidean_distance_between_two_points(row['x_original'], row['y_original'], row['z_original'],
             row['predicted_x'], row['predicted_y'], row['predicted_z']), axis = 1)

            prediction_accuracy_sum = prediction_dataframe['prediction_error'].sum()
            prediction_accuracy_mean = prediction_dataframe['prediction_error'].mean()
            prediction_accuracy_min = prediction_dataframe['prediction_error'].min()
            prediction_accuracy_max = prediction_dataframe['prediction_error'].max()
            prediction_accuracy_median = prediction_dataframe['prediction_error'].median()

            prediction_accuracy = [round(i,output_precision_euclidean_distance) for i in [prediction_accuracy_sum, prediction_accuracy_mean, prediction_accuracy_min, prediction_accuracy_max, prediction_accuracy_median]]

        elif accuracy_estimation_method == 'CLEAR-MOT-metrics':

            if output_include_dataframe == True:
                raise ValueError('Not possible to output dataframe with CLEAR MOT metrics')

            

            if remove_upsamples:
                prediction_dataframe = self.filter_false_positives_introduced_due_to_upsampling()
            else:
                prediction_dataframe = self.generate_prediction_dataframe()


            ground_truth_dataframe = self.add_object_identifier_mapping_to_measurement_dataframe()

            

            # TODO: Write documentation
            def convert_list_to_integer_mapping_dictionary(list_with_ids_to_convert):
                
                unique_id_list = list(set(list_with_ids_to_convert))

                mapping_dict_int_to_id = {}

                for i in range(0,len(unique_id_list)):

                    mapping_dict_int_to_id[i] = unique_id_list[i]

                # Inverse mapping
                mapping_dict_int_to_id_inverse = {v: k for k, v in mapping_dict_int_to_id.items()}

                return mapping_dict_int_to_id_inverse

            # Conversion of class ids to integers (necessary for CLEAR MOT metrics)
            ## Convert true object class ids to integer
            mapping_occupant_id_integer = convert_list_to_integer_mapping_dictionary(ground_truth_dataframe['occupant_id'])
            ground_truth_dataframe['integer_occupant_id'] = ground_truth_dataframe['occupant_id'].map(mapping_occupant_id_integer)

            ## Convert predicted object class ids to integer
            mapping_predicted_object_id_integer = convert_list_to_integer_mapping_dictionary(prediction_dataframe['prediction_obj_identifier'])
            prediction_dataframe['integer_pred_obj_id'] = prediction_dataframe['prediction_obj_identifier'].map(mapping_predicted_object_id_integer)

            # Create an accumulator that will be updated during each frame
            acc = mm.MOTAccumulator(auto_id=True)

            all_timeframes = list(set(ground_truth_dataframe['timestamp']))

            for each_timeframe in all_timeframes:
    
                # TODO: Write documentation
                def generate_acc_update(specific_timeframe, true_objects_dataframe, predicted_objects_dataframe, clear_mot_threshold):

                    positions_of_true_objects_for_specific_timeframe, list_true_obj_ids_for_specific_timeframe = extract_true_objects_in_this_timeframe(specific_timeframe, true_objects_dataframe) ## Returns matrix of position coordinats and list of true object ids

                    positions_of_predicted_objects_for_specific_timeframe, list_predicted_obj_ids_for_specific_timeframe = extract_predicted_objects_in_this_timeframe(specific_timeframe, predicted_objects_dataframe) ## Returns matrix of position coordinats and list of true object ids

                    o = np.array(positions_of_true_objects_for_specific_timeframe)


                    h = np.array(positions_of_predicted_objects_for_specific_timeframe)


                    C = mm.distances.norm2squared_matrix(o, h, max_d2 = clear_mot_threshold*clear_mot_threshold)

                    return list_true_obj_ids_for_specific_timeframe, list_predicted_obj_ids_for_specific_timeframe, C

                # TODO: Write documentation
                def extract_true_objects_in_this_timeframe(specific_timeframe, true_objects_dataframe):
                    
                    ## Keep only values for that specific timeframe and that only one for each integer occupant id
                    unique_and_timefiltered_true_obj_dataframe = true_objects_dataframe[true_objects_dataframe['timestamp'] == specific_timeframe].drop_duplicates(subset=['integer_occupant_id'])
                   
                    # Extract list of true object ids
                    list_true_object_ids_for_specific_timeframe = unique_and_timefiltered_true_obj_dataframe['integer_occupant_id'].tolist()

                    # Extract individual cordinates
                    x_pos_list_true_objects_for_specific_timeframe = unique_and_timefiltered_true_obj_dataframe['x_original'].tolist()
                    y_pos_list_true_objects_for_specific_timeframe = unique_and_timefiltered_true_obj_dataframe['y_original'].tolist()
                    z_pos_list_true_objects_for_specific_timeframe = unique_and_timefiltered_true_obj_dataframe['z_original'].tolist()

                    # Generate matrix of positions of individual ground truth objects
                    positions_of_true_objects_for_specific_timeframe = [list(a) for a in zip(x_pos_list_true_objects_for_specific_timeframe, y_pos_list_true_objects_for_specific_timeframe, z_pos_list_true_objects_for_specific_timeframe)]

                    return positions_of_true_objects_for_specific_timeframe, list_true_object_ids_for_specific_timeframe

              
                # TODO: Write documentation
                def extract_predicted_objects_in_this_timeframe(specific_timeframe, predictions_dataframe):

                    
                    ## Keep only values for that specific timeframe and that only one for each integer predicted object id
                    unique_and_timefiltered_pred_obj_dataframe = predictions_dataframe[predictions_dataframe['prediction_timestamp'] == specific_timeframe].drop_duplicates(subset=['integer_pred_obj_id'])

                    # Extract list of true object ids
                    list_predicted_object_ids_for_specific_timeframe = unique_and_timefiltered_pred_obj_dataframe['integer_pred_obj_id'].tolist()

                    # Extract individual cordinates
                    x_pos_list_predicted_objects_for_specific_timeframe = unique_and_timefiltered_pred_obj_dataframe['predicted_x'].tolist()
                    y_pos_list_predicted_objects_for_specific_timeframe = unique_and_timefiltered_pred_obj_dataframe['predicted_y'].tolist()
                    z_pos_list_predicted_objects_for_specific_timeframe = unique_and_timefiltered_pred_obj_dataframe['predicted_z'].tolist()

                    # Generate matrix of positions of individual ground truth objects
                    positions_of_predicted_objects_for_specific_timeframe = [list(a) for a in zip(x_pos_list_predicted_objects_for_specific_timeframe, y_pos_list_predicted_objects_for_specific_timeframe, z_pos_list_predicted_objects_for_specific_timeframe)]


                    return positions_of_predicted_objects_for_specific_timeframe, list_predicted_object_ids_for_specific_timeframe

                list_true_ids, list_predicted_ids, distance_matrix = generate_acc_update(each_timeframe, ground_truth_dataframe, prediction_dataframe, clear_mot_threshold)

                frameid = acc.update(list_true_ids, list_predicted_ids, distance_matrix)



            # Prints all mot events
            # print(acc.mot_events)

            mh = mm.metrics.create()
            summary_reduced = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
            print(summary_reduced)

            summary = mh.compute_many(
            [acc, acc.events.loc[0:19]],
            metrics=mm.metrics.motchallenge_metrics,
            names=['full', 'first 20 frames'],
            generate_overall=True
            )

            strsummary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names
            )
                       

            prediction_accuracy = strsummary

                  
        else:
            raise ValueError('Please select valid accuracy estimation method')

        

        if(debugger_files):
            prediction_dataframe.to_excel('evaluation/assets/generated_files/full_prediction_dataframe.xlsx')

        if output_include_dataframe:
            return prediction_accuracy, prediction_dataframe

        else:
            return prediction_accuracy

    def filter_false_positives_introduced_due_to_upsampling(self):
        """Function that removes upsampled results from API response (i.e. predictions for each timestamp where there were no reported sensor measurements)

        :return: Returns dataframe of predictions excluding umsampled values
        :rtype: pandas dataframe
        """
    
        prediction_dataframe = self.generate_prediction_dataframe(add_matching_initial_identifiers = True)

        input_dataframe = self.measurement_data

        unique_timestamp_list = list(set(input_dataframe['timestamp']))
        
        mapping_dictionary_timestamp_on_sensor_plus_object_identifier = {}

        
        # Generate filter dictionary for measurement data based on individual timestamps as keys
        for i in unique_timestamp_list:

            filtered_input_dataframe = input_dataframe[input_dataframe['timestamp'] == i]

            # Type conversion to ensure both are strings for concatenation
            filtered_input_dataframe['sensor_id'] =  filtered_input_dataframe['sensor_id'].astype(str)
            filtered_input_dataframe['object_id'] =  filtered_input_dataframe['object_id'].astype(str)

            columns_to_concat = ['object_id', 'sensor_id']
            filtered_input_dataframe['sensor_plus_object_identifier'] = filtered_input_dataframe[columns_to_concat].apply(lambda row: '___'.join(row.values.astype(str)), axis=1)

          
            list_of_relevant_objects = list(filtered_input_dataframe['sensor_plus_object_identifier'].unique())

            mapping_dictionary_timestamp_on_sensor_plus_object_identifier[int(i)] = list_of_relevant_objects

        def list_compare(list1, list2):
            """Function that checks if an element in list 2 exists in list 1

            :param list1: List to compare to
            :type list1: list
            :param list2: List to compare from
            :type list2: list


            :return: Returns True if element is in other list, and False otherwise
            :rtype: boolean
            """

            # Check if both lists exist
            if list2 == None:
                    return False
            else:
                for val in list1:
                    if val in list2:
                        return True
                return False

        prediction_dataframe['is upsampled value'] = [list_compare(x, mapping_dictionary_timestamp_on_sensor_plus_object_identifier.get(y)) for x,y in zip(prediction_dataframe['mapping_keys'],prediction_dataframe['prediction_timestamp'])]

        prediction_dataframe_without_upsamples = prediction_dataframe[prediction_dataframe['is upsampled value'] == True]

        return(prediction_dataframe_without_upsamples)

## YOUR TEST PARAMETERS FOR AN END TO END API TEST GO HERE
endpoint_path = 'http://185.147.10.249:5000'
test_coord_sys = CoordinateSystem(0,0,0, 0,0,0)
test_sensor = Sensor('WiFi 2.4GHz', test_coord_sys, 0, 0, 0) 
test_location = Location('test_name', 'test_id_ext', [test_sensor])

api_output, measurement_data = APIClient.end_to_end_API_test(test_location, [test_sensor], 
                                                             endpoint_path, measurement_points = 20000, print_progress = True, debugger_files = True,
                                                             identifier_randomization_method = 'sensor_and_object_based', identifier_type = 'mac-address',
                                                             transform_to_3D_data = False, synthetic_data_generation = True, measurement_file_path = None)


prediction_outcome = PredictionEvaluator(api_output, measurement_data)   

# END OF TEST PARAMETER DEFINITION

selected_clustering_evaluation_method = ['naive','bcubed_recall','bcubed_precision','bcubed_fscore']

for i in selected_clustering_evaluation_method:
    print('\n ---- Object matching accuracy ----')
    print('Chosen clustering evaluation method: %s'%(i))
    print(prediction_outcome.calculate_object_matching_accuracy(clustering_evaluation_method = i))

output_dataframe_included = True
metrics = 'euclidean-distance'
metrics2 = 'CLEAR-MOT-metrics'
clear_mot_threshold = 100


print('\n ---- Position prediction accuracy according to naive euclidean distance measures in cm [sum, mean, min, max, median] ----')
if output_dataframe_included:
    position_prediction_accuracy, prediction_outcome_dataframe = prediction_outcome.calculate_prediction_accuracy(debugger_files = False, output_include_dataframe = output_dataframe_included, accuracy_estimation_method= metrics, clear_mot_threshold = clear_mot_threshold)
else:
    position_prediction_accuracy = prediction_outcome.calculate_prediction_accuracy(debugger_files = False, output_include_dataframe = output_dataframe_included, accuracy_estimation_method= metrics, clear_mot_threshold = clear_mot_threshold)
print(position_prediction_accuracy)

print('\n ---- Position prediction accuracy according to CLEAR MOT metrics [in cm² if metric with unit] WITH UPSAMPLED DATA ----')
position_prediction_accuracy = prediction_outcome.calculate_prediction_accuracy(debugger_files = False, output_include_dataframe = False, accuracy_estimation_method= metrics2, clear_mot_threshold = clear_mot_threshold)
print(position_prediction_accuracy)


print('\n ---- Position prediction accuracy according to CLEAR MOT metrics [in cm² if metric with unit] WITHOUT UPSAMPLED DATA----')
position_prediction_accuracy = prediction_outcome.calculate_prediction_accuracy(debugger_files = False, output_include_dataframe = False, accuracy_estimation_method= metrics2, clear_mot_threshold = clear_mot_threshold, remove_upsamples = True)
print(position_prediction_accuracy)


def print_CLEAR_MOT_explanation():
    print("Explanation: \n \
        'idf1': identification measure - global min-cost F1 score, \n \
        'idp' identification measure - global min-cost precision, \n \
        'idr' identification measure - global min-cost recall,\n \
        'recall' Number of detections over number of objects,\n\
        'precision' Number of detected objects over sum of detected and false positives.,\n\
        GT 'num_unique_objects' Total number of unique object ids encountered.,\n\
        MT 'mostly_tracked' Number of objects tracked for at least 80 percent of lifespan.,\n\
        PT 'partially_tracked' Number of objects tracked between 20 and 80 percent of lifespan.,\n\
        ML 'mostly_lost' Number of objects tracked less than 20 percent of lifespan.,\n\
        FP 'num_false_positives',\n\
        FN 'num_misses' Total number of misses.,\n\
        IDs 'num_switches' Total number of detected objects including matches and switches.,\n\
        FM 'num_fragmentations' Total number of switches from tracked to not tracked.,\n\
        'mota' Multiple object tracker accuracy.,\n\
        'motp' Multiple object tracker precision.,\n\
        IDt 'num_transfer' Total number of track transfer.,\n\
        IDa 'num_ascend' Total number of track ascend.,\n\
        IDm 'num_migrate' Total number of track migrate.")

#print_CLEAR_MOT_explanation()

def plot_position_accuracy_distribution(dataframe, analysis_dimension = 'total', bin_size = 5):
    """Function that generates plots of error distribution for naive prediction accuracy
    
    :param dataframe: Dataframe that contains position prediction errors and sensor data
    :type dataframe: pandas dataframe
    :param analysis_dimension: dimension in which data should be analyzed (e.g. total, per sensor, per sensor type), default = total
    :type analysis_dimension: string
    :param bin_size: bin size of distribution in cm, default = 5
    :type bin_size: integer

    :return: None return (but generation of .html plots in generated_files folder)
    :rtype: None
    """


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

if output_dataframe_included:
    plot_position_accuracy_distribution(prediction_outcome_dataframe, analysis_dimension = 'sensor_type')
    plot_position_accuracy_distribution(prediction_outcome_dataframe, analysis_dimension = 'sensor')
    plot_position_accuracy_distribution(prediction_outcome_dataframe, analysis_dimension = 'total')

visualize_2D_movement_data(prediction_outcome_dataframe.drop_duplicates(['occupant_id','timestamp']))