from os import close
import requests
import json
import pandas as pd
import openpyxl
import numpy as np

# Project internal imports
from requests import api
from scripts.synthetic_data_generation.main.transform_test_data_set import CoordinateSystem, Sensor, Location, simulate_sensor_measurement_for_multiple_sensors

# TODO: Write documentation
def generate_fake_data():

    fake_api_output = {
        "input_batch_id": 131,
        "location_id": 93,
        "object_identifier_mappings": {
            "61204d02-e2b2-4959-9ae9-e4453120bcf4":  
            {
                "ffe7048b-9cae-4845-b561-baed2913eff8": ["39202"],
                "b1c162be-22e4-419a-ad54-020f165ac7c2": ["39202"],
                "20affda5-dce1-4d46-ab8b-ba2de6cbdecd": ["39202"]
            }
            ,
            "a76b4b03-5c46-4382-bfd6-b9a8baa13295": 
            {
                "ffe7048b-9cae-4845-b561-baed2913eff8": ["39222"],
                "20affda5-dce1-4d46-ab8b-ba2de6cbdecd": ["37957"]
            }
            ,

            "84a10ccd-9c15-4c5a-96b1-3ddf91be9f32": 
            {
                "20affda5-dce1-4d46-ab8b-ba2de6cbdecd": ["39222"]
            }
            ,

            "2f83cdf2-c5d9-4986-a7d5-d0ad50b96b65": 
            {
                "b1c162be-22e4-419a-ad54-020f165ac7c2": ["37957"]
            }
            ,
        },
        "positions": [
            {
                "object_identifier": "2f83cdf2-c5d9-4986-a7d5-d0ad50b96b65",
                "confidence": 1.0,
                "timestamp": 1559376006000,
                "x": 242.92016573379493,
                "y": 237.7153713751764,
                "z": 39.07671300353569
            },
            {
                "object_identifier": "61204d02-e2b2-4959-9ae9-e4453120bcf4",
                "confidence": 1.0,
                "timestamp": 1559376006000,
                "x": 384.0217205510169,
                "y": 99.89764945583775,
                "z": 24.664297962724383
            },
            {
                "object_identifier": "84a10ccd-9c15-4c5a-96b1-3ddf91be9f32",
                "confidence": 1.0,
                "timestamp": 1559376002000,
                "x": 173.25836702570462,
                "y": 282.41745763347814,
                "z": 56.82374895066931
            },
            {
                "object_identifier": "a76b4b03-5c46-4382-bfd6-b9a8baa13295",
                "confidence": 1.0,
                "timestamp": 1559376007000,
                "x": 225.67293764364638,
                "y": 227.40511153799474,
                "z": 22.836052331421442
            }
        ]
    }

    fake_input_data = [
        {
            "object_identifier": "37957",
            "x": 243.0,
            "y": 275.0,
            "z": 11.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376000000
        },
        {
            "object_identifier": "37957",
            "x": 271.0,
            "y": 232.0,
            "z": -9.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376000495
        },
        {
            "object_identifier": "39202",
            "x": 348.0,
            "y": 117.0,
            "z": 10.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376000691
        },
        {
            "object_identifier": "39202",
            "x": 336.0,
            "y": 120.0,
            "z": -9.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376000992
        },
        {
            "object_identifier": "37957",
            "x": 218.0,
            "y": 254.0,
            "z": -4.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376001002
        },
        {
            "object_identifier": "39222",
            "x": 169.0,
            "y": 287.0,
            "z": 27.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376001892
        },
        {
            "object_identifier": "37957",
            "x": 243.0,
            "y": 243.0,
            "z": -18.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376003496
        },
        {
            "object_identifier": "37957",
            "x": 234.0,
            "y": 255.0,
            "z": 0.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376004000
        },
        {
            "object_identifier": "39202",
            "x": 353.0,
            "y": 102.0,
            "z": -5.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376004992
        },
        {
            "object_identifier": "39202",
            "x": 339.0,
            "y": 147.0,
            "z": 18.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376005197
        },
        {
            "object_identifier": "39202",
            "x": 387.0,
            "y": 104.0,
            "z": 4.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376006490
        },
        {
            "object_identifier": "37957",
            "x": 242.0,
            "y": 212.0,
            "z": -17.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376006499
        },
        {
            "object_identifier": "37957",
            "x": 210.0,
            "y": 243.0,
            "z": 3.0,
            "sensor_type": "RFID",
            "sensor_identifier": "20affda5-dce1-4d46-ab8b-ba2de6cbdecd",
            "timestamp": 1559376007004
        },
        {
            "object_identifier": "39222",
            "x": 122.3869497573,
            "y": 269.1892330254,
            "z": 6.0212434949,
            "sensor_type": "WiFi 2.4GHz",
            "sensor_identifier": "ffe7048b-9cae-4845-b561-baed2913eff8",
            "timestamp": 1559376000992
        },
        {
            "object_identifier": "39202",
            "x": 350.8086521583,
            "y": 128.0728320727,
            "z": -4.8731245164,
            "sensor_type": "WiFi 2.4GHz",
            "sensor_identifier": "ffe7048b-9cae-4845-b561-baed2913eff8",
            "timestamp": 1559376000992
        },
        {
            "object_identifier": "39202",
            "x": 363.0,
            "y": 129.0,
            "z": 3.0,
            "sensor_type": "camera",
            "sensor_identifier": "b1c162be-22e4-419a-ad54-020f165ac7c2",
            "timestamp": 1559376001186
        },
        {
            "object_identifier": "37957",
            "x": 238.0,
            "y": 241.0,
            "z": 10.0,
            "sensor_type": "camera",
            "sensor_identifier": "b1c162be-22e4-419a-ad54-020f165ac7c2",
            "timestamp": 1559376006499
        }
    ]

    return fake_input_data, fake_api_output

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

    def end_to_end_API_test(location, sensors, api_endpoint_path, measurement_points = 250, return_simulation_output = True, print_progress = True, store_measurement_data_in_json_file = True):

        # API request: GET all locations
        APIClient.API_get_all_locations_call(api_endpoint_path)

        # Construct test location payload
        location_payload = location.construct_json_payload()

        # API request: POST new location
        post_location_response, location_id, test_location_name = (APIClient.API_post_new_location_call(api_endpoint_path, location_payload))

        # Simulate sensor data measurement
        measurement_data = simulate_sensor_measurement_for_multiple_sensors(sensors, measurement_points)

        # Generate synthetic measurement data payload
        API_payload = APIClient.convert_sensor_measurements_to_api_conform_payload(measurement_data, additional_file_generation = store_measurement_data_in_json_file)

        # API request: POST new input batch
        post_input_batch_response, input_batch_id, input_batch_status, location_id_for_input_batch = APIClient.API_post_input_batch_call(api_endpoint_path, API_payload, location_id)

        if print_progress == True:
            # API request: GET input batch status
            print('Get input batch by id')
            print(APIClient.API_get_input_batch_by_id_call(api_endpoint_path, location_id_for_input_batch, input_batch_id))

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
            print('Get output batch by id')
            print(output_batch_response)

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
            for sensor_id in self.object_identifier_mappings[pred_obj_id]:
                for initial_obj_id in self.object_identifier_mappings[pred_obj_id][sensor_id]:
                    pred_obj_id_list.append(pred_obj_id)
                    sensor_id_list.append(sensor_id)
                    pred_initial_obj_id_list.append(initial_obj_id)


        object_identifier_dataframe = pd.DataFrame({'sensor_id_from_api_output': sensor_id_list,
                   'pred_obj_id': pred_obj_id_list,
                   'pred_initial_obj_id': pred_initial_obj_id_list})
        
        measurement_df = pd.DataFrame(self.measurement_data)

        merged_identifier_df = pd.merge(measurement_df, object_identifier_dataframe, left_on=['object_identifier','sensor_identifier'], right_on = ['pred_initial_obj_id','sensor_id_from_api_output'])

        return merged_identifier_df

    # TODO: Write documentation
    def add_prediction_data_to_merged_identifier_dataframe(self):

        merged_identifier_dataframe = self.add_object_identifier_mapping_to_measurement_dataframe()


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
        
        prediction_df = pd.DataFrame({'temp_pred_obj_identifier': pred_obj_id_list,
                   'prediction_timestamp': pred_timestamp_list,
                   'prediction_confidence': pred_confidence_list,
                   'predicted_x': pred_x_list,
                   'predicted_y': pred_y_list,
                   'predicted_z': pred_z_list,
                   })

        merged_prediction_df = pd.merge(merged_identifier_dataframe, prediction_df, how ='left', left_on=['pred_obj_id','timestamp'], right_on = ['temp_pred_obj_identifier','prediction_timestamp']).drop('temp_pred_obj_identifier', axis=1)

        return merged_prediction_df

    # TODO: Write documentation
    def calculate_object_matching_accuracy(self):

        dataframe_with_matched_object_ids = self.add_object_identifier_mapping_to_measurement_dataframe()

        #dataframe_with_matched_object_ids['object_id_matched_correctly'] = [(dataframe_with_matched_object_ids['pred_initial_obj_id'] == dataframe_with_matched_object_ids['object_identifier']) for x in dataframe_with_matched_object_ids['pred_initial_obj_id']]

        dataframe_with_matched_object_ids['object_id_matched_correctly'] = np.where(dataframe_with_matched_object_ids['pred_initial_obj_id'] == dataframe_with_matched_object_ids['pred_initial_obj_id'], 'True', 'False')

        object_matching_accuarcy = (sum(dataframe_with_matched_object_ids['object_id_matched_correctly'] == 'True')) / dataframe_with_matched_object_ids.shape[0]
        
        return object_matching_accuarcy

    # TODO: Write documentation
    # TODO: Test when sufficient change in API was provided
    def calculate_prediction_accuracy(self, accuracy_estimation_method = 'euclidean-distance'):

        prediction_dataframe = self.add_prediction_data_to_merged_identifier_dataframe()
        print(prediction_dataframe)

        def calculate_euclidean_distance_between_two_points(true_x, true_y, true_z, pred_x, pred_y, pred_z):

            distance = ((true_x - pred_x)**2 + (true_y - pred_y)**2 + (true_z - pred_z)**2)**(0.5)

            return distance

        if accuracy_estimation_method == 'euclidean-distance':
            prediction_dataframe['prediction_error'] = [calculate_euclidean_distance_between_two_points(prediction_dataframe['x'], prediction_dataframe['y'], prediction_dataframe['z'], prediction_dataframe['predicted_x'], prediction_dataframe['predicted_y'], prediction_dataframe['predicted_z']) for x in prediction_dataframe]
        
        else:
            raise ValueError('Please select valid accuracy estimation method')

        prediction_accuracy = prediction_dataframe['prediction_error'].sum()

        return prediction_accuracy



# Test setup parameters 
endpoint_path = 'http://localhost:5000'

test_coord_sys = CoordinateSystem(6,-2,4, 0,0,0)
test_coord_sys2 = CoordinateSystem(0,-1,1, 2,3,4)
test_sensor = Sensor('RFID', test_coord_sys, 40, 20, 1000) # 20ms seems reasonable (https://electronics.stackexchange.com/questions/511278/low-latency-passive-rfid-solution)
test_sensor3 = Sensor('camera', test_coord_sys, 10, 17, 5000) # 16.666 ms is equal to 60fps
test_sensor2 = Sensor('WiFi 2.4GHz', test_coord_sys2, 30, 3, 4000) # 3ms is average wifi latency

# Generate test_location
test_location = Location('test_name', 'test_id_ext', [test_sensor,test_sensor2, test_sensor3])


# api_output, measurement_data = APIClient.end_to_end_API_test(test_location,[test_sensor, test_sensor2, test_sensor3], \
#                                                             endpoint_path, measurement_points= 100, print_progress= False)



# Generate fake data as long as Bug #13 exists in API
f_input, f_output = generate_fake_data()

test = PredictionEvaluator(f_output, f_input)

test.calculate_prediction_accuracy()

