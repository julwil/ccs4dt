import requests
import json
from scripts.synthetic_data_generation.main.transform_test_data_set import CoordinateSystem, Sensor, Location, simulate_sensor_measurement_for_multiple_sensors
import pandas as pd
import openpyxl


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
            json_data = dataframe.to_json(path_or_buf= 'scripts/synthetic_data_generation/assets/generated_files/measurement.json', default_handler=str, orient='records')
            json_data = dataframe.to_json(default_handler=str, orient='records')
        elif additional_file_generation == False:
            json_data = dataframe.to_json(default_handler=str, orient='records')

        return(json_data)

    def end_to_end_API_test(location, sensors, api_endpoint_path, measurement_points = 250, return_simulation_output = True, print_progress = True):

        # API request: GET all locations
        APIClient.API_get_all_locations_call(api_endpoint_path)

        # Construct test location payload
        location_payload = location.construct_json_payload()

        # API request: POST new location
        post_location_response, location_id, test_location_name = (APIClient.API_post_new_location_call(api_endpoint_path, location_payload))

        # Simulate sensor data measurement
        measurement_data = simulate_sensor_measurement_for_multiple_sensors(sensors, measurement_points)

        # Generate synthetic measurement data payload
        API_payload = APIClient.convert_sensor_measurements_to_api_conform_payload(measurement_data, additional_file_generation = True)

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

class PredictionEvaluator(object):

    def __init__(self) -> None:
        pass

    def gather_data():

        combined_dataframe = []

        return combined_dataframe


# Test setup parameters 
endpoint_path = 'http://localhost:5000'

test_coord_sys = CoordinateSystem(6,-2,4, 0,0,0)
test_coord_sys2 = CoordinateSystem(0,-1,1, 2,3,4)
test_sensor = Sensor('RFID', test_coord_sys, 40, 20, 1000) # 20ms seems reasonable (https://electronics.stackexchange.com/questions/511278/low-latency-passive-rfid-solution)
test_sensor3 = Sensor('camera', test_coord_sys, 10, 17, 5000) # 16.666 ms is equal to 60fps
test_sensor2 = Sensor('WiFi 2.4GHz', test_coord_sys2, 30, 3, 4000) # 3ms is average wifi latency

# Generate test_location
test_location = Location('test_name', 'test_id_ext', [test_sensor,test_sensor2, test_sensor3])


api_output, measurement_data = APIClient.end_to_end_API_test(test_location,[test_sensor, test_sensor2, test_sensor3],endpoint_path, measurement_points= 1500)

#b.to_excel('test.xlsx')