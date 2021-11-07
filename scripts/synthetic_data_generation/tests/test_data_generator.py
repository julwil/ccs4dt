from scripts.synthetic_data_generation.main.transform_test_data_set import CoordinateSystem, Sensor, Location
import json

def test_location_json_generation():

    test_coord_sys = CoordinateSystem(6,-2,4, 0,0,0)
    test_coord_sys2 = CoordinateSystem(0,-1,1, 2,3,4)
    test_sensor = Sensor('RFID', test_coord_sys, 30, 10, 500, sensor_identifier='a')
    test_sensor3 = Sensor('camera', test_coord_sys, 1, 1, 500, sensor_identifier='b')
    test_sensor2 = Sensor('NFC', test_coord_sys2, 30, 10, 500, sensor_identifier='c')

    test_location = Location('test_name', 'test_id_ext', [test_sensor,test_sensor2, test_sensor3])
    location_json_payload = test_location.construct_json_payload()

    correct_payload = json.load(open('./scripts/synthetic_data_generation/tests/location_payload_dummy.json', 'r'))

    assert(location_json_payload == correct_payload)