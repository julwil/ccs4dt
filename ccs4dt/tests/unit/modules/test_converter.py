import time

import pandas as pd

from ccs4dt.main.modules.conversion.converter import Converter


def test_coordinate_offset_no_rotation():
    input = pd.DataFrame([
        {
            "object_identifier": "my-object",
            "x": 1,
            "y": 0.5,
            "z": 0,
            "sensor_identifier": "sensor_1",
            "sensor_type": "rfid",
            "timestamp": int(time.time())
        }
    ])
    converter = Converter(input)
    converter.add_sensor('sensor_1', x_origin=1, y_origin=1.5, z_origin=0, yaw=0, pitch=0, roll=0, measurement_unit='cm')
    output = converter.run()

    assert output.shape == input.shape
    assert round(output.loc[0]['x'], 1) == 2.0
    assert round(output.loc[0]['y'], 1) == 2.0
    assert round(output.loc[0]['z'], 1) == 0.0


def test_coordinate_offset_with_rotation():
    input = pd.DataFrame([
        {
            "object_identifier": "my-object",
            "x": -0.025,
            "y": 0.06,
            "z": 0,
            "sensor_identifier": "sensor_2",
            "sensor_type": "rfid",
            "timestamp": int(time.time())
        }
    ])
    converter = Converter(input)
    converter.add_sensor('sensor_2', x_origin=8, y_origin=4.5, z_origin=0, yaw=90, pitch=0, roll=0, measurement_unit='m')
    output = converter.run()

    assert output.shape == input.shape
    assert round(output.loc[0]['x'], 1) == 2.0
    assert round(output.loc[0]['y'], 1) == 2.0
    assert round(output.loc[0]['z'], 1) == 0.0
