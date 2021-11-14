import pytest
import time
from http import HTTPStatus

from ccs4dt import app
from ccs4dt.tests.integration.test_location_controller import test_post as test_post_location


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_all(client):
    location_id = test_post_location(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/inputs')
    assert response.status_code == HTTPStatus.OK


def test_get_by_id(client):
    location_id = test_post_location(client).get_json()["id"]
    input_batch_id = test_post(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/inputs/{input_batch_id}')
    assert response.status_code == HTTPStatus.OK


def test_post(client):
    location_id = test_post_location(client).get_json()["id"]
    response = client.post(f'/locations/{location_id}/inputs', json=get_input_batch_dummy())
    assert response.status_code == HTTPStatus.ACCEPTED
    return response


def test_get_outputs(client):
    location_id = test_post_location(client).get_json()["id"]
    input_batch_id = test_post(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/inputs/{input_batch_id}/outputs')
    assert response.status_code == HTTPStatus.OK


def get_input_batch_dummy():
    return [
        {
            "object_identifier": "cam_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "cam_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "cam_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359780808
        },
        {
            "object_identifier": "cam_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "cam_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "cam_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359780808
        },
        {
            "object_identifier": "cam_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "cam_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "cam_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "camera",
            "sensor_type": "camera",
            "timestamp": 1636359780808
        },
        {
            "object_identifier": "rfid_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "rfid_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "rfid_obj_1",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359780808
        },
        {
            "object_identifier": "rfid_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "rfid_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "rfid_obj_2",
            "x": 5,
            "y": 7,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359780808
        },
        {
            "object_identifier": "rfid_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359776808
        },
        {
            "object_identifier": "rfid_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359777808
        },
        {
            "object_identifier": "rfid_obj_3",
            "x": 9,
            "y": 3,
            "z": 1,
            "sensor_identifier": "rfid",
            "sensor_type": "rfid",
            "timestamp": 1636359780808
        }
    ]
