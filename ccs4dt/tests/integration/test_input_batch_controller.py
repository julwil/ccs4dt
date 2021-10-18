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
            "object_identifier": "my-object",
            "x": 1,
            "y": 0.5,
            "z": 1,
            "sensor_identifier": "sensor_1",
            "sensor_type": "rfid",
            "timestamp": int(time.time())
        },
        {
            "object_identifier": "my-object",
            "x": -2.5,
            "y": 6,
            "z": 1,
            "sensor_identifier": "sensor_",
            "sensor_type": "rfid",
            "timestamp": int(time.time())
        }
    ]
