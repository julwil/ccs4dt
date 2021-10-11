from http import HTTPStatus

import pytest

from ccs4dt import app
from ccs4dt.tests.integration.test_location_controller import test_post as test_post_location


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_all(client):
    location_id = test_post_location(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/input-batches')
    assert response.status_code == HTTPStatus.OK


def test_get_by_id(client):
    location_id = test_post_location(client).get_json()["id"]
    input_batch_id = test_post(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/input-batches/{input_batch_id}')
    assert response.status_code == HTTPStatus.OK


def test_post(client):
    location_id = test_post_location(client).get_json()["id"]
    response = client.post(f'/locations/{location_id}/input-batches', json=get_input_batch_dummy())
    assert response.status_code == HTTPStatus.ACCEPTED
    return response


def get_input_batch_dummy():
    return [
        {
            "object_identifier": "a1d0c6e83f027327d8461063f4ac58a6",
            "x": 55,
            "y": 35,
            "z": 1,
            "sensor_id": 1,
            "sensor_type": "camera",
            "timestamp": 1633859021123456000,
            "payload": {
                "gender": "female"
            }
        },
        {
            "object_identifier": "a1d0c6e83f027327d8461063f4ac58a6",
            "x": 57,
            "y": 33,
            "z": 1,
            "sensor_id": 1,
            "sensor_type": "rfid",
            "timestamp": 1633859021123456000
        },
        {
            "object_identifier": "a1d0c6e83f027327d8461063f4ac58a6",
            "x": 65,
            "y": 27,
            "z": 1,
            "sensor_id": 2,
            "sensor_type": "wifi",
            "timestamp": 1633859021123456000
        }
    ]
