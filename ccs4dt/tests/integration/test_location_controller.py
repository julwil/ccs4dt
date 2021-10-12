from http import HTTPStatus

import pytest

from ccs4dt import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_all(client):
    response = client.get('/locations')
    assert response.status_code == HTTPStatus.OK


def test_get_by_id(client):
    post_response = test_post(client)
    id = post_response.get_json()["id"]
    get_response = client.get(f'/locations/{id}')
    assert get_response.status_code == HTTPStatus.OK


def test_post(client):
    response = client.post('/locations', json=get_location_dummy())
    assert response.status_code == HTTPStatus.CREATED
    return response


def get_location_dummy():
    return {
        "name": "BIN-2.A.10",
        "external_identifier": "room_1234",
        "sensors": [
            {
                "identifier": "abc123",
                "type": "rfid",
                "x_origin": 0,
                "y_origin": 10,
                "z_origin": 3,
                "measurement_unit": "meter",
                "payload_properties": [
                    "gender"
                ]
            }
        ]
    }
