from http import HTTPStatus

import pytest

from ccs4dt import app
from ccs4dt.tests.integration.test_input_batch_controller import test_post as test_post_input_batch
from ccs4dt.tests.integration.test_location_controller import test_post as test_post_location


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_get_all(client):
    location_id = test_post_location(client).get_json()["id"]
    response = client.get(f'/locations/{location_id}/output-batches')
    assert response.status_code == HTTPStatus.OK


def test_get_by_id(client):
    location_id = test_post_location(client).get_json()["id"]
    output_batch_id = test_post_input_batch(client).get_json()["output_batch_id"]
    response = client.get(f'/locations/{location_id}/output-batches/{output_batch_id}')
    assert response.status_code == HTTPStatus.OK
