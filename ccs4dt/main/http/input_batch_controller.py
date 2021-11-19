import json
from http import HTTPStatus

from flask import request, Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.input_batch_service import InputBatchService
from ccs4dt.main.modules.data_management.location_service import LocationService
from ccs4dt.main.modules.data_management.object_matching_service import ObjectMatchingService
from ccs4dt.main.shared.database import core_db, influx_db

location_service = LocationService(core_db)
object_matching_service = ObjectMatchingService(core_db)
input_batch_service = InputBatchService(core_db, influx_db, location_service, object_matching_service)

@app.route('/locations/<location_id>/inputs', endpoint='input_batches_get_all', methods=['GET'])
def get_all(location_id):
    """
    Gets all input batches of given location

    :param location_id: id of the location
    :type location_id: str
    :rtype: flask.Response
    """
    location_id = int(location_id)
    input_batches = input_batch_service.get_all_by_location_id(location_id)
    return Response(json.dumps(input_batches), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations/<location_id>/inputs/<batch_id>', endpoint='input_batches_get_by_id', methods=['GET'])
def get_input_by_id(location_id, batch_id):
    """
    Gets an input batch of a given location by id

    :param location_id: id of the location
    :type location_id: str
    :param batch_id: id of the batch
    :type batch_id: str
    :rtype: flask.Response
    """
    location_id = int(location_id)
    batch_id = int(batch_id)
    input_batch = input_batch_service.get_by_id(batch_id)
    return Response(json.dumps(input_batch), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations/<location_id>/inputs', endpoint='input_batches_post', methods=['POST'])
def post(location_id):
    """
    Create a new input batch for a given location. Input batch will be processed async and results can be polled when the computation is finished

    :param location_id: id of the location
    :type location_id: str
    :rtype: flask.Response
    """
    location_id = int(location_id)
    input_batch = request.get_json()
    input_batch = input_batch_service.create(location_id, input_batch)
    return Response(json.dumps(input_batch), status=HTTPStatus.ACCEPTED, mimetype='application/json')

@app.route('/locations/<location_id>/inputs/<batch_id>/outputs', endpoint='output_batches_get_by_id', methods=['GET'])
def get_output_by_id(location_id, batch_id):
    """
    Gets an output batch of a given location and input   batch

    :param location_id: id of the location
    :type location_id: str
    :param batch_id: id of the input batch
    :type batch_id: str
    :rtype: flask.Response
    """
    batch_id = int(batch_id)
    output_batch = input_batch_service.get_output_by_id(batch_id)
    return Response(json.dumps(output_batch), status=HTTPStatus.OK, mimetype='application/json')
