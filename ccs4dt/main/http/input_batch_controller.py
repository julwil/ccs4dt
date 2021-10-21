import json
from http import HTTPStatus

from flask import request, Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.input_batch_service import InputBatchService
from ccs4dt.main.shared.database import core_db, influx_db

input_batch_service = InputBatchService(core_db, influx_db)


# TODO: Implement CRUDs and some input validation

@app.route('/locations/<location_id>/inputs', endpoint='input_batches_get_all', methods=['GET'])
def get_all(location_id):
    location_id = int(location_id)
    input_batches = input_batch_service.get_all()
    return Response(json.dumps(input_batches), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations/<location_id>/inputs/<batch_id>', endpoint='input_batches_get_by_id', methods=['GET'])
def get_by_id(location_id, batch_id):
    location_id = int(location_id)
    batch_id = int(batch_id)
    input_batch = input_batch_service.get_by_id(batch_id)
    return Response(json.dumps(input_batch), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations/<location_id>/inputs', endpoint='input_batches_post', methods=['POST'])
def post(location_id):
    location_id = int(location_id)
    input_batch = request.get_json()
    input_batch = input_batch_service.create(location_id, input_batch)
    return Response(json.dumps(input_batch), status=HTTPStatus.ACCEPTED, mimetype='application/json')

@app.route('/locations/<location_id>/inputs/<batch_id>/outputs', endpoint='output_batches_get_by_id', methods=['GET'])
def get_by_id(location_id, batch_id):
    return Response(json.dumps({'id': batch_id}), status=HTTPStatus.OK, mimetype='application/json')
