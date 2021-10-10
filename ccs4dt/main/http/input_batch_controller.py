import json
from http import HTTPStatus

from flask import request, Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.input_batch_service import InputBatchService
from ccs4dt.main.modules.data_management.output_batch_service import OutputBatchService
from ccs4dt.main.shared.database import core_db, influx_db

output_batch_service = OutputBatchService(core_db)
input_batch_service = InputBatchService(core_db, influx_db, output_batch_service)


# TODO: Implement CRUDs and some input validation

@app.route('/spaces/<space_id>/input-batches', endpoint='input_batches_get_all', methods=['GET'])
def get_all(space_id):
    space_id = int(space_id)
    return Response(json.dumps([]), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces/<space_id>/input-batches/<batch_id>', endpoint='input_batches_get_by_id', methods=['GET'])
def get_by_id(space_id, batch_id):
    space_id = int(space_id)
    batch_id = int(batch_id)
    return Response(json.dumps(request.get_json()), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces/<space_id>/input-batches', endpoint='input_batches_post', methods=['POST'])
def post(space_id):
    space_id = int(space_id)
    input_batch = input_batch_service.create(request.get_json())
    return Response(json.dumps(input_batch), status=HTTPStatus.ACCEPTED, mimetype='application/json')
