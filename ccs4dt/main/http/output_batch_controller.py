import json
from http import HTTPStatus

from flask import Response

from ccs4dt import app


# TODO: Implement CRUDs and some input validation

@app.route('/spaces/<space_id>/output-batches', endpoint='output_batches_get_all', methods=['GET'])
def get_all(space_id):
    return Response(json.dumps([]), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces/<space_id>/output-batches/<batch_id>', endpoint='output_batches_get_by_id', methods=['GET'])
def get_by_id(space_id, batch_id):
    return Response(json.dumps({'id': batch_id}), status=HTTPStatus.OK, mimetype='application/json')
