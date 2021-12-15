import json
from http import HTTPStatus

from flask import Response

from ccs4dt import app


# TODO: Implement CRUDs and some input validation


@app.route('/locations/<location_id>/output-batches/<batch_id>', endpoint='output_batches_get_by_id', methods=['GET'])
def get_by_id(location_id, batch_id):
    return Response(json.dumps({'id': batch_id}), status=HTTPStatus.OK, mimetype='application/json')
