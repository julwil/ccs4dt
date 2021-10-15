import json
from http import HTTPStatus

from flask import Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.output_batch_service import OutputBatchService
from ccs4dt.main.shared.database import core_db

output_batch_service = OutputBatchService(core_db)


# TODO: Implement CRUDs and some input validation


@app.route('/locations/<location_id>/output-batches/<batch_id>', endpoint='output_batches_get_by_id', methods=['GET'])
def get_by_id(location_id, batch_id):
    batch_id = int(batch_id)
    output_batch = output_batch_service.get_by_id(batch_id)
    return Response(json.dumps(output_batch), status=HTTPStatus.OK, mimetype='application/json')
