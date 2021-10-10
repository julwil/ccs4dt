import json
from http import HTTPStatus

from flask import request, Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.space_service import SpaceService
from ccs4dt.main.shared.database import core_db

# TODO: Implement CRUDs and some input validation

space_service = SpaceService(core_db)


@app.route('/spaces', endpoint='spaces_get_all', methods=['GET'])
def get_all():
    spaces = space_service.get_all()
    return Response(json.dumps(spaces), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces/<space_id>', endpoint='spaces_get_by_id', methods=['GET'])
def get_by_id(space_id):
    space_id = int(space_id)
    space = space_service.get_by_id(space_id)
    return Response(json.dumps(space), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces', endpoint='spaces_post', methods=['POST'])
def post():
    space = space_service.create(request.get_json())
    return Response(json.dumps(space), status=HTTPStatus.CREATED, mimetype='application/json')


@app.route('/spaces/<space_id>', endpoint='spaces_put', methods=['PUT'])
def put(space_id):
    space_id = int(space_id)
    space = space_service.update(space_id, request.get_json())
    return Response(json.dumps(space), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/spaces/<space_id>', endpoint='spaces_delete', methods=['DELETE'])
def delete(space_id):
    space_id = int(space_id)
    space_service.delete(space_id)
    return Response(None, status=HTTPStatus.NO_CONTENT, mimetype='application/json')
