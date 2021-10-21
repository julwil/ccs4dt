import json
from http import HTTPStatus

from flask import request, Response

from ccs4dt import app
from ccs4dt.main.modules.data_management.location_service import LocationService
from ccs4dt.main.shared.database import core_db

# TODO: Implement CRUDs and some input validation

location_service = LocationService(core_db)


@app.route('/locations', endpoint='locations_get_all', methods=['GET'])
def get_all():
    """
    Get all locations

    :return: flask.Response
    """
    locations = location_service.get_all()
    return Response(json.dumps(locations), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations/<location_id>', endpoint='locations_get_by_id', methods=['GET'])
def get_by_id(location_id):
    """
    Get a location by id

    :param location_id: id of the location
    :return: flask.Response
    """
    location_id = int(location_id)
    location = location_service.get_by_id(location_id)
    return Response(json.dumps(location), status=HTTPStatus.OK, mimetype='application/json')


@app.route('/locations', endpoint='locations_post', methods=['POST'])
def post():
    """
    Create a new location
    
    :return: flask.Response
    """
    location = location_service.create(request.get_json())
    return Response(json.dumps(location), status=HTTPStatus.CREATED, mimetype='application/json')
