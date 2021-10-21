class LocationService:
    """
    Location service responsible to manage locations

    :param core_db: connection of the core_db
    :type core_db: CoreDB
    """
    def __init__(self, core_db):
        self.__core_db = core_db

    def create(self, data):
        """
        Create a new location

        :param data: location data
        :type data: dict
        :rtype: dict
        """
        connection = self.__core_db.connection()
        location_query = '''INSERT INTO locations (name, external_identifier) VALUES (:name, :external_identifier)'''
        location_id = connection.cursor().execute(location_query, data).lastrowid

        sensor_query = '''
                INSERT INTO sensors 
                (location_id, identifier, type, x_origin, y_origin, z_origin, yaw, pitch, roll, measurement_unit)
                VALUES (:location_id, :identifier, :type, :x_origin, :y_origin, :z_origin, :yaw, :pitch, :roll, :measurement_unit)
            '''

        for sensor in data['sensors']:
            sensor['location_id'] = location_id
            connection.cursor().execute(sensor_query, sensor)

        connection.commit()
        return self.get_by_id(location_id)

    def get_by_id(self, location_id):
        """
        Get a location by id

        :param location_id: id of the location
        :type location_id: int
        :rtype: dict
        """
        connection = self.__core_db.connection()
        location_query = '''SELECT * FROM locations WHERE id=?'''
        sensor_query = '''SELECT * FROM sensors WHERE location_id=?'''
        location = dict(connection.cursor().execute(location_query, (location_id,)).fetchone())
        sensors = [dict(sensor) for sensor in connection.cursor().execute(sensor_query, (location_id,)).fetchall()]
        location['sensors'] = sensors
        return location

    def get_all(self):
        """
        Get all locations

        :rtype: list
        """
        connection = self.__core_db.connection()
        query = '''SELECT id FROM locations WHERE TRUE'''
        location_ids = [dict(location)['id'] for location in connection.cursor().execute(query).fetchall()]
        return [self.get_by_id(id) for id in location_ids]
