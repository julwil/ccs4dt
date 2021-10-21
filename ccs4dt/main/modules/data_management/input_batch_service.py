from datetime import datetime

from influxdb_client import Point

from ccs4dt.main.modules.data_management.process_batch_thread import ProcessBatchThread
from ccs4dt.main.shared.enums.input_batch_status import InputBatchStatus

class InputBatchService:
    """
    InputBatchService responsible for handling input batches

    :param core_db: database connection of core_db
    :type core_db: CoreDB
    :param influx_db: database connection of influx_db
    :type influx_db: InfluxDB
    :param location_service: location service
    :type location_service: LocationService
    """
    def __init__(self, core_db, influx_db, location_service):
        self.__core_db = core_db
        self.__influx_db = influx_db
        self.__location_service = location_service

    def create(self, location_id, batch):
        """
        Start the processing of the input batch async in a new thread

        :param location_id: id of the location
        :type location_id: int
        :param batch: the input data batch
        :type batch: list
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''INSERT INTO input_batches (location_id, status, created_at) VALUES(?,?,?)'''
        input_batch_id = connection.cursor().execute(query, (location_id, InputBatchStatus.SCHEDULED, datetime.now())).lastrowid
        connection.commit()

        ProcessBatchThread(kwargs={
            'input_batch_service': self,
            'location_service': self.__location_service,
            'location_id': location_id,
            'input_batch_id': input_batch_id,
            'batch': batch
        }).start()

        return self.get_by_id(input_batch_id)

    def get_by_id(self, input_batch_id):
        """
        Get an input batch by id

        :param input_batch_id: id of the input batch
        :type input_batch_id: int
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''SELECT * FROM input_batches WHERE id=?'''
        return dict(connection.cursor().execute(query, (input_batch_id,)).fetchone())

    def update(self, input_batch_id, data):
        """
        Update an input batch by id

        :param input_batch_id: id of the input batch
        :type input_batch_id: int
        :param data: the data to update
        :type: data: dict
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''UPDATE input_batches SET location_id=?, status=? WHERE id =?'''
        connection.cursor().execute(query, (data['location_id'], data['status'], input_batch_id))
        connection.commit()
        return self.get_by_id(input_batch_id)

    def update_status(self, input_batch_id, new_status):
        """
        Update status of an input batch by id

        :param input_batch_id: id of the input batch
        :type input_batch_id: int
        :param new_status: the new status
        :type new_status: str
        :rtype: dict
        """
        if new_status not in list(InputBatchStatus):
            raise RuntimeError(f'unknown input batch status {new_status}')

        input_batch = self.get_by_id(input_batch_id)
        input_batch['status'] = new_status
        self.update(input_batch_id, input_batch)
        return self.get_by_id(input_batch_id)

    def save_batch_to_influx(self, batch):
        """
        Save input batch to influxDB

        :param batch: input data
        """
        for measurement in batch:
            point = Point("raw_measurement") \
                .tag("identifier", measurement["object_identifier"]) \
                .tag("sensor_id", measurement["sensor_id"]) \
                .tag("sensor_type", measurement["sensor_type"]) \
                .field("x", measurement["x"]) \
                .field("y", measurement["y"]) \
                .field("z", measurement["z"]) \
                .time(measurement["timestamp"])
            self.__influx_db.write_api.write("ccs4dt", "ccs4dt", point)

    def get_all(self):
        """
        Get all input batches

        :rtype: list
        """
        connection = self.__core_db.connection()
        query = '''SELECT id FROM input_batches WHERE TRUE'''
        input_batch_ids = [dict(input_batch)['id'] for input_batch in connection.cursor().execute(query).fetchall()]
        return [self.get_by_id(id) for id in input_batch_ids]
