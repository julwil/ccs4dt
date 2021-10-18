from datetime import datetime

from influxdb_client import Point

from ccs4dt.main.modules.data_management.process_batch_thread import ProcessBatchThread


class InputBatchService:
    def __init__(self, core_db, influx_db):
        self.__core_db = core_db
        self.__influx_db = influx_db
        self.STATUS_SCHEDULED = 'scheduled'
        self.STATUS_PROCESSING = 'processing'
        self.STATUS_FINISHED = 'finished'
        self.STATUS_FAILED = 'failed'

    def create(self, location_id, batch):
        connection = self.__core_db.connection()
        query = '''INSERT INTO input_batches (location_id, status, created_at) VALUES(?,?,?)'''
        input_batch_id = connection.cursor().execute(query, (location_id, self.STATUS_SCHEDULED, datetime.now())).lastrowid
        connection.commit()

        ProcessBatchThread(kwargs={
            'location_id': location_id,
            'input_batch_id': input_batch_id,
            'batch': batch
        }).start()

        return self.get_by_id(input_batch_id)

    def get_by_id(self, input_batch_id):
        connection = self.__core_db.connection()
        query = '''SELECT * FROM input_batches WHERE id=?'''
        return dict(connection.cursor().execute(query, (input_batch_id,)).fetchone())

    def update(self, input_batch_id, data):
        connection = self.__core_db.connection()
        query = '''UPDATE input_batches SET location_id=?, status=? WHERE id =?'''
        connection.cursor().execute(query, (data['location_id'], data['status'], input_batch_id))
        connection.commit()
        return self.get_by_id(input_batch_id)

    def update_status(self, input_batch_id, new_status):
        if new_status not in [self.STATUS_SCHEDULED, self.STATUS_PROCESSING, self.STATUS_FINISHED, self.STATUS_FAILED]:
            raise RuntimeError(f'unknown input batch status {new_status}')

        input_batch = self.get_by_id(input_batch_id)
        input_batch['status'] = new_status
        self.update(input_batch_id, input_batch)
        return self.get_by_id(input_batch_id)

    def save_batch_to_influx(self, batch):
        """
        Write a received batch to influxDB.

        This is an example how we can ingest data into influxDB

        :param batch:
        :return:
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
        connection = self.__core_db.connection()
        query = '''SELECT id FROM input_batches WHERE TRUE'''
        input_batch_ids = [dict(input_batch)['id'] for input_batch in connection.cursor().execute(query).fetchall()]
        return [self.get_by_id(id) for id in input_batch_ids]
