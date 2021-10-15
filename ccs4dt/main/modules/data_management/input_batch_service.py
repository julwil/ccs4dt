import logging
import time
from datetime import datetime

from influxdb_client import Point

from ccs4dt.main.modules.data_management.process_batch_thread import ProcessBatchThread


class InputBatchService:
    def __init__(self, core_db, influx_db, output_batch_service, location_service):
        self.__core_db = core_db
        self.__influx_db = influx_db
        self.__output_batch_service = output_batch_service
        self.__location_service = location_service
        self.STATUS_SCHEDULED = 'scheduled'
        self.STATUS_PROCESSING = 'processing'
        self.STATUS_FINISHED = 'finished'
        self.STATUS_FAILED = 'failed'

    def create(self, location_id, batch):
        input_batch_id = self.__core_db.input_batch_table.insert({
            'location_id': location_id,
            'status': self.STATUS_SCHEDULED,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        input_batch = self.get_by_id(input_batch_id)
        output_batch = self.__output_batch_service.create(location_id, input_batch_id)
        self.update(input_batch_id, {'output_batch_id': output_batch['id'], **input_batch})

        ProcessBatchThread(self, self.__location_service, kwargs={
            'location_id': location_id,
            'input_batch_id': input_batch_id,
            'output_batch_id': output_batch['id'],
            'batch': batch
        }).start()

        return self.get_by_id(input_batch_id)

    def get_by_id(self, input_batch_id):
        return {
            'id': input_batch_id,
            **dict(self.__core_db.input_batch_table.get(doc_id=input_batch_id))
        }

    def update(self, input_batch_id, data):
        self.__core_db.input_batch_table.update(data, doc_ids=[input_batch_id])
        return self.get_by_id(input_batch_id)


    def update_status(self, input_batch_id, new_status):
        if new_status not in [self.STATUS_SCHEDULED, self.STATUS_PROCESSING, self.STATUS_FINISHED, self.STATUS_FAILED]:
            raise RuntimeError(f'unknown input batch status {new_status}')

        input_batch = self.get_by_id(input_batch_id)
        input_batch['status'] = new_status
        self.update(input_batch_id, input_batch)

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
        return [self.get_by_id(input_batch.doc_id) for input_batch in self.__core_db.input_batch_table.all()]
