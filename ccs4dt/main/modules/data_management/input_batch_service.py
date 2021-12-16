from collections import defaultdict
from datetime import datetime
from influxdb_client import Point
from influxdb_client.domain.write_precision import WritePrecision

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
    :param object_identifier_mapping_service: object matching service
    :type object_identifier_mapping_service: ObjectMatchingService
    """

    def __init__(self, core_db, influx_db, location_service, object_identifier_mapping_service):
        self.__core_db = core_db
        self.__influx_db = influx_db
        self.__location_service = location_service
        self.__object_identifier_mapping_service = object_identifier_mapping_service

    def create(self, location_id, input_batch):
        """
        Start the processing of the input batch async in a new thread

        :param location_id: id of the location
        :type location_id: int
        :param input_batch: the input data batch
        :type input_batch: list
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''INSERT INTO input_batches (location_id, status, created_at) VALUES(?,?,?)'''
        input_batch_id = connection.cursor().execute(query, (
        location_id, InputBatchStatus.SCHEDULED, datetime.now())).lastrowid
        connection.commit()

        ProcessBatchThread(kwargs={
            'input_batch_service': self,
            'location_service': self.__location_service,
            'object_identifier_mapping_service': self.__object_identifier_mapping_service,
            'location_id': location_id,
            'input_batch_id': input_batch_id,
            'input_batch': input_batch
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

    def get_output_by_id(self, input_batch_id):
        """
        Get output batch by input batch id.

        :param input_batch_id: id of input batch
        :type input_batch_id: int
        :rtype: dict
        """
        input_batch = self.get_by_id(input_batch_id)
        object_identifier_mappings = defaultdict(list)
        for mapping in self.__object_identifier_mapping_service.get_by_input_batch_id(input_batch_id):
            object_identifier_mappings[mapping['object_identifier']].append(mapping['external_object_identifier'])

        query = f'''
                from(bucket: "ccs4dt")
                  |> range(start: 1970-01-01T00:00:00Z)
                  |> filter(fn: (r) => r["_measurement"] == "object_positions")
                  |> filter(fn: (r) => r["_field"] == "confidence" or r["_field"] == "x" or r["_field"] == "y" or r["_field"] == "z")
                  |> filter(fn: (r) => r["input_batch_id"] == "{input_batch_id}")
                  |> group(columns: ["object_identifier"])
                '''
        positions = []

        for table in self.__influx_db.query_api.query(org='ccs4dt', query=query):
            object_positions = defaultdict(dict)
            object_identifier = ''
            for record in table.records:
                timestamp = int(record.get_time().timestamp() * 1000)
                object_identifier = record.values.get('object_identifier')
                object_positions[timestamp][record.get_field()] = record.get_value()

            for timestamp, fields in object_positions.items():
                positions.append({
                    'object_identifier': object_identifier,
                    'timestamp': timestamp,
                    **fields
                })

        return {
            'input_batch_id': input_batch['id'],
            'location_id': input_batch['location_id'],
            'object_identifier_mappings': object_identifier_mappings,
            'positions': positions
        }

    def save_batch_to_influx(self, input_batch_id, output_batch):
        """
        Save output batch to influxDB.

        :param input_batch_id: id of input batch
        :type input_batch_id: int
        :param output_batch: Result of input batch calculation
        :type: list
        """
        write_precision = WritePrecision.MS  # For now hardcoded to milliseconds
        for prediction in output_batch:
            point = Point("object_positions") \
                .tag("object_identifier", prediction['object_identifier']) \
                .tag("input_batch_id", input_batch_id) \
                .field("x", prediction["x"]) \
                .field("y", prediction["y"]) \
                .field("z", prediction["z"]) \
                .field("confidence", 1.0) \
                .time(prediction["timestamp"], write_precision=write_precision)
            self.__influx_db.write_api.write("ccs4dt", "ccs4dt", point, write_precision=write_precision)

    def get_all_by_location_id(self, location_id):
        """
        Get all input batches of the given location

        :rtype: list
        """
        connection = self.__core_db.connection()
        query = '''SELECT id FROM input_batches WHERE location_id=?'''
        input_batch_ids = [dict(input_batch)['id'] for input_batch in connection.cursor().execute(query, (location_id,)).fetchall()]
        return [self.get_by_id(id) for id in input_batch_ids]
