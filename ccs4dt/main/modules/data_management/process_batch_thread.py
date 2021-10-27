import threading

import pandas as pd

from ccs4dt.main.modules.conversion.converter import Converter
from ccs4dt.main.shared.enums.input_batch_status import InputBatchStatus


class ProcessBatchThread(threading.Thread):
    """Handle the processing of an input data batch and store the result to influxDB."""
    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = kwargs['input_batch_service']
        self.__location_service = kwargs['location_service']
        self.__location_id = kwargs['location_id']
        self.__input_batch_id = kwargs['input_batch_id']
        self.__input_batch_df = pd.DataFrame(kwargs['input_batch'])
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        """Run the input batch processing"""
        try:
            self.__update_status(InputBatchStatus.PROCESSING)

            self.__convert()
            self.__cluster()
            self.__predict()
            self.__persist()

            self.__update_status(InputBatchStatus.FINISHED)
        except Exception as e:
            self.__update_status(InputBatchStatus.FAILED)
            raise e

    def __convert(self):
        """Convert input data batch into a standardized and shared format"""
        location = self.__location_service.get_by_id(self.__location_id)
        converter = Converter(self.__input_batch_df)

        for sensor in location['sensors']:
            converter.add_sensor(
                sensor['identifier'],
                sensor['x_origin'],
                sensor['y_origin'],
                sensor['z_origin'],
                sensor['yaw'],
                sensor['pitch'],
                sensor['roll'],
                sensor['measurement_unit']
            )

        self.__input_batch_df = converter.run()

    def __cluster(self):
        pass

    def __predict(self):
        pass

    def __persist(self):
        output_batch = self.__input_batch_df.T.to_dict().values()
        self.__input_batch_service.save_batch_to_influx(self.__input_batch_id, output_batch)
        pass

    def __update_status(self, new_status):
        self.__input_batch_service.update_status(self.__input_batch_id, new_status)
