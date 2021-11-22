import threading

import numpy as np
import pandas as pd

from ccs4dt.main.modules.conversion.converter import Converter
from ccs4dt.main.modules.object_matching.object_matcher import ObjectMatcher
from ccs4dt.main.modules.smoothing.smoother import Smoother
from ccs4dt.main.modules.upsampling.upsampler import Upsampler
from ccs4dt.main.shared.enums.input_batch_status import InputBatchStatus


class ProcessBatchThread(threading.Thread):
    """Handle the processing of an input data batch and store the result to influxDB."""

    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = kwargs['input_batch_service']
        self.__location_service = kwargs['location_service']
        self.__object_identifier_mapping_service = kwargs['object_identifier_mapping_service']
        self.__location_id = kwargs['location_id']
        self.__input_batch_id = kwargs['input_batch_id']
        self.__input_batch_df = pd.DataFrame(kwargs['input_batch'])
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        """Run the input batch processing"""
        try:
            self.__update_status(InputBatchStatus.PROCESSING)

            self.__unique_identifiers()
            self.__convert()
            self.__upsample()
            self.__smoothe()
            self.__object_matching()
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

    def __upsample(self):
        upsampler = Upsampler(self.__input_batch_df)
        self.__input_batch_df = upsampler.run()

    def __smoothe(self):
        """Apply smoothing to raw sensor data to remove noise"""
        smoother = Smoother(self.__input_batch_df)
        self.__input_batch_df = smoother.run()

    def __object_matching(self):
        object_matcher = ObjectMatcher(self.__input_batch_df)
        self.__input_batch_df = object_matcher.run()

        for object_identifier, cluster in object_matcher.get_clusters().items():
            for external_object_identifier in cluster:
                self.__object_identifier_mapping_service.create(self.__input_batch_id, object_identifier,
                                                                external_object_identifier)

    def __predict(self):
        pass

    def __persist(self):
        self.__input_batch_df.drop(['object_identifier'], axis=1, inplace=True)
        self.__input_batch_df.reset_index(drop=False, inplace=True)
        self.__input_batch_df['timestamp'] = self.__input_batch_df['timestamp'].view(
            np.int64) // 10 ** 6  # Convert back timestamp
        output_batch = self.__input_batch_df.to_dict(orient='records')
        self.__input_batch_service.save_batch_to_influx(self.__input_batch_id, output_batch)

    def __update_status(self, new_status):
        self.__input_batch_service.update_status(self.__input_batch_id, new_status)

    def __unique_identifiers(self):
        self.__input_batch_df['object_identifier'] = self.__input_batch_df['object_identifier'] + self.__input_batch_df['sensor_identifier']