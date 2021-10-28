import threading

import pandas as pd
import numpy as np

from ccs4dt.main.modules.conversion.converter import Converter
from ccs4dt.main.modules.clustering.clusterer import Clusterer
from ccs4dt.main.shared.enums.input_batch_status import InputBatchStatus


class ProcessBatchThread(threading.Thread):
    """Handle the processing of an input data batch and store the result to influxDB."""
    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = kwargs['input_batch_service']
        self.__location_service = kwargs['location_service']
        self.__location_id = kwargs['location_id']
        self.__input_batch_id = kwargs['input_batch_id']
        self.__input_batch_df = self.__init_input_batch_df(kwargs['input_batch'])
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)


    def __init_input_batch_df(self, input_batch):
        """
        Create pd.DataFrame form input_batch

        :param input_batch: Input batch as list
        :type input_batch: list
        :returns: Input Batch as pd.DataFrame
        :rtype: pd.DataFrme
        """
        df = pd.DataFrame(input_batch)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').round('1s')
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.set_index(keys=['timestamp', 'object_identifier'], inplace=True)
        return df

    def run(self):
        """Run the input batch processing"""
        try:
            self.__update_status(InputBatchStatus.PROCESSING)

            self.__convert()
            self.__cluster() # TODO rename
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
        clusterer = Clusterer(self.__input_batch_df)
        self.__input_batch_df = clusterer.run()

    def __predict(self):
        pass

    def __persist(self):
        self.__input_batch_df['timestamp'] = self.__input_batch_df.index.get_level_values(0)
        self.__input_batch_df['timestamp'] = self.__input_batch_df['timestamp'].view(np.int64) // 10**6
        output_batch = self.__input_batch_df.T.to_dict().values()
        self.__input_batch_service.save_batch_to_influx(self.__input_batch_id, output_batch)

    def __update_status(self, new_status):
        self.__input_batch_service.update_status(self.__input_batch_id, new_status)
