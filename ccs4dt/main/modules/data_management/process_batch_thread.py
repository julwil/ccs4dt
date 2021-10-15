import threading, logging
import pandas as pd
from ccs4dt.main.modules.conversion.converter import Converter
import traceback

class ProcessBatchThread(threading.Thread):

    def __init__(self, input_batch_service, location_service, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = input_batch_service
        self.__location_service = location_service
        self.__location_id = kwargs['location_id']
        self.__input_batch_id = kwargs['input_batch_id']
        self.__output_batch_id = kwargs['output_batch_id']
        self.__batch_df = pd.DataFrame(kwargs['batch'])
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        try:
            self.__input_batch_service.update_status(self.__input_batch_id, self.__input_batch_service.STATUS_PROCESSING)
            batch = self.convert(self.__batch_df)
            logging.error(batch)

            # self.__input_batch_service.save_batch_to_influx(self.args)
            # Coordinate Transformation
            # Clustering
            # Prediction
            # Update output batch

            # influx_db.write_api.write("ccs4dt", "ccs4dt", ["h2o_feet,location=coyote_creek water_level=1".encode()])
        except:
            self.__input_batch_service.update_status(self.__input_batch_id, self.__input_batch_service.STATUS_FAILED)
            traceback.print_exc()

    def convert(self, batch_df):
        location = self.__location_service.get_by_id(self.__location_id)
        converter = Converter(batch_df)
        for sensor in location['sensors']:
            converter.add_sensor(
                sensor['identifier'],
                sensor['x_origin'],
                sensor['y_origin'],
                sensor['z_origin'],
                sensor['orientation'],
                sensor['measurement_unit']
            )
        return converter.convert()