import threading

import pandas as pd

from ccs4dt.main.modules.conversion.converter import Converter
from ccs4dt.main.shared.enums.input_batch_status import InputBatchStatus


class ProcessBatchThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = kwargs['input_batch_service']
        self.__location_service = kwargs['location_service']
        self.__location_id = kwargs['location_id']
        self.__input_batch_id = kwargs['input_batch_id']
        self.__batch_df = pd.DataFrame(kwargs['batch'])
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
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
        location = self.__location_service.get_by_id(self.__location_id)
        converter = Converter(self.__batch_df)

        for sensor in location['sensors']:
            converter.add_sensor(
                sensor['identifier'],
                sensor['x_origin'],
                sensor['y_origin'],
                sensor['z_origin'],
                sensor['y_rotation'],
                sensor['measurement_unit']
            )

        self.__batch_df = converter.run()

    def __cluster(self):
        pass

    def __predict(self):
        pass

    def __persist(self):
        # influx_db.write_api.write("ccs4dt", "ccs4dt", ["h2o_feet,location=coyote_creek water_level=1".encode()])
        pass

    def __update_status(self, new_status):
        self.__input_batch_service.update_status(self.__input_batch_id, new_status)
