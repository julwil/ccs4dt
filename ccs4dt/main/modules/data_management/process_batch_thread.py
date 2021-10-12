import threading


class ProcessBatchThread(threading.Thread):

    def __init__(self, input_batch_service, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.__input_batch_service = input_batch_service
        self.args = args
        self.kwargs = kwargs
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        self.__input_batch_service.save_batch_to_influx(self.args)
        # Coordinate Transformation
        # Clustering
        # Prediction
        # Update output batch

        # influx_db.write_api.write("ccs4dt", "ccs4dt", ["h2o_feet,location=coyote_creek water_level=1".encode()])
