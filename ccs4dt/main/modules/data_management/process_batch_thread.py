import threading


class ProcessBatchThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        self.args = args
        self.kwargs = kwargs
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    def run(self):
        pass
        # Coordinate Transformation
        # Clustering
        # Prediction
        # Update output batch

        # influx_db.write_api.write("ccs4dt", "ccs4dt", ["h2o_feet,location=coyote_creek water_level=1".encode()])
