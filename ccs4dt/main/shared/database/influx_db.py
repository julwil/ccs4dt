import os

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient


class InfluxDB:
    def __init__(self):
        load_dotenv()
        url = os.environ.get('DOCKER_INFLUX_DB_URL', 'influxdb')
        token = os.environ.get('DOCKER_INFLUXDB_INIT_ADMIN_TOKEN')
        org = os.environ.get('DOCKER_INFLUXDB_INIT_ORG', 'ccs4dt')
        self.db = self.__connect(url, token, org)
        self.write_api = self.db.write_api()
        self.query_api = self.db.query_api()

    def __connect(self, url, token, org):
        return InfluxDBClient(url=url, token=token, org=org)
