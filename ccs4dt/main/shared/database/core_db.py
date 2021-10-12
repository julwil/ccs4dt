import os

from dotenv import load_dotenv
from tinydb import TinyDB


class CoreDB:
    def __init__(self):
        load_dotenv()
        db_path = os.environ.get('CORE_DB_PATH', 'storage/core_db/core_db.json')
        self.__db = self.__connect(db_path)
        self.location_table = self.__db.table('locations')
        self.input_batch_table = self.__db.table('input_batches')
        self.output_batch_table = self.__db.table('output_batches')

    def __connect(self, db_path):
        return TinyDB(db_path)
