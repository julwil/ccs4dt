import os
import sqlite3

from dotenv import load_dotenv


class CoreDB:
    def __init__(self):
        load_dotenv()
        db_path = os.environ.get('CORE_DB_PATH', 'storage/core_db/sqlite.db')
        self.__db = self.__connect(db_path)
        self.__enable_foreign_keys()
        self.__create_schema()
        self.__db.row_factory = sqlite3.Row

    def connection(self):
        return self.__db

    def __connect(self, db_path):
        return sqlite3.connect(db_path, check_same_thread=False) # Enable concurrent reads/writes

    def __enable_foreign_keys(self):
        self.__db.executescript('PRAGMA foreign_keys=1;')

    def __create_schema(self):
        self.connection().cursor().executescript('''

            CREATE TABLE IF NOT EXISTS locations
            (
                id                  INTEGER
                    PRIMARY KEY AUTOINCREMENT,
                name                TEXT NOT NULL,
                external_identifier TEXT
            );
            
            CREATE TABLE IF NOT EXISTS input_batches
            (
                id              INTEGER
                    PRIMARY KEY AUTOINCREMENT,
                location_id     INTEGER NOT NULL
                    constraint input_batches_locations_id_fk
                        references locations,
                status          TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS sensors
            (
                id               INTEGER
                    PRIMARY KEY AUTOINCREMENT,
                location_id      INTEGER NOT NULL
                    constraint sensors_locations_id_fk
                        references locations,
                identifier       TEXT    NOT NULL,
                type             TEXT    NOT NULL,
                x_origin         REAL NOT NULL,
                y_origin         REAL NOT NULL,
                z_origin         REAL NOT NULL,
                orientation      REAL NOT NULL,
                measurement_unit TEXT    NOT NULL
            );
        ''')
