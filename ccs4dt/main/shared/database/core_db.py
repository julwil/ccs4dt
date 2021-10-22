import os
import sqlite3

from dotenv import load_dotenv


class CoreDB:
    """CoreDB stores all location data and its associated sensor configuration"""
    def __init__(self):
        load_dotenv()
        self.__db_path = os.environ.get('CORE_DB_PATH', 'storage/core_db/sqlite.db')

    def connection(self):
        """
        Returns a new database connection
        :rtype CoreDB
        """
        connection = self.__connect(self.__db_path)
        self.__enable_foreign_keys(connection)
        self.__create_schema(connection)
        connection.row_factory = sqlite3.Row
        return connection

    def __connect(self, db_path):
        return sqlite3.connect(db_path)

    def __enable_foreign_keys(self, connection):
        connection.executescript('PRAGMA foreign_keys=1;')

    def __create_schema(self, connection):
        connection.cursor().executescript('''

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
                yaw              REAL NOT NULL,
                pitch            REAL NOT NULL,
                roll             REAL NOT NULL,
                measurement_unit TEXT    NOT NULL
            );
        ''')
