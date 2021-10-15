import logging
from math import sin, cos, radians

class Converter:
    def __init__(self, batch):
        self.__batch = batch
        self.__sensors = {}

    def add_sensor(self, sensor_identifier, x_origin, y_origin, z_origin, orientation, measurement_unit):
        self.__sensors[sensor_identifier] = {
            'x_origin': x_origin,
            'y_origin': y_origin,
            'z_origin': z_origin,
            'orientation': orientation,
            'measurement_unit': measurement_unit
        }

        return self

    def convert(self):
        if not self.__sensors:
            return self.__batch

        self.__batch = self.__batch.apply(self.__convert_units, axis=1)
        self.__batch = self.__batch.apply(self.__convert_coordinates, axis=1)
        return self.__batch


    def __convert_units(self, row):
        sensor = self.__sensors[row['sensor_identifier']]
        factor = 1

        # Our internal unit
        if sensor['measurement_unit'] == 'cm':
            return row

        if sensor['measurement_unit'] == 'mm':
            factor = 0.1

        if sensor['measurement_unit'] == 'm':
            factor = 100

        row['x'], row['y'], row['z'] = row['x'] * factor, row['y'] * factor, row['z'] * factor
        return row


    def __convert_coordinates(self, row):
        sensor = self.__sensors[row['sensor_identifier']]
        angle = radians(90 - sensor['orientation'])
        x_old, y_old = row['x'], row['y']

        logging.error(sensor['orientation'])

        # See Transformation in euclidean space for more info:
        # https://en.wikipedia.org/wiki/Rotation_matrix
        x_new = (cos(angle) * x_old + sin(angle) * y_old) + sensor['x_origin']
        y_new = (-sin(angle) * x_old + cos(angle) * y_old) + sensor['y_origin']

        row['x'], row['y'] = x_new, y_new

        return row




