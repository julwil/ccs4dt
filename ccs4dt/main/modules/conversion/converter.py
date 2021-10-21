from math import sin, cos, radians
from ccs4dt.main.shared.enums.measurement_unit import MeasurementUnit

class Converter:
    def __init__(self, batch):
        self.__batch_df = batch
        self.__sensors = {}
        self.__location_y_rotation = 0

    def add_sensor(self, sensor_identifier, x_origin, y_origin, z_origin, y_rotation, measurement_unit):
        self.__sensors[sensor_identifier] = {
            'x_origin': x_origin,
            'y_origin': y_origin,
            'z_origin': z_origin,
            'y_rotation': y_rotation,
            'measurement_unit': measurement_unit
        }

        return self

    def run(self):
        if not self.__sensors:
            return self.__batch_df

        self.__batch_df = self.__batch_df.apply(self.__convert_units, axis=1)
        self.__batch_df = self.__batch_df.apply(self.__convert_coordinates, axis=1)
        return self.__batch_df


    def __convert_units(self, row):
        sensor = self.__sensors[row['sensor_identifier']]
        factor = 1

        if sensor['measurement_unit'] == MeasurementUnit.CENTIMETER:
            return row

        if sensor['measurement_unit'] == MeasurementUnit.MILLIMETER:
            factor = 0.1

        if sensor['measurement_unit'] == MeasurementUnit.METER:
            factor = 100

        for axis in ['x', 'y', 'z']:
            row[axis] *= factor

        return row


    def __convert_coordinates(self, row):
        sensor = self.__sensors[row['sensor_identifier']]
        x_old, y_old = row['x'], row['y'] # z is assumed to be constant for all sensors

        # First we handle the rotation transformation in euclidean space.
        # The orientation of the sensor's and the location's coordinate system need to be the same.
        # More info: https://en.wikipedia.org/wiki/Rotation_matrix.
        rotation_angle = radians(self.__location_y_rotation - sensor['y_rotation'])
        x_rotated = cos(rotation_angle) * x_old + sin(rotation_angle) * y_old
        y_rotated = -sin(rotation_angle) * x_old + cos(rotation_angle) * y_old

        # Second we handle the coordinate offset between the location's coordinate system
        # and the sensor's coordinate system.
        x_new = x_rotated + sensor['x_origin']
        y_new = y_rotated + sensor['y_origin']

        row['x'], row['y'] = x_new, y_new
        return row




