import numpy as np
from math import sin, cos, radians

from ccs4dt.main.shared.enums.measurement_unit import MeasurementUnit


class Converter:
    """Converter class is responsible for converting measurements from different sensors into a common format.
    Sensors collect measurements with respect to their coordinate system. The conversion ensures that measurements of
    different sensors are harmonized such that they can be compared and aggregated. The conversion includes:
    1. Measurement Unit transformation (e.g. cm, mm, m)
    2. Rotation transformation (rotate axis of sensors)
    3. Coordinate offset transformation (depending on the location where the sensor is deployed within a location)

    :param batch: The input measurement batch
    :type batch: pandas.DataFrame
    """

    def __init__(self, batch):
        self.__batch_df = batch
        self.__sensors = {}

    def add_sensor(self, sensor_identifier, x_origin, y_origin, z_origin, yaw, pitch, roll, measurement_unit):
        """
        Add a sensor configuration

        :param sensor_identifier: unique id of the sensor
        :type sensor_identifier: str
        :param x_origin: x coordinate where the sensor is installed in the space
        :type x_origin: float
        :param y_origin: y coordinate where the sensor is installed in the space
        :type y_origin: float
        :param z_origin: z coordinate where the sensor is installed in the space
        :type z_origin: float
        :param yaw: angle (deg) counterclockwise rotation of the sensor xy-plane in relation to the location xy-plane
        :type yaw: float
        :param pitch: angle (deg) counterclockwise rotation of the sensor yz-plane in relation to the location yz-plane
        :type pitch: float
        :param roll: angle (deg) counterclockwise rotation of the sensor xz-plane in relation to the location xz-plane
        :type roll: float
        :param measurement_unit: the unit of the measurement (cm, mm, m, ...)
        :type measurement_unit: str
        """
        self.__sensors[sensor_identifier] = {
            'x_origin': x_origin,
            'y_origin': y_origin,
            'z_origin': z_origin,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'measurement_unit': measurement_unit
        }

        return self

    def run(self):
        """
        Run the converter for the provided input batch and the configured sensors
        """
        if not self.__sensors:
            return self.__batch_df

        self.__batch_df = self.__batch_df.apply(self.__convert_units, axis=1)
        self.__batch_df = self.__batch_df.apply(self.__convert_axis_rotation, axis=1)
        self.__batch_df = self.__batch_df.apply(self.__convert_coordinate_offset, axis=1)
        return self.__batch_df

    def __convert_units(self, row):
        """
        Handle conversion of measurement units.
        :param row: pd.Series
        :return: pd.Series
        """
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

    def __convert_coordinate_offset(self, row):
        """
        Handle the coordinate offset between the sensor's coordinate system and the common frame of reference.
        :param row: pd.Series
        :return: pd.Series
        """
        sensor = self.__sensors[row['sensor_identifier']]

        for axis in ['x', 'y', 'z']:
            row[axis] += sensor[f'{axis}_origin']

        return row

    def __convert_axis_rotation(self, row):
        """
        Handle axis rotation in 3D space. The sensor's coordinate system and the frame of reference
        must have the same orientation. Thus, we need to perform an axis rotation if they are not aligned.
        Read more: https://en.wikipedia.org/wiki/Rotation_matrix --> General Rotations
        Graphical illustration: https://drive.google.com/file/d/1D9SjnO0xFJpuGy1T1oRNXpANSXOs9jRS/view
        :param row: pd.Series
        :return: pd.Series
        """
        sensor = self.__sensors[row['sensor_identifier']]

        # Yaw: counterclockwise rotation of the sensor xy-plane in relation to the location xy-plane
        yaw_angle = radians(sensor['yaw'])

        # Pitch: counterclockwise rotatin of the sensor yz-plane in relation to the location yz-plane
        pitch_angle = radians(sensor['pitch'])

        # Roll: counterlockwise rotation of the sensor xz-plane in relation to the location xz-plane
        roll_angle = radians(sensor['roll'])

        # Yaw rotation matrix
        yaw_rotation_matrix = np.array([
            [cos(yaw_angle), -sin(yaw_angle), 0],
            [sin(yaw_angle), cos(yaw_angle), 0],
            [0, 0, 1],
        ])

        # Pitch rotation matrix
        pitch_rotation_matrix = np.array([
            [cos(pitch_angle), 0, sin(pitch_angle)],
            [0, 1, 0],
            [-sin(pitch_angle), 0, cos(pitch_angle)],
        ])

        # Roll rotation matrix
        roll_rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos(roll_angle), -sin(roll_angle)],
            [0, sin(roll_angle), cos(roll_angle)],
        ])

        # Final rotation matrix is the product of yaw, pitch, and roll rotation matrices
        final_rotation_matrix = np.matmul(np.matmul(yaw_rotation_matrix, pitch_rotation_matrix), roll_rotation_matrix)

        # Now we apply the rotation matrix to the sensor measurement (input_vector).
        input_vector = np.array([row['x'], row['y'], row['z']])
        output_vector = np.matmul(final_rotation_matrix, input_vector)

        # Save transformed values to row.
        for i, axis in enumerate(['x', 'y', 'z']):
            row[axis] = output_vector[i]

        return row
