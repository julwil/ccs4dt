import pandas as pd
import datetime
import uuid
import random as rand
import matplotlib.pyplot as plt
import numpy as np
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation




class CoordinateSystem(object):
    """This class represents a coordinate system/orientation of an object in the real world. It is seen relative to a arbitrary 
    frame of reference, which could be e.g. a location that is analyzed.


    :param origin_with_respect_to_ref_sys_x: x-translation parameter of coordinate system in relation to frame of refrence
    :type origin_with_respect_to_ref_sys_x: numeric
    :param origin_with_respect_to_ref_sys_y: y-translation parameter of coordinate system in relation to frame of refrence
    :type origin_with_respect_to_ref_sys_y: numeric
    :param origin_with_respect_to_ref_sys_z: z-translation parameter of coordinate system in relation to frame of refrence
    :type origin_with_respect_to_ref_sys_z: numeric
    :param yaw_xy_with_respect_to_ref_sys: xy-rotational parameter (counterclockwise) of coordinate system in relation to frame of refrence, see https://bit.ly/3AZM5iP for graphical representation
    :type yaw_xy_with_respect_to_ref_sys: numeric
    :param pitch_yz_with_respect_to_ref_sys: yz-rotational parameter (counterclockwise) of coordinate system in relation to frame of refrence, see https://bit.ly/3AZM5iP for graphical representation
    :type pitch_yz_with_respect_to_ref_sys: numeric
    :param roll_xz_with_respect_to_ref_sys: xz-rotational parameter (counterclockwise) of coordinate system in relation to frame of refrence, see https://bit.ly/3AZM5iP for graphical representation
    :type roll_xz_with_respect_to_ref_sys: numeric

    :return: Returns string of coordinate system parameters on successfull creation
    :rtype: string
    """

    def __init__(self, origin_with_respect_to_ref_sys_x, origin_with_respect_to_ref_sys_y, origin_with_respect_to_ref_sys_z, yaw_xy_with_respect_to_ref_sys, pitch_yz_with_respect_to_ref_sys, roll_xz_with_respect_to_ref_sys):
        
        # Define translational parameters with repsect to global frame of reference
        self.origin_with_respect_to_ref_sys_x =  origin_with_respect_to_ref_sys_x
        self.origin_with_respect_to_ref_sys_y =  origin_with_respect_to_ref_sys_y
        self.origin_with_respect_to_ref_sys_z =  origin_with_respect_to_ref_sys_z

        # Define rotational parameters with repsect to global frame of reference
        self.yaw_xy_with_respect_to_ref_sys =  yaw_xy_with_respect_to_ref_sys
        self.pitch_yz_with_respect_to_ref_sys =  pitch_yz_with_respect_to_ref_sys
        self.roll_xz_with_respect_to_ref_sys =  roll_xz_with_respect_to_ref_sys

        return("Creation successfull for: \n" + self.__str__)

    def __str__(self):
        """String representation of the coordinate system

        :return: Returns string of coordinate system parameters
        :rtype: string
        """
        return ('Coordinate system orientation: \n translation (x,y,z): (' + str(self.origin_with_respect_to_ref_sys_x) + ", " + str(self.origin_with_respect_to_ref_sys_y) + ", " 
                + str(self.origin_with_respect_to_ref_sys_z) + ') \n and rotation (xy,yz,xz): (' + str(self.yaw_xy_with_respect_to_ref_sys) + ", " + str(self.pitch_yz_with_respect_to_ref_sys) + ", " 
                + str(self.roll_xz_with_respect_to_ref_sys) + ')' )

    def get_translation_x(self):
        """Getter function for x-translation parameter of coordinate system in relation to frame of reference    

        :return: Returns x-translation parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.origin_with_respect_to_ref_sys_x)

    def get_translation_y(self):
        """Getter function for y-translation parameter of coordinate system in relation to frame of reference   

        :return: Returns y-translation parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.origin_with_respect_to_ref_sys_y)

    def get_translation_z(self):
        """Getter function for z-translation parameter of coordinate system in relation to frame of reference    

        :return: Returns z-translation parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.origin_with_respect_to_ref_sys_z)

    def get_yaw_xy(self):
        """Getter function for yaw (xy-rotational) parameter of coordinate system in relation to frame of reference     

        :return: Returns yaw (xy-rotational) parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.yaw_xy_with_respect_to_ref_sys)

    def get_pitch_yz(self):
        """Getter function for pitch (yz-rotational) parameter of coordinate system in relation to frame of reference    

        :return: Returns pitch (yz-rotational) parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.pitch_yz_with_respect_to_ref_sys)

    def get_roll_xz(self):
        """Getter function for roll (xz-rotational) parameter of coordinate system in relation to frame of reference   

        :return: Returns roll (xz-rotational)) parameter of coordinate system in relation to frame of reference
        :rtype: numeric
        """
        return(self.roll_xz_with_respect_to_ref_sys)

class Sensor(object):
    """This class represents a synthetic sensor that emulates the measurement of datapoints based on the (virtual) sensor's parameters and the true position of the object to measure
 

    :param sensor_type: Description of the sensor type, e.g. RFID, camera or NFC
    :type sensor_type: string
    :param coordinate_system: Coordinate system that describes the position and rotation of the sensor inside the frame of reference (e.g. of the location)
    :type coordinate_system: CoordinateSystem
    :param sensor_precision: Precision of the sensor, i.e. minimal spatial resolution of the sensor in the measurement unit defined in parameter "sensor_precision_measurement_unit"
    :type sensor_precision: numeric
    :param sensor_pollingrate: Pollingrate of the sensor, i.e. maximal temporal resolution of the sensor in the measurement unit defined in parameter "sensor_pollingrate_measurement_unit"
    :type sensor_pollingrate: numeric
    :param measurement_reach: Maximum measurement reach of the sensor, past this distance the sensor is not able to measure anything, measurement unit defined in parameter "measurement_reach_measurement_unit"
    :type measurement_reach: numeric
    :param sensor_precision_measurement_unit: Measurement unit of the sensor precision, allowed are all available SI prefixes for meters TODO: NOT YET CONSIDERED
    :type sensor_precision_measurement_unit: String
    :param sensor_pollingrate_measurement_unit: Measurement unit of the sensor polling rate, allowed are the SI unit "s" (including all prefixes) and the non-SI units "min" (minutes), "h"(hours) and "d"(days)  TODO: NOT YET CONSIDERED
    :type sensor_pollingrate_measurement_unit: String
    :param measurement_reach_measurement_unit: Measurement unit of the sensor polling rate allowed are all available SI prefixes for meters  TODO: NOT YET CONSIDERED
    :type measurement_reach_measurement_unit: String
    :param sensor_identifier: Unique identifier of the sensor, defaults to automatically generated uuid4
    :type sensor_identifier: string
    :param stability: Function of the stability function of the measurement of the sensor, i.e. how large the signal degradation is based on distance between object to be measured and the sensor, defaults to 0 TODO: NOT YET CONSIDERED
    :type stability: function
    

    :return: Returns string of sensor parameters on successfull creation
    :rtype: string
    """

    def __init__(self, sensor_type, coordinate_system, sensor_precision, sensor_pollingrate, measurement_reach, sensor_pollingrate_measurement_unit = "s", sensor_precision_measurement_unit='cm', measurement_reach_measurement_unit = "cm", sensor_identifier = uuid.uuid4(), stability = 0):
        # Type of the sensor
        self.sensor_type = sensor_type

        # Absolute position of the sensor inside the location (in cm)
        self.absolute_pos_x = coordinate_system.get_translation_x()
        self.absolute_pos_y = coordinate_system.get_translation_y()
        self.absolute_pos_z = coordinate_system.get_translation_z()

        # Determined by the geopraphical orientation in which the y-axis of the readers coordinate systems increases. 0-359 degrees where East=0, North=90, West=180 South=270
        self.orientation_x = coordinate_system.get_yaw_xy()
        self.orientation_y = coordinate_system.get_pitch_yz()
        self.orientation_z = coordinate_system.get_roll_xz()


        # Precision and unit of precision of the sensor
        self.sensor_precision = sensor_precision
        self.sensor_precision_measurement_unit = sensor_precision_measurement_unit

        # Maximum measurement distance from the position of the sensor and unit
        self.measurement_reach = measurement_reach
        self.measurement_reach_measurement_unit = measurement_reach_measurement_unit

        # Pollingrate, how frequent the sensor will be able to measure & unit
        self.sensor_pollingrate = sensor_pollingrate
        self.sensor_pollingrate_measurement_unit = sensor_pollingrate_measurement_unit
        
        # Must be unique, identifier of the sensor
        self.sensor_identifier = sensor_identifier

        # Stability (with what percentage the sensor randomly drops a measurement)
        self.stability = stability

        return("Sensor successfully created! \n" + self.__str__)

    def __str__(self):
        """String representation of the senor

        :return: Returns string of coordinate system parameters
        :rtype: string
        """
        return (str('Sensor of type "'+ self.sensor_type + '" with id: ' + str(self.sensor_identifier) + 
                    '\nat absolute position: (' + str(self.absolute_pos_x) + ', ' + str(self.absolute_pos_y) + ', ' + str(self.absolute_pos_z) + ') ' + ' (x, y, z), with orientation (x,y,z) (' +
                      str(self.orientation_x) + '°, ' +  str(self.orientation_y) + '°, ' +  str(self.orientation_z) + '°), ' + 
                    '\nand precision: ' + str(self.sensor_precision) + ' ' + str(self.sensor_precision_measurement_unit) + ', ' + 
                    '\nand pollingrate: ' + str(self.sensor_pollingrate) + ' ' + str(self.sensor_pollingrate_measurement_unit) + 
                    '\nand the sensor drops measurements with a probability of ' + str(self.stability) + '%' ))

    def get_sensor_position(self):
        """Getter function for positional parameters (x,y,z) of sensors in the frame of reference      

        :return: Returns x,y,z-coordinate of the sensor in the frame of reference
        :rtype: numeric
        """
        return (self.absolute_pos_x, self.absolute_pos_y, self.absolute_pos_z)

    def get_sensor_id(self):
        """Getter function for the sensor id       

        :return: Returns x,y,z-coordinate of the sensor in the frame of reference
        :rtype: string
        """
        return (self.sensor_identifier)

    def get_sensor_precision(self):
        """Getter function for the sensor precision       

        :return: Returns sensor precision
        :rtype: numeric
        """
        return (self.sensor_precision)

    # TODO: Implement getter sensor precision measurement unit

    def get_sensor_type(self):
        """Getter function for the sensor type       

        :return: Returns sensor type
        :rtype: string
        """
        return (self.sensor_type)

    # TODO: Implement getter sensor stability
    # TODO: Implement getter sensor pollingrate
    # TODO: Implement getter sensor pollingrate measurement unit

    def get_sensor_reach(self):
        """Getter function for the sensor reach       

        :return: Returns sensor reach
        :rtype: numeric
        """
        return (self.measurement_reach)

    # TODO: Implement getter sensor reach measurement unit

    # Function simulates precision loss for measurement relative to true position reference frame
    def generate_random_point_in_sphere(self, point_x, point_y, point_z):
        """Simualtes measurement of a sensor. Given a random true position as x,y,z coordinates the function creates a random point around the true position given based on the precision of the sensor.
        This is done via rejection sampling (https://en.wikipedia.org/wiki/Rejection_sampling) to generate a uniform distribution.

        :param point_x: x-Coordinate of the true position (as relative coordinate) of object for which a measurement should be simulated
        :type point_x: numeric
        :param point_y: y-Coordinate of the true position (as relative coordinate) of object for which a measurement should be simulated
        :type point_y: numeric
        :param point_z: z-Coordinate of the true position (as relative coordinate) of object for which a measurement should be simulated
        :type point_z: numeric

        :return: Returns positional parameters (x,y,z) for synthetically generated measurement point. Returned parameters are in coordinate system of sensor for which the measurement was simulated.
        :rtype: 3-tuple (numeric,numeric,numeric)
        """
        
        def randomize_positions(self):
            # Assume cube and randomize all three directions based on precision
            randomized_x = rand.randint(-self.sensor_precision, self.sensor_precision) 
            randomized_y = rand.randint(-self.sensor_precision, self.sensor_precision) 
            randomized_z = rand.randint(-self.sensor_precision, self.sensor_precision) 

            return (randomized_x, randomized_y, randomized_z)

        # Initial simulation
        # Random point generation
        (random_pos_x, random_pos_y, random_pos_z) = randomize_positions(self)
        # Rejection criterium
        random_point_distance_to_sphere_origin = (random_pos_x*random_pos_x + random_pos_y*random_pos_y + random_pos_z*random_pos_z)**0.5

        # Rejection sampling
        while(random_point_distance_to_sphere_origin > self.sensor_precision):
            # Random point generation
            (random_pos_x, random_pos_y, random_pos_z) = randomize_positions(self)
            # Rejection criterium
            random_point_distance_to_sphere_origin = (random_pos_x*random_pos_x + random_pos_y*random_pos_y + random_pos_z*random_pos_z)**0.5

        return (random_pos_x + point_x, random_pos_y + point_y, random_pos_z + point_z)

# Transforms point coordinates in it's own coordinate system into frame of reference (f.o.r.) coordinate system
def transform_cartesian_coordinate_system(point_x, point_y, point_z, coordinate_system, inverse_transformation = False, output_transformation_matrix = False):
    """Transforms positional coordinates of a point in a specific coordinate system into its frame of reference

        :param point_x: x-Coordinate of the true position (as relative coordinate) of object for which the coordinates should be transformed into the frame of reference
        :type point_x: numeric
        :param point_y: y-Coordinate of the true position (as relative coordinate) of object for which a measurement should be transformed into the frame of reference
        :type point_y: numeric
        :param point_z: z-Coordinate of the true position (as relative coordinate) of object for which a measurement should be transformed into the frame of reference
        :type point_z: numeric
        :param coordinate_system: Coordinate system of the object (point_x,y,z) for which the coordinate transformation should be performed
        :type coordinate_system: CoordinateSystem
        :param inverse_transformation: Parameter that specifies wheter the coordinate transformation should be performed from "Point coordinate system" -> "Frame of reference" (if == False) or from "Frame of reference" -> "Point coordinate system" (if == True), default is False
        :type inverse_transformation: boolean
        :param output_transformation_matrix: Specifies if output should contain the output transformation matrix (== True) or not (== False), default is False
        :type output_transformation_matrix: boolean

        :return: Return dependent on output_transformation_matrix parameter. If set to false function outputs transformed coordinates of point. If set to true, function outputs the transformation matrix and the transformed points coordinates.
        :rtype: point coordinate tuple (x,y,z) or transformation_matrix and point coordinate tuple (x,y,z)
        """


    # Define transformation matrices (point -> f.o.r. & f.o.r. -> point)
    ## Init diagonal base matrix to fill with rotation and translation parameters
    point_coordinate_system_to_global_frame_of_reference = np.eye(4)
    
    ## Define rotational parameters to go from point coordinate system to global frame of reference (basically answer the question: "how is the coord system aliogned that is to be transformed")
    R = Rotation.from_euler("XYZ",[coordinate_system.yaw_xy_with_respect_to_ref_sys, coordinate_system.pitch_yz_with_respect_to_ref_sys, coordinate_system.roll_xz_with_respect_to_ref_sys], degrees = True).as_matrix()

    ## Apply rotational parameters to transformation matrix
    point_coordinate_system_to_global_frame_of_reference[:3,:3] = R

    ## Apply translational parameters to transformation matrix (basically answer the question: "where is origin of coord system that is to be transformed")
    point_coordinate_system_to_global_frame_of_reference[:3,3] = np.array([coordinate_system.origin_with_respect_to_ref_sys_x, coordinate_system.origin_with_respect_to_ref_sys_y, coordinate_system.origin_with_respect_to_ref_sys_z])

    # Decide wheter we want to transform from point_coord_sys -> f.o.r. (False) or from f.o.r. -> point_coord_sys
    if inverse_transformation == True:
        ## Invert matrix to have transformation matrix for back conversion
        global_frame_of_reference_to_point_coordinate_system = np.linalg.inv(point_coordinate_system_to_global_frame_of_reference)
        transformation_matrix = global_frame_of_reference_to_point_coordinate_system

    else:
        transformation_matrix = point_coordinate_system_to_global_frame_of_reference


    ## Definition of point and transformation into other reference frame: def as 4D array (that will be transposed) in the format [x,y,z,1] (the one allows the multiplication to work, 0 if already transformed into correct system)
    point_A_in_frame_of_reference = transformation_matrix@np.array([point_x, point_y, point_z, 1]).reshape(4,1)

    transformed_x = point_A_in_frame_of_reference[0][0]
    transformed_y = point_A_in_frame_of_reference[1][0]
    transformed_z = point_A_in_frame_of_reference[2][0]

    # Define wheter or not output should contain transformation matrix or only transformed point coordinates
    if output_transformation_matrix == True:
      return (transformation_matrix,(transformed_x, transformed_y, transformed_z))

    else:
        return transformed_x, transformed_y, transformed_z

# Plots randomized points and sphere based on sensor input and number of measurements
def plot_randomized_sphere(sensor, randomization_steps):
    """Plots (in 3D) x randomized measurements and sphere of precision based on sensor parameters around origin

    :param sensor: Sensor for which the measurements should be simulated
    :type sensor: Sensor
    :param randomization_steps: Amount of measurements that should be simulated
    :type randomization_steps: integer


    :return: Returns nothing
    :rtype: None
    """

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')

    ax.grid()

    i = 1
    x_array = []
    y_array = []
    z_array = []
    while(i <= randomization_steps):
        (x,y,z) = sensor.generate_random_point_in_sphere(0,0,0)
        x_array.append(x)
        y_array.append(y)
        z_array.append(z)

        i += 1

    # draw points
    ax.scatter(x_array, y_array, z_array)

    # draw origin
    ax.scatter(0, 0, 0, s = 200)

    # draw wireframe sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    # Scale sphere with (sensor.get_precision()) (half of diameter)
    a = np.cos(u)*np.sin(v)*(sensor.get_sensor_precision())
    b = np.sin(u)*np.sin(v)*(sensor.get_sensor_precision())
    c = np.cos(v)*(sensor.get_sensor_precision())

    ax.plot_wireframe(a, b, c, color="grey", alpha = 0.2)

    plt.show()

    return None

# Plot two coordinate systems (frame of reference and point coordinate system that contain the same point transformed)
def plot_point_in_two_coordinate_systems(point_x, point_y, point_z, point_coord_sys, plot_system_indicators = True):
    """Plot two coordinate systems (frame of reference and point coordinate system that contain the same point transformed).
    Upper plot is in frame of reference, Lower plot in point coordinate system

    :param point_x: x-coordinate of point in point coordinate system
    :type point_x: numeric
    :param point_y: y-coordinate of point in point coordinate system
    :type point_y: numeric
    :param point_z: z-coordinate of point in point coordinate system
    :type point_z: numeric
    :param point_coord_sys: Coordinate system of the point to be plotted
    :type point_coord_sys: CoordinateSystem
    :param plot_system_indicators: Speficies wheter or not system indicators (x,y,z and identity vector for coordinate systems incl. labels) should be plotted or not, default is True
    :type plot_system_indicators: boolean

    :return: Returns nothing
    :rtype: None
    """

    transformation_matrix, transformed_point = transform_cartesian_coordinate_system(point_x, point_y, point_z, point_coord_sys, output_transformation_matrix = True)

    tm = TransformManager()
    tm.add_transform("f.o.r.", "P", transformation_matrix)

    plt.figure(figsize=(15, 15))

    ax = make_3d_axis(10, 211)
    if plot_system_indicators == True:
        ax = tm.plot_frames_in("f.o.r.", ax = ax, s = 3)
    ax.plot(*transformed_point[:3],"yo")
    ax.view_init(20, 20)

    ax = make_3d_axis(10, 212)
    if plot_system_indicators == True:
        ax = tm.plot_frames_in("P", ax = ax, s = 3)
    ax.plot(point_x, point_y, point_z,"yo")
    ax.view_init(20, 20)


    plt.show()

    return None

# Import the true position dataset
def import_occupancy_presence_dataset (filepath, import_rows_count, drop_irrelevant_columns = True, transform_to_3D_data = True, starting_date = '01.06.2019', date_format = '%d.%m.%Y'):
    """Imports the true position dataset and 

    :param filepath: filepath of trueposition dataset
    :type filepath: raw string literal
    :param import_rows_count: Max. number of rows that should be imported from dataset
    :type import_rows_count: integer
    :param drop_irrelevant_columns: Specifies if (for this analysis) irrelevant columns should be dropped, default is True
    :type drop_irrelevant_columns: boolean
    :param transform_to_3D_data: Specifies if data should either consider the true position coordinate as 0 (if == False) or set the z-coordinate of the true positions as 'height' (if == True), default is True
    :type transform_to_3D_data: boolean
    :param starting_date: As dataset only provides date ids, they need to be transformed into actual dates. starting_date specifies actual date for day_id == 0. Uses date_format for date parsing, default is '01.06.2019'
    :type starting_date: string
    :param date_format: Specifies format of starting_date, default is '%d.%m.%Y'
    :type date_format: string

    :raises ValueError: ValueError risen if not expected columns are present

    :return: Returns the imported file as dataframe 
    :rtype: dataframe
    """


    if drop_irrelevant_columns == True:
        relevant_columns = ['time', 'day_id', 'x', 'y', 'occupant_id', 'camera_id', 'height']
        import_file = pd.read_csv(filepath, nrows = import_rows_count, header = 0,  usecols = relevant_columns)

        # If data is only in 2D we add a 3D coordinate that is empty
        if transform_to_3D_data == True:
            import_file['z'] = 0
        else: 
            import_file['z'] = import_file['height']

        # Convert time column to datetime format
        #import_file['time'] = pd.to_datetime(import_file['time'])

        # Take day id and transform into datetime
        import_file['date'] = datetime.datetime.strptime(starting_date, date_format) + pd.to_timedelta(np.ceil(import_file['day_id']), unit="D")


        # Take time and transform into datetime
        date_time_format = '%d.%m.%Y %H:%M:%S.%f'
        time_format = '%H:%M:%S.%f'

        #import_file['date_time'] = pd.to_datetime(import_file['date'].apply(str)+' '+import_file['time'])
        
      
        #import_file['date_time'] = pd.to_datetime(str(import_file['date']) + ' ' + import_file['time'])
    else:
        raise ValueError('Case not covered (include irrelevant columns)')

    return import_file


def simulate_measure_data_from_true_positions(true_position_dataframe, sensor):
    
    # Generate empty data frame for measurements
    measurement_dataframe = pd.DataFrame()

    # Add time and date column to measured data frame
    # TODO: Should we combine these?
    measurement_dataframe['date'] = [x for x in true_position_dataframe['date']]
    measurement_dataframe['time'] = [x for x in true_position_dataframe['time']]


    # Add sensor id, sensor type and occupant id
    measurement_dataframe['occupant_id'] = true_position_dataframe['occupant_id']
    measurement_dataframe['sensor_type'] = sensor.get_sensor_type()
    measurement_dataframe['sensor_id'] = sensor.get_sensor_id()

    # Add original coords (can be dropped later, only for verification purposes)
    measurement_dataframe['x_original'] = true_position_dataframe['x']
    measurement_dataframe['y_original'] = true_position_dataframe['y']
    measurement_dataframe['z_original'] = true_position_dataframe['z']
 
    # Generate measurements (with random precision) in coordinate system of sensor
    measurement_dataframe['xyz_measured'] = ([(sensor.generate_random_point_in_sphere(x, y, z)) for x, y, z in zip(measurement_dataframe['x_original'], measurement_dataframe['y_original'], measurement_dataframe['z_original'])])
    # Unpack those measure coordinates
    measurement_dataframe[['x_measured_abs_pos', 'y_measured_abs_pos','z_measured_abs_pos']] = pd.DataFrame(measurement_dataframe['xyz_measured'].tolist(), index=measurement_dataframe.index)

    # Transform measured coordinates into absolute coordinate system (frame of reference)
    measurement_dataframe['x_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[0]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])
    measurement_dataframe['y_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[1]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])
    measurement_dataframe['z_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[2]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])

    # TODO: Add measurement boundary, polling rate and decaying stability (as function of measurement distance)
    # PSEUDOCODE HERE
    # calculate distance(xyz_measured, sensor_position)
    # # Drop all rows that are out of reach for sensor
    # for all rows where distance_to_sensor > sensor.get_reach():
    #   drop(row)
    # # Drop randomized rows as decaying function of distance to sensor, model function so that at sensor.get_reach()+1 the likelihood of measurement reaches 0
    # for all rows:
    #   calculate decay likelihood as function of distance
    #   if random(0,1) > decay likelihood:
    #       drop(row)
    # # Drop all rows where sensor pollingrate is not quick enough:
    # for all rows where (timediff(row[n+1]-row[n]) < sensor.get_pollingrate():
    #   drop(row)


    print(measurement_dataframe)

    return measurement_dataframe


def function_wrapper_data_ingestion(path, import_rows, test_coord_parameters, test_sensor_parameters):



    imported_dataset = import_occupancy_presence_dataset(path, import_rows_count = import_rows)

    (x,y,z,yaw_xy,pitch_yz,roll_xz) = test_coord_parameters
    test_coord_sys = CoordinateSystem(x,y,z,yaw_xy,pitch_yz,roll_xz)

    (sensor_type, empty, precision, pollingrate, reach) =  test_sensor_parameters
    test_sensor = Sensor(sensor_type, test_coord_sys, precision, pollingrate, reach)

    simulate_measure_data_from_true_positions(imported_dataset, test_sensor)

    return None

function_wrapper_data_ingestion(r'scripts\synteticDataGeneration\assets\sampledata\occupancy_presence_and_trajectories.csv', 5, (3,1,0, 30,-15,45), ('RFID',None,30,10,500))




def function_wrapper_plotting_examples(plot_type):

    def plot_examples(sensor, coord_sys, point_x, point_y, point_z, repeated_steps):
        #print(sensor)
        #print(coord_sys)

        #print(sensor.generate_random_point_in_sphere())
        plot_randomized_sphere(sensor, repeated_steps)

        #print(transform_cartesian_coordinate_system(1,5,-1, coord_sys))
        plot_point_in_two_coordinate_systems(point_x, point_y, point_z, coord_sys, plot_system_indicators = True)

    test_coord_sys = CoordinateSystem(3,1,-3, 90,135,42)
    test_sensor = Sensor('RFID', test_coord_sys, 30, 10, 500)

    plot_examples(test_sensor, test_coord_sys, 1, 5, -1, 3000)

    return None