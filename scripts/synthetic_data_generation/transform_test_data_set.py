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

    def __init__(self, origin_with_respect_to_ref_sys_x, origin_with_respect_to_ref_sys_y, origin_with_respect_to_ref_sys_z, rotation_with_respect_to_ref_sys_x, rotation_with_respect_to_ref_sys_y, rotation_with_respect_to_ref_sys_z):
        
        # Define translational parameters with repsect to global frame of reference
        self.origin_with_respect_to_ref_sys_x =  origin_with_respect_to_ref_sys_x
        self.origin_with_respect_to_ref_sys_y =  origin_with_respect_to_ref_sys_y
        self.origin_with_respect_to_ref_sys_z =  origin_with_respect_to_ref_sys_z

        # Define rotational parameters with repsect to global frame of reference
        self.rotation_with_respect_to_ref_sys_x =  rotation_with_respect_to_ref_sys_x
        self.rotation_with_respect_to_ref_sys_y =  rotation_with_respect_to_ref_sys_y
        self.rotation_with_respect_to_ref_sys_z =  rotation_with_respect_to_ref_sys_z

    def __str__(self):
        return ('Coordinate system orientation: \n translation (x,y,z): (' + str(self.origin_with_respect_to_ref_sys_x) + ", " + str(self.origin_with_respect_to_ref_sys_y) + ", " 
                + str(self.origin_with_respect_to_ref_sys_z) + ') \n and rotation (x,y,z): (' + str(self.rotation_with_respect_to_ref_sys_x) + ", " + str(self.rotation_with_respect_to_ref_sys_y) + ", " 
                + str(self.rotation_with_respect_to_ref_sys_z) + ')' )

    def get_translation_x(self):
        return(self.origin_with_respect_to_ref_sys_x)

    def get_translation_y(self):
        return(self.origin_with_respect_to_ref_sys_y)

    def get_translation_z(self):
        return(self.origin_with_respect_to_ref_sys_z)

    def get_rotation_x(self):
        return(self.rotation_with_respect_to_ref_sys_x)

    def get_rotation_y(self):
        return(self.rotation_with_respect_to_ref_sys_y)

    def get_rotation_z(self):
        return(self.rotation_with_respect_to_ref_sys_z)

class Sensor(object):

    def __init__(self, sensor_type, coordinate_system, sensor_precision, sensor_pollingrate, measurement_reach ,sensor_pollingrate_measurement_unit = "seconds", sensor_precision_measurement_unit='centimeter', sensor_identifier = uuid.uuid4(), stability = 0):
        # Type of the sensor
        self.sensor_type = sensor_type

        # Absolute position of the sensor inside the location (in cm)
        self.absolute_pos_x = coordinate_system.get_translation_x()
        self.absolute_pos_y = coordinate_system.get_translation_y()
        self.absolute_pos_z = coordinate_system.get_translation_z()

        # Determined by the geopraphical orientation in which the y-axis of the readers coordinate systems increases. 0-359 degrees where East=0, North=90, West=180 South=270
        self.orientation_x = coordinate_system.get_rotation_x()
        self.orientation_y = coordinate_system.get_rotation_y()
        self.orientation_z = coordinate_system.get_rotation_z()


        # Precision and unit of precision of the sensor
        self.sensor_precision = sensor_precision
        self.sensor_precision_measurement_unit = sensor_precision_measurement_unit

        # Maximum measurement distance from the absolute position of the 
        self.measurement_reach = measurement_reach
        self.sensor_pollingrate_measurement_unit = sensor_pollingrate_measurement_unit

        # Pollingrate, how frequent the sensor will be able to measure & unit
        self.sensor_pollingrate = sensor_pollingrate
        self.sensor_pollingrate_measurement_unit = sensor_pollingrate_measurement_unit
        
        # Must be unique, identifier of the sensor
        self.sensor_identifier = sensor_identifier

        # Stability (with what percentage the sensor randomly drops a measurement)
        self.stability = stability

    def __str__(self):
        return (str('Sensor of type "'+ self.sensor_type + '" with id: ' + str(self.sensor_identifier) + 
                    '\nat absolute position: (' + str(self.absolute_pos_x) + ', ' + str(self.absolute_pos_y) + ', ' + str(self.absolute_pos_z) + ') ' + ' (x, y, z), with orientation (x,y,z) (' +
                      str(self.orientation_x) + '°, ' +  str(self.orientation_y) + '°, ' +  str(self.orientation_z) + '°), ' + 
                    '\nand precision: ' + str(self.sensor_precision) + ' ' + str(self.sensor_precision_measurement_unit) + ', ' + 
                    '\nand pollingrate: ' + str(self.sensor_pollingrate) + ' ' + str(self.sensor_pollingrate_measurement_unit) + 
                    '\nand the sensor drops measurements with a probability of ' + str(self.stability) + '%' ))

    def get_sensor_position(self):
        return (self.absolute_pos_x, self.absolute_pos_y, self.absolute_pos_z)

    def get_sensor_id(self):
        return (self.sensor_identifier)

    def get_sensor_precision(self):
        return (self.sensor_precision)

    def get_sensor_type(self):
        return (self.sensor_type)

    def get_sensor_reach(self):
        return (self.measurement_reach)

    def transform_absolute_coordinates_into_relative_frame_of_sensor(self):

            ##TODO
        return (None)

    # TODO: (Maybe) optimize and switch from rejection sampling to a more advanced trigonometric function model
    # Function simulates precision loss for measurement relative to true position reference frame (true position at (0,0,0))
    def generate_random_point_in_sphere(self, point_x, point_y, point_z):
        
        def randomize_positions(self):
            # Assume cube and randomize all three directions based on precision
            randomized_x = rand.randint(-self.sensor_precision, self.sensor_precision) 
            randomized_y = rand.randint(-self.sensor_precision, self.sensor_precision) 
            randomized_z = rand.randint(-self.sensor_precision, self.sensor_precision) 

            return (randomized_x, randomized_y, randomized_z)

        # Initial simulation
        (random_pos_x, random_pos_y, random_pos_z) = randomize_positions(self)
        random_point_distance_to_sphere_origin = (random_pos_x*random_pos_x + random_pos_y*random_pos_y + random_pos_z*random_pos_z)**0.5

        # Rejection sampling
        while(random_point_distance_to_sphere_origin > self.sensor_precision):
            (random_pos_x, random_pos_y, random_pos_z) = randomize_positions(self)
            random_point_distance_to_sphere_origin = (random_pos_x*random_pos_x + random_pos_y*random_pos_y + random_pos_z*random_pos_z)**0.5

        return (random_pos_x+point_x, random_pos_y+point_y, random_pos_z+point_z)

# Transforms point coordinates in it's own coordinate system into frame of reference (f.o.r.) coordinate system
def transform_cartesian_coordinate_system(point_x, point_y, point_z, coordinate_system, inverse_transformation = False, output_transformation_matrix = False):
    
    # Define transformation matrices (point -> f.o.r. & f.o.r. -> point)
    ## Init diagonal base matrix to fill with rotation and translation parameters
    point_coordinate_system_to_global_frame_of_reference = np.eye(4)
    
    ## Define rotational parameters to go from point coordinate system to global frame of reference (basically answer the question: "how is the coord system aliogned that is to be transformed")
    R = Rotation.from_euler("XYZ",[coordinate_system.rotation_with_respect_to_ref_sys_x, coordinate_system.rotation_with_respect_to_ref_sys_y, coordinate_system.rotation_with_respect_to_ref_sys_z], degrees = True).as_matrix()

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

# Plot two coordinate systems (frame of reference and point coordinate system that contain the same point transformed)
def plot_point_in_two_coordinate_systems(point_x, point_y, point_z, point_coord_sys, plot_system_indicators = True):

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
    measurement_dataframe['x_original'] = imported_dataset['x']
    measurement_dataframe['y_original'] = imported_dataset['y']
    measurement_dataframe['z_original'] = imported_dataset['z']
 
    # Generate measurements (with random precision) in coordinate system of sensor
    measurement_dataframe['xyz_measured'] = ([(sensor.generate_random_point_in_sphere(x, y, z)) for x, y, z in zip(measurement_dataframe['x_original'], measurement_dataframe['y_original'], measurement_dataframe['z_original'])])
    # Unpack those measure coordinates
    measurement_dataframe[['x_measured_rel_pos', 'y_measured_rel_pos','z_measured_rel_pos']] = pd.DataFrame(measurement_dataframe['xyz_measured'].tolist(), index=measurement_dataframe.index)

    # Transform measured coordinates into absolute coordinate system (frame of reference)
    measurement_dataframe['x_measured_abs_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[0]) for x, y, z in zip(measurement_dataframe['x_measured_rel_pos'], measurement_dataframe['y_measured_rel_pos'], measurement_dataframe['z_measured_rel_pos'])])
    measurement_dataframe['y_measured_abs_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[1]) for x, y, z in zip(measurement_dataframe['x_measured_rel_pos'], measurement_dataframe['y_measured_rel_pos'], measurement_dataframe['z_measured_rel_pos'])])
    measurement_dataframe['z_measured_abs_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, test_coord_sys)[2]) for x, y, z in zip(measurement_dataframe['x_measured_rel_pos'], measurement_dataframe['y_measured_rel_pos'], measurement_dataframe['z_measured_rel_pos'])])

    # TODO: Add measurement boundary, polling rate and decaying stability (as function of measurement distance)
    # PSEUDOCODE HERE
    # calculate distance(xyz_measured, sensor_position)
    # # Drop all rows that are out of reach for sensor
    # for all rows where distance_to_sensor > sensor.get_reach():
    #   drop(row)
    # # Drop all rows where sensor pollingrate is not quick enough:
    # for all rows where (timediff(row[n+1]-row[n]) < sensor.get_pollingrate():
    #   drop(row)
    # # Drop randomized rows as decaying function of distance to sensor, model function so that at sensor.get_reach()+1 the likelihood of measurement reaches 0
    # for all rows:
    #   calculate decay likelihood as function of distance
    #   if random(0,1) > decay likelihood:
    #       drop(row)

    print(measurement_dataframe)

    return measurement_dataframe


########### EXECUTE data ingestion
# path = r'scripts\synthetic_data_generation\assets\sampledata\occupancy_presence_and_trajectories.txt'

# imported_dataset = import_occupancy_presence_dataset(path, import_rows_count=5)

#print(imported_dataset)

# test_coord_sys = CoordinateSystem(3,1,0, 30,-15,45)
# test_sensor = Sensor('RFID', test_coord_sys, 30, 10, 500)

# simulate_measure_data_from_true_positions(imported_dataset, test_sensor)


########### EXECUTE plotting examples (Showcase part)
def plot_examples(sensor, coord_sys, point_x, point_y, point_z, repeated_steps):
    #print(sensor)
    #print(coord_sys)

    #print(sensor.generate_random_point_in_sphere())
    plot_randomized_sphere(sensor, repeated_steps)

    #print(transform_cartesian_coordinate_system(1,5,-1, coord_sys))
    plot_point_in_two_coordinate_systems(point_x, point_y, point_z, coord_sys, plot_system_indicators = True)

# test_coord_sys_2 = CoordinateSystem(3,1,-3, 90,135,42)
# test_sensor_2 = Sensor('RFID', test_coord_sys, 30, 10, 500)

#plot_examples(test_sensor_2, test_coord_sys_2, 1, 5, -1, 3000)