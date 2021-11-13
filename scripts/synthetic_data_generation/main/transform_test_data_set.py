import pandas as pd
import datetime
import uuid
import random as rand
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import os
import plotly.graph_objects as go
import random
import requests
import json



class Location(object):
    """This class represents a real world measurement location.

    :param name: Name of the location or venue
    :type name: string
    :param external_identifier: Code / ID of room in external system
    :type external_identifier: string
    :param sensors: Sensors placed inside the location for measurement purposes
    :type sensors: Array of Sensor
    """

    def __init__(self, name, external_identifier, sensors):
        
        self.name = name
        self.external_identifier =  external_identifier
        self.sensors = sensors
        

    # TODO: define __str__

    def get_location_name(self):

        return(self.name)

    def get_location_external_id(self):

        return(self.external_identifier)
 
    
    def construct_json_payload(self):

        sensors_dict = []

        for sensor in self.sensors:
            sensor_dict = {
                'identifier': str(sensor.get_sensor_id()),
                'type': str(sensor.get_sensor_type()),
                'x_origin': sensor.get_sensor_position()[0],
                'y_origin': sensor.get_sensor_position()[1],
                'z_origin': sensor.get_sensor_position()[2],
                'yaw': sensor.get_sensor_orientation()[0],
                'pitch': sensor.get_sensor_orientation()[1],
                'roll': sensor.get_sensor_orientation()[2],
                'measurement_unit': sensor.get_sensor_spatial_measurement_unit()           
            }

            sensors_dict.append(sensor_dict)

        location_json = {
            'name': str(self.get_location_name()),
            'external_identifier': str(self.get_location_external_id()),
            'sensors': sensors_dict
        }

        json_payload = json.dumps(location_json, indent=4)

        return json_payload



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
    :param sensor_precision: Precision of the sensor, i.e. minimal spatial resolution of the sensor in the measurement unit defined in parameter "sensor_spatial_measurement_unit"
    :type sensor_precision: numeric
    :param sensor_pollingrate: Pollingrate of the sensor, i.e. maximal temporal resolution of the sensor in the measurement unit defined in parameter "sensor_temporal_measurement_unit"
    :type sensor_pollingrate: numeric
    :param measurement_reach: Maximum measurement reach of the sensor, past this distance the sensor is not able to measure anything, measurement unit defined in parameter "sensor_spatial_measurement_unit"
    :type measurement_reach: numeric
    :param sensor_spatial_measurement_unit: Measurement unit of the sensor in a temporal dimension, allowed are the SI unit "s" (including all prefixes) and the non-SI units "min" (minutes), "h"(hours) and "d"(days)  TODO: NOT YET CONSIDERED
    :type sensor_spatial_measurement_unit: String
    :param sensor_temporal_measurement_unit: Measurement unit of the sensor in a spatial dimension, allowed are all available SI prefixes for meters TODO: NOT YET CONSIDERED
    :type sensor_temporal_measurement_unit: String
    :param sensor_identifier: Unique identifier of the sensor, defaults to automatically generated uuid4
    :type sensor_identifier: string
    :param stability_function_type: Defines type of the stability function of the measurement of the sensor, i.e. how large the signal degradation is based on distance between object to be measured and the sensor. Covered are 'linear' and 'static', default 'linear'
    :type stability_function_type: string
    """

    def __init__(self, sensor_type, coordinate_system, sensor_precision, sensor_pollingrate, measurement_reach, sensor_temporal_measurement_unit = 'ms', sensor_spatial_measurement_unit='cm', sensor_identifier = '', stability_function_type = 'linear'):
        # Type of the sensor
        self.sensor_type = sensor_type

        # Absolute position of the sensor inside the location (in cm)
        self.absolute_pos_x = coordinate_system.get_translation_x()
        self.absolute_pos_y = coordinate_system.get_translation_y()
        self.absolute_pos_z = coordinate_system.get_translation_z()

        # Determined by the geopraphical orientation in which the y-axis of the readers coordinate systems increases. 0-359 degrees where East=0, North=90, West=180 South=270
        self.orientation_xy = coordinate_system.get_yaw_xy()
        self.orientation_yz = coordinate_system.get_pitch_yz()
        self.orientation_xz = coordinate_system.get_roll_xz()

        # Coordinate system
        self.coordinate_system = coordinate_system

        # Set measurement units
        self.sensor_spatial_measurement_unit = str(sensor_spatial_measurement_unit)
        self.sensor_temporal_measurement_unit = str(sensor_temporal_measurement_unit)

        # Precision of the sensor
        self.sensor_precision = sensor_precision

        # Maximum measurement distance from the position of the sensor
        self.measurement_reach = measurement_reach

        # Pollingrate, how frequent the sensor will be able to measure
        self.sensor_pollingrate = sensor_pollingrate

        # Must be unique, identifier of the sensor
        if sensor_identifier == "":
            self.sensor_identifier = uuid.uuid4()
        else:
            self.sensor_identifier = str(sensor_identifier)

        # Stability_function (with what percentage the sensor randomly drops a measurement)
        if stability_function_type == 'linear':
            def linear_degradation(self, distance_to_point):
                
                drop_likelihood = distance_to_point/(self.measurement_reach)
                if drop_likelihood > 1: drop_likelihood = 1

                return(drop_likelihood)       
            self.stability_function = linear_degradation

        elif stability_function_type == 'static':

            def static_limit(self, distance_to_point):
                
                if distance_to_point > self.measurement_reach:
                    drop_likelihood = 1
                else:
                    drop_likelihood = 0

                return(drop_likelihood)

            self.stability_function = static_limit
        # TODO: implement exponential drop function
        else: 
            raise ValueError('Input for stability_function is not valid')

    def __str__(self):
        """String representation of the senor

        :return: Returns string of coordinate system parameters
        :rtype: string
        """
        return (str('Sensor of type "'+ self.sensor_type + '" with id: ' + str(self.sensor_identifier) +
                    '\nat absolute position: (' + str(self.absolute_pos_x) + ', ' + str(self.absolute_pos_y) + ', ' + str(self.absolute_pos_z) + ') ' + 
                    ' (x, y, z), with orientation (yaw [xy], pitch [yz], roll [xz]) (' +
                      str(self.orientation_xy) + '°, ' +  str(self.orientation_yz) + '°, ' +  str(self.orientation_xz) + '°), ' +
                    '\nand precision: ' + str(self.sensor_precision) + ' ' + str(self.sensor_precision_measurement_unit) + ', ' +
                    '\nand pollingrate: ' + str(self.sensor_pollingrate) + ' ' + str(self.sensor_pollingrate_measurement_unit) +
                    '\nand the sensor drops measurements with a probability of ' + str(self.stability) + '%' ))

    def get_sensor_position(self):
        """Getter function for positional parameters (x,y,z) of sensors in the frame of reference      

        :return: Returns x,y,z-coordinate of the sensor in the frame of reference
        :rtype: tuple(numeric)
        """
        return (self.absolute_pos_x, self.absolute_pos_y, self.absolute_pos_z)

    def get_sensor_orientation(self):
        """Getter function for orientation parameters (yaw [xy], pitch [yz], roll [xz]) of sensors in the frame of reference      

        :return: Returns orientation parameters of the sensor in the frame of reference
        :rtype: tuple(numeric)
        """
        return (self.orientation_xy, self.absolute_pos_y, self.absolute_pos_z)

    def get_sensor_spatial_measurement_unit(self):
        """Getter function for spatial measurement unit of the sensor      

        :return: Returns spatial measurent unit of the sensor
        :rtype: string
        """
        return (self.sensor_spatial_measurement_unit)

    def get_sensor_temporal_measurement_unit(self):
        """Getter function for temporal measurement unit of the sensor      

        :return: Returns temporal measurent unit of the sensor
        :rtype: string
        """
        return (self.sensor_temporal_measurement_unit)

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

    def get_sensor_reach(self):
        """Getter function for the sensor reach       

        :return: Returns sensor reach
        :rtype: numeric
        """
        return (self.measurement_reach)
    
    def get_sensor_coordinate_system(self):
        """Getter function for the sensor coordinate system       

        :return: Returns sensor coordinate system
        :rtype: CoordinateSystem
        """
        return(self.coordinate_system)

    def check_sensor_measurement_distance(self, distance_to_point):

        random_chance = random.randint(0,100)
        # print(random_chance) 
        # print(distance_to_point)
        # print(type(distance_to_point))
        is_measured = True if random_chance > (self.stability_function(self, distance_to_point)*100) else False

        return(is_measured)

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
    
    def calculate_distance_to_point(self, point_abs_x, point_abs_y, point_abs_z):

        distance = ((self.get_sensor_position()[0]-point_abs_x)**2 + (self.get_sensor_position()[1]-point_abs_y)**2 + (self.get_sensor_position()[2]-point_abs_z)**2)**(1/2)

        return abs(distance)



def transform_cartesian_coordinate_system(point_x, point_y, point_z, coordinate_system, inverse_transformation = False, output_transformation_matrix = False):
    """Transforms positional coordinates of a point in a specific coordinate system into its frame of reference (f.o.r)

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



def plot_randomized_sphere(sensor, randomization_steps):
    """Plots (in 3D) x randomized measurements and sphere of precision based on sensor parameters around origin, where x = randomization_steps

    :param sensor: Sensor for which the measurements should be simulated
    :type sensor: Sensor
    :param randomization_steps: Amount of measurements that should be simulated
    :type randomization_steps: integer

    :return: Returns nothing
    :rtype: None
    """

    # Generate simulated measurement points
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

    # Add simulated measurement points to figure
    trace_simulated_measurements = go.Scatter3d(
        x = x_array,
        y = y_array,
        z = z_array,
        mode='markers',
        name='Simulated measure'
    )

    # Add true position (origin) to figure
    trace_true_position = go.Scatter3d(
        x = [0],
        y = [0],
        z = [0],
        marker=dict(
            color='black',
            size=20),
        mode = 'markers',
        name = 'True position'
    )

    def spheres(radius, sphere_color = '#1d212e', sphere_opacity = 0.2): 
        """Creates trace for sphere as surface in plotly graph_object

        :param radius: Radius of the sphere to be plotted
        :type radius: numerical
        :param sphere_color: Color of the sphere in plot, default is #1d212e
        :type sphere_color: str
        :param sphere_opacity: Opacity of the sphere in plot, default is 0.2
        :type sphere_opacity: numerical

        :return: Returns constructed trace that can directly be added to figure
        :rtype: str
        """

        # Set up 100 points. First, do angles
        theta = np.linspace(0,2*np.pi,100)
        phi = np.linspace(0,np.pi,100)
        
        # Set up coordinates for points on the sphere
        x0 = radius * np.outer(np.cos(theta),np.sin(phi))
        y0 = radius * np.outer(np.sin(theta),np.sin(phi))
        z0 = radius * np.outer(np.ones(100),np.cos(phi))
        
        # Set up trace
        trace= go.Surface(x=x0, y=y0, z=z0, colorscale=[[0,sphere_color], [1,sphere_color]], opacity = sphere_opacity, name='Sphere simulation boundaries (as given by sensor precision)', showlegend = True)
        trace.update(showscale=False)

        return trace

    # Generate simulation sphere trace
    trace_simulation_sphere = spheres(sensor.get_sensor_precision())

    # Add datapoints to figure
    fig = go.Figure(data= trace_simulated_measurements)
    fig.add_trace(trace_true_position)
    fig.add_trace(trace_simulation_sphere)

    # Add title to figure
    fig.update_layout(title= ('Simulated ' + str(randomization_steps) + ' measures of the sensor (with precision: ' + str(sensor.get_sensor_precision()) + ') around true position'),
    title_font_size = 40)

    # Generate html file for plot
    fig.write_html('scripts/synthetic_data_generation/assets/generated_graphs/plot_measure_simulations.html')
    
    return None



def plot_point_in_two_coordinate_systems(point_x, point_y, point_z, point_coord_sys, plot_system_indicators = True):
    """Plots two coordinate systems (frame of reference and point coordinate system that contain the same point transformed).
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

    # Define basic coordinate system vectors
    origin = [0,0,0]
    x_identity = [1,0,0]
    y_identity = [0,1,0]
    z_identity = [0,0,1]

    # Calculate frame of reference coordinate system vectors in point coordinate system
    transformed_frame_of_reference_origin = list(transform_cartesian_coordinate_system(origin[0], origin[1], origin[2], point_coord_sys, inverse_transformation = True))
    transformed_frame_of_reference_x_identity = list(transform_cartesian_coordinate_system(x_identity[0], x_identity[1], x_identity[2], point_coord_sys, inverse_transformation = True))
    transformed_frame_of_reference_y_identity = list(transform_cartesian_coordinate_system(y_identity[0], y_identity[1], y_identity[2], point_coord_sys, inverse_transformation = True))
    transformed_frame_of_reference_z_identity = list(transform_cartesian_coordinate_system(z_identity[0], z_identity[1], z_identity[2], point_coord_sys, inverse_transformation = True))

    trace_point_in_coordinate_system = go.Scatter3d(
        x=[point_x],
        y=[point_y],
        z=[point_z],
        mode='markers',
        name='Point in relative coordinates'
    )

    def vector_plot(tvects, vector_name_prefix , orig = [0, 0, 0]):
        """Generates vectors using plotly, between supplied vector and endpoint (tvects and origin)

        :param tvects: x-,y-,z-coordinate
        :type tvects: list
        :param vector_name_prefix: prefix of name as shown in plot legend
        :type vector_name_prefix: string
        :param orig: x-,y-,z-coordinate of origin
        :type orig: list

        :return: Returns transformed coordinate vectors for x-,y-,z-coordinate 
        :rtype: Tuple (x,y,z)
        """

        coords = [[orig, np.sum([orig, v], axis = 0)] for v in tvects]


        data = []
        for i,c in enumerate(coords):
            if i == 0:
                vector_name_suffix = ': x-coordinate'
            elif i == 1:
                vector_name_suffix = ': y-coordinate'
            elif i == 2:
                vector_name_suffix = ': z-coordinate'
            X1, Y1, Z1 = zip(c[0])
            X2, Y2, Z2 = zip(c[1])
            vector = go.Scatter3d(x = [X1[0], X2[0] - X1[0]],
                                y = [Y1[0], Y2[0] - Y1[0]],
                                z = [Z1[0], Z2[0] - Z1[0]],
                                #line = dict(color="#ffe476"),
                                marker = dict(size = [0,10],
                                                color = ['DarkSlateGrey'],
                                                line = dict(width=1,
                                                        color='DarkSlateGrey')),
                                name = str(vector_name_prefix)+str(vector_name_suffix))
            data.append(vector)

        return data[0], data[1], data[2]

    
    # Define vectors for point coordinate system
    x_identity_vector_point_coordinate_system, y_identity_vector_point_coordinate_system,\
     z_identity_vector_point_coordinate_system = (vector_plot([x_identity, y_identity, z_identity],'Point coordinate system'))

    # Define vectors for frame of refrence in point coordinate system
    x_identity_frameofreference_in_pointcoordinatesystem_vector, \
        y_identity_frameofreference_in_pointcoordinatesystem_vector, \
            z_identity_frameofreference_in_pointcoordinatesystem_vector = (vector_plot([transformed_frame_of_reference_x_identity, 
            transformed_frame_of_reference_y_identity, transformed_frame_of_reference_z_identity], 'Frame of reference', orig = transformed_frame_of_reference_origin))

    elements_to_plot_in_point_coordinate_system = [trace_point_in_coordinate_system, 
    x_identity_vector_point_coordinate_system, y_identity_vector_point_coordinate_system, z_identity_vector_point_coordinate_system, 
    x_identity_frameofreference_in_pointcoordinatesystem_vector, y_identity_frameofreference_in_pointcoordinatesystem_vector, z_identity_frameofreference_in_pointcoordinatesystem_vector]

    # Calculate max absolute coordinate position to be able to scale graphic correctly in all directions (to avoid distortion)
    all_coordinates = []
    for i in elements_to_plot_in_point_coordinate_system:
            all_coordinates.append(i.x[0])
            all_coordinates.append(i.y[0])
            all_coordinates.append(i.z[0])
    # identify maximum absolute coordinate from all coordinates, round to nearest 10
    max_coordinate = round(max([abs(ele) for ele in all_coordinates]),-1)

    # Defines number of grid divisions based on scaling
    if max_coordinate <= 10:
        number_grid_lines = 10
    elif max_coordinate <= 20:
        number_grid_lines = 20
    elif max_coordinate <= 50:
        number_grid_lines = 50
    else:
        number_grid_lines = 100

    fig = go.Figure(data= elements_to_plot_in_point_coordinate_system)

    # Formatting of plot
    fig.update_layout(scene = dict(
                    xaxis_title='x coordinate',
                    yaxis_title='y coordinate',
                    zaxis_title='z coordinate'),
                    title = 'Point coordinate system plot',
                    title_font_size=50,
                    colorway=['#e02a44','#8EDFFF', '#1974D2',  '#00308F', '#F89F05', '#E36005', '#D04711', '#7f7f7f', '#bcbd22', '#17becf'])

    fig.update_layout(scene = dict(
                    xaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),
                    yaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),
                    zaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),))


    # Generate html file for plot
    fig.write_html('scripts/synthetic_data_generation/assets/generated_graphs/plot_point_in__coord_system.html')



    # Start constructing figure for frame of reference plot
    # Get transformed point in f.o.r.
    transformed_point = transform_cartesian_coordinate_system(point_x, point_y, point_z, point_coord_sys)
    # Construct point in frame of reference
    trace_point_in_frame_of_reference = go.Scatter3d(
        x=[transformed_point[:3][0]],
        y=[transformed_point[:3][1]],
        z=[transformed_point[:3][2]],
        mode='markers',
        name='Point in frame of reference'
    )

    # Calculate point coordinate system vectors in frame of reference
    transformed_origin = list(transform_cartesian_coordinate_system(origin[0], origin[1], origin[2], point_coord_sys))
    transformed_x_identity = list(transform_cartesian_coordinate_system(x_identity[0], x_identity[1], x_identity[2], point_coord_sys))
    transformed_y_identity = list(transform_cartesian_coordinate_system(y_identity[0], y_identity[1], y_identity[2], point_coord_sys))
    transformed_z_identity = list(transform_cartesian_coordinate_system(z_identity[0], z_identity[1], z_identity[2], point_coord_sys))

    # Define vectors for frame of reference coordinate system plot
    x_identity_vector_frame_of_reference, y_identity_vector_point_frame_of_reference,\
     z_identity_vector_frame_of_reference = (vector_plot([x_identity, y_identity, z_identity],'Frame of reference'))

    # Define vectors for point coordinate system in frame of reference plot
    x_identity_pointcoordinatesystem_in_frameofreference_vector, \
        y_identity_pointcoordinatesystem_in_frameofreference_vector, \
            z_identity_pointcoordinatesystem_in_frameofreference_vector = (vector_plot([transformed_x_identity, 
            transformed_y_identity, transformed_z_identity], 'Point Coordinate system', orig = transformed_origin))

    # Define points and vectors to be plotted
    elements_to_plot_in_frame_of_reference =[trace_point_in_frame_of_reference, 
    x_identity_vector_frame_of_reference, y_identity_vector_point_frame_of_reference, z_identity_vector_frame_of_reference, 
    x_identity_pointcoordinatesystem_in_frameofreference_vector, y_identity_pointcoordinatesystem_in_frameofreference_vector, z_identity_pointcoordinatesystem_in_frameofreference_vector]
    
    # Establish plot
    fig = go.Figure(data= elements_to_plot_in_frame_of_reference)

    # Calculate max absolute coordinate position to be able to scale graphic correctly in all directions (to avoid distortion)
    all_coordinates = []
    for i in elements_to_plot_in_frame_of_reference:
            all_coordinates.append(i.x[0])
            all_coordinates.append(i.y[0])
            all_coordinates.append(i.z[0])
    # identify maximum absolute coordinate from all coordinates, round to nearest 10
    max_coordinate = round(max([abs(ele) for ele in all_coordinates]),-1)

    # Defines number of grid divisions based on scaling
    if max_coordinate <= 10:
        number_grid_lines = 10
    elif max_coordinate <= 20:
        number_grid_lines = 20
    elif max_coordinate <= 50:
        number_grid_lines = 50
    else:
        number_grid_lines = 100

    # Formatting of plot
    fig.update_layout(scene = dict(
                    xaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),
                    yaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),
                    zaxis = dict(nticks=number_grid_lines, range=[-max_coordinate,max_coordinate],),))

    fig.update_layout(scene = dict(
                    xaxis_title='x coordinate',
                    yaxis_title='y coordinate',
                    zaxis_title='z coordinate'),
                    title = 'Frame of reference plot',
                    title_font_size=50,
                    colorway=['#e02a44', '#F89F05', '#E36005', '#D04711', '#8EDFFF', '#1974D2', '#00308F', '#7f7f7f', '#bcbd22', '#17becf'])

    # Generate html file for plot
    fig.write_html('scripts/synthetic_data_generation/assets/generated_graphs/plot_point_in_frame_of_reference.html')


    return None



# TODO: only works with standard dataset, potentially extend to other dataset formats
def import_occupancy_presence_dataset (filepath, import_rows_count, drop_irrelevant_columns = True, transform_to_3D_data = True, starting_date = '01.06.2019', date_format = '%d.%m.%Y'):
    """Imports the true position dataset of the given format from source: https://www.kaggle.com/claytonmiller/occupancy-presencetrajectory-data-from-a-building/version/1

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

        # Take day id and transform into datetime
        import_file['date'] = datetime.datetime.strptime(starting_date, date_format) + pd.to_timedelta(np.ceil(import_file['day_id']), unit="D")

        # Combine date and time columns and convert to datetime format
        import_file['date_time'] = pd.to_datetime((import_file['date'].astype(str)) + ' ' + (import_file['time']))

    else:
        raise ValueError('Case not covered (include irrelevant columns)')

    return import_file



def simulate_measure_data_from_true_positions(true_position_dataframe, sensor):
    """Simulates measurement of one sensor

    :param true_position_dataframe: Dataframe that contains true positions
    :type true_position_dataframe: dataframe
    :param sensor: Sensor for which the measurements should be simulated
    :type sensor: Sensor

    :return: Returns the simulated measurement output of the sensor as dataframe
    :rtype: dataframe
    """

    # Generate empty data frame for measurements
    measurement_dataframe = pd.DataFrame()

    # Add time and date column to measured data frame
    measurement_dataframe['date'] = [x for x in true_position_dataframe['date']]
    measurement_dataframe['time'] = [x for x in true_position_dataframe['time']]
    measurement_dataframe['date_time'] = [x for x in true_position_dataframe['date_time']]

    # Add sensor id, sensor type and occupant id
    measurement_dataframe['occupant_id'] = true_position_dataframe['occupant_id'].astype(str)
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
    measurement_dataframe['x_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, sensor.get_sensor_coordinate_system())[0]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])
    measurement_dataframe['y_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, sensor.get_sensor_coordinate_system())[1]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])
    measurement_dataframe['z_measured_rel_pos'] = ([(transform_cartesian_coordinate_system(x, y, z, sensor.get_sensor_coordinate_system())[2]) for x, y, z in zip(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])])

    ## TODO: Validiation / Testing of this part outstanding
    # Calculates distance between sensor and point and stores in dataframe
    measurement_dataframe['distance'] = sensor.calculate_distance_to_point(measurement_dataframe['x_measured_abs_pos'], measurement_dataframe['y_measured_abs_pos'], measurement_dataframe['z_measured_abs_pos'])

    print(measurement_dataframe)

    # Adds drop flag if distance to point is larger than sensore measurement range
    measurement_dataframe['drop_due_to_distance'] = [sensor.check_sensor_measurement_distance(x) for x in measurement_dataframe['distance']]

    print(measurement_dataframe)

    # # Adds time difference between column and last column TODO: How should this work for a shift different of 1?
    # measurement_dataframe['timediff'] = measurement_dataframe['time'] - measurement_dataframe.shift(-1)['time']



    # # TODO: Add measurement boundary, polling rate and decaying stability (as function of measurement distance)
    # # PSEUDOCODE HERE
    # # calculate distance(xyz_measured, sensor_position)
    # # # Drop all rows that are out of reach for sensor
    # # for all rows where distance_to_sensor > sensor.get_reach():
    # #   drop(row)

    # # # Drop all rows where sensor pollingrate is not quick enough:
    # # for all rows where (timediff(row[n+1]-row[n]) < sensor.get_pollingrate():
    # #   drop(row)

    # # # Drop randomized rows as decaying function of distance to sensor, model function so that at sensor.get_reach()+1 the likelihood of measurement reaches 0
    # # for all rows:
    # #   calculate decay likelihood as function of distance
    # #   if random(0,1) > decay likelihood:
    # #       drop(row)


    return measurement_dataframe



# TODO: Write documentation
def function_wrapper_data_ingestion(path, import_rows, measurement_sensor):

    imported_dataset = import_occupancy_presence_dataset(path, import_rows_count = import_rows)

    simulation_data_dataframe = simulate_measure_data_from_true_positions(imported_dataset, measurement_sensor)

    return simulation_data_dataframe



# TODO: Write documentation
def function_wrapper_example_plots(example_sensor, point_x, point_y, point_z, repeated_steps):

    example_coord_sys = example_sensor.get_sensor_coordinate_system()

    plot_randomized_sphere(example_sensor, repeated_steps)

    plot_point_in_two_coordinate_systems(point_x, point_y, point_z, example_coord_sys, plot_system_indicators = True)

    return None



# TODO: Write documentation
def convert_measurement_dataframe_to_api_conform_payload(dataframe, generate_file = False):

    dataframe = dataframe[['occupant_id','x_measured_rel_pos','y_measured_rel_pos','z_measured_rel_pos','sensor_type','sensor_id','date_time']]

    dataframe = dataframe.rename(columns = {'occupant_id':'object_identifier', 'x_measured_rel_pos':'x', 'y_measured_rel_pos':'y', 'z_measured_rel_pos':'z',
     'sensor_id':'sensor_identifier', 'sensor_type':'sensor_type', 'date_time':'timestamp'})

    if generate_file == True:
        json_data = dataframe.to_json(path_or_buf= 'scripts/synthetic_data_generation/assets/generated_files/measurement.json', default_handler=str, orient='records')
    elif generate_file == False:
        json_data = dataframe.to_json(default_handler=str, orient='records')

    return(json_data)



# TODO: Write documentation
def generate_random_mac_address():

    """Generation of a random MAC Address

    :return: Returns a randomized 12-byte MAC Address divided by semicolons
    :rtype: string
    """

    mac_address = []
    for i in range(1,7):
        mac_address_characters = "".join(random.sample("0123456789abcdef",2))
        mac_address.append(mac_address_characters)
    randomized_mac_adress = ":".join(mac_address)
    return randomized_mac_adress



# TODO: Write documentation
# TODO: Request not correct, validate with Julius
def API_post_input_batch_call(API_endpoint_path, payload, location_id):

    response = requests.post(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs', json = json.loads(payload))

    # 202 Code with successful delivery
    response = response.json() if response and response.status_code == 202 else None

    return (response, response['id'], response['status'], response['location_id'])



# TODO: Write documentation
def API_get_input_batch_by_id_call(API_endpoint_path, location_id, input_batch_id):
    
    response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs/' + str(input_batch_id))

    json_data = response.json() if response and response.status_code == 200 else None

    return json_data



# TODO: Write documentation
def API_get_all_input_batches_call(API_endpoint_path, location_id):
    
    response = requests.get(API_endpoint_path + '/locations/'+ str(location_id) + '/inputs')

    json_data = response.json() if response and response.status_code == 200 else None

    return json_data



# TODO: Write documentation
def API_get_location_by_id_call(API_endpoint_path, location_id):
    
    response = requests.get(API_endpoint_path + '/locations/'+ str(location_id))
    json_data = response.json() if response and response.status_code == 200 else None

    return json_data



# TODO: Write documentation
def API_get_all_locations_call(API_endpoint_path):
    
    response = requests.get(API_endpoint_path + '/locations')
    json_data = response.json() if response and response.status_code == 200 else None

    return json_data



# TODO: Write documentation
def API_post_new_location_call(API_endpoint_path, payload):
    
    response = requests.post(API_endpoint_path + '/locations', json = json.loads(payload))

    # 201 Code with successful delivery
    response = response.json() if response and response.status_code == 201 else None

    return (response, response['id'], response['name'])



# TODO: Write documentation
def API_get_output_batch_call(API_endpoint_path, location_id, batch_id):

    response = requests.get(API_endpoint_path + '/locations/' + str(location_id) + '/inputs/' + str(batch_id) + '/outputs')
    json_data = response.json() if response and response.status_code == 200 else None

    return json_data



# Test setup parameters 
endpoint_path = 'http://localhost:5000'

test_coord_sys = CoordinateSystem(6,-2,4, 0,0,0)
test_coord_sys2 = CoordinateSystem(0,-1,1, 2,3,4)
test_sensor = Sensor('RFID', test_coord_sys, 30, 10, 800)
test_sensor3 = Sensor('camera', test_coord_sys, 1, 1, 500)
test_sensor2 = Sensor('WiFi 2.4GHz', test_coord_sys2, 30, 10, 4000)

# test_location = Location('test_name', 'test_id_ext', [test_sensor,test_sensor2, test_sensor3])
# test_location_payload = test_location.construct_json_payload()

# API request: GET all locations
#API_get_all_locations_call(endpoint_path)

# API request: POST new location
#post_location_response, test_location_id, test_location_name = (API_post_new_location_call(endpoint_path, test_location_payload))

# Ingest true position data
data_ingested = (function_wrapper_data_ingestion(str(os.getcwd()) + '/scripts/synthetic_data_generation/assets/sampledata/occupancy_presence_and_trajectories.csv', 200, test_sensor))

# Generate synthetic measurement data payload
#API_payload = convert_measurement_dataframe_to_api_conform_payload(data_ingested)

# API request: POST new input batch
#post_input_batch_response, input_batch_id, input_batch_status, location_id_for_input_batch = API_post_input_batch_call(endpoint_path, API_payload, test_location_id)

# API request: GET input batch status
#print('Get input batch by id')
#print(API_get_input_batch_by_id_call(endpoint_path, location_id_for_input_batch, input_batch_id))

# API request: GET output batch based on generated id
# Pause to let API process
# import time
# time.sleep(20)
# print('Get output batch by id')

# output_batch_response = (API_get_output_batch_call(endpoint_path, location_id_for_input_batch, input_batch_id))

# print(output_batch_response)







#function_wrapper_example_plots(test_sensor, 1, 1, 1, 100)






## TODO Placholder convert function
#  def __convert_units(self, row):
#         """
#         Handle conversion of measurement units.
#         :param row: pd.Series
#         :return: pd.Series
#         """
#         sensor = self.__sensors[row['sensor_identifier']]
#         factor = 1

#         if sensor['measurement_unit'] == MeasurementUnit.CENTIMETER:
#             return row

#         if sensor['measurement_unit'] == MeasurementUnit.MILLIMETER:
#             factor = 0.1

#         if sensor['measurement_unit'] == MeasurementUnit.METER:
#             factor = 100

#         for axis in ['x', 'y', 'z']:
#             row[axis] *= factor

#         return row


## TODO: Randomize occupant_id for different sensors (e.g. unique MACaddress)
## TODO: Use pollingrate from livealytics dataset

# TODO: Bash script to run & Bash script with parameter input





