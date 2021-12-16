import pandas as pd
import datetime



def import_occupancy_presence_dataset (filepath, drop_irrelevant_columns = True):
    
    if drop_irrelevant_columns == True:
        relevant_columns = ['time', 'day_id', 'x', 'y', 'occupant_id', 'camera_id']
        import_file = pd.read_csv(filepath, header = 0,  usecols = relevant_columns)

        # import_file['date'] = datetime.datetime(2019, 6, import_file['day_id'])
        # import_file['date_time'] = datetime.datetime.combine(import_file['date'], import_file['time'])
    else:
        raise ValueError('Case not covered (include irrelevant columns)')

    return import_file

def add_synthestic_sensor_data(initial_dataset, sensor):


    return dataset



class Sensor(object):

    def __init__(self, sensor_name, sensor_min_precision, sensor_max_lag):
        self.sensor_name = sensor_name
        self.sensor_min_precision = sensor_min_precision
        self.sensor_max_lag = sensor_max_lag


    def __str__(self):
        return(str('Sensor with name: ' + self.sensor_name + 
        '\nand min precision: ' + str(self.sensor_min_precision) + ', ' + 
        '\nand max timelag: ' + str(self.sensor_max_lag) ))


test_sensor = Sensor('RFID1', 30, 0)
    
path = r'scripts\synteticDataGeneration\assets\sampledata\occupancy_presence_and_trajectories.csv'
print(import_occupancy_presence_dataset(path))

