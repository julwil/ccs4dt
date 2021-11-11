import pandas as pd

class Upsampler:
    """
    Upsample input data batch

    :param input_data_batch: Input data batch
    :type input_data_batch: pd.DataFrame
    """

    def __init__(self, input_data_batch):
        self.__input_data_batch = input_data_batch


    def run(self):
        """
        Run upsampling

        :returns: upsampled input data batch
        :rtype: pd.DataFrame
        """
        df = self.__input_data_batch
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').round('1s')
        # Remove duplicates from this group
        df = df.groupby(['timestamp', 'object_identifier', 'sensor_identifier'], as_index=False).first()
        df.set_index(keys=['timestamp'], inplace=True, drop=True)
        df = df.groupby(['object_identifier', 'sensor_identifier'], as_index=False).apply(self.__upsample)
        df.set_index(keys=[df.index.get_level_values('timestamp'), 'object_identifier'], inplace=True)
        return df


    def __upsample(self, df):
        """
        Upsample df
        """
        # in case we have more than 1 measurement per object/sensor/time, take only first
        df = df.asfreq('1S')
        dont_fill = ['x', 'y', 'z']
        mask = df.columns.isin(dont_fill) == False
        df.loc[:, mask] = df.loc[:, mask].fillna(method='ffill')
        return df