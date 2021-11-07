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
        df.set_index(keys=['timestamp'], inplace=True, drop=True)
        df = df.groupby(['object_identifier', 'sensor_identifier']).apply(lambda x: self.__upsample(x))
        df.reset_index(level=1, drop=True, inplace=True)
        df.drop(columns=['object_identifier'], inplace=True)
        df = df.swaplevel()
        return df


    def __upsample(self, df):
        """
        Upsample df
        """
        df = df.asfreq('1S')
        dont_fill = ['x', 'y', 'z']
        mask = df.columns.isin(dont_fill) == False
        df.loc[:, mask] = df.loc[:, mask].fillna(method='ffill')
        return df