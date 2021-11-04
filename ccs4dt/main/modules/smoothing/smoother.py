class Smoother:
    """
    Smooth raw sensor data. Remove noise and spikes from raw sensor data.

    :param batch: The input measurement batch
    :type batch: pandas.DataFrame
    """

    def __init__(self, input_batch_df):
        self.__input_batch_df = input_batch_df


    def run(self):
        """
        Run moving average rolling window algorithm to smooth the raw data

        :rtype: pd.DataFrame
        :returns: smoothed input batch
        """
        # Apply smoothing for each object
        for object_identifier in self.__input_batch_df.index.levels[1].unique():
            row_mask = self.__input_batch_df.index.get_level_values('object_identifier') == object_identifier
            column_mask = ['x', 'y', 'z']
            self.__input_batch_df.loc[row_mask, column_mask] = self.__smooth(self.__input_batch_df.loc[row_mask, column_mask])
        return self.__input_batch_df


    def __smooth(self, df):
        """Apply moving avergae of 3s smoothing"""
        # TODO: Evaluate other smoothing algorithms (median, kalman, etc)
        return df.rolling('3s', on=df.index.get_level_values('timestamp')).mean()