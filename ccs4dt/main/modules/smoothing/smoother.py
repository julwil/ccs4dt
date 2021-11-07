import numpy as np
from pykalman import KalmanFilter

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
            self.__input_batch_df.loc[row_mask, column_mask] = self.__kalman_filtering(self.__input_batch_df.loc[row_mask, column_mask])
        return self.__input_batch_df


    def __moving_average(self, df):
        """Apply moving avergae of 3s smoothing"""
        # TODO: Evaluate other smoothing algorithms (median, kalman, etc)
        return df.rolling('3s', on=df.index.get_level_values('timestamp')).mean()

    def __kalman_filtering(self, df):
        import logging
        logging.error(df)
        measurements = np.ma.masked_invalid(df[['x', 'y', 'z']])

        # [x, x_hat, y, y_hat, z, z_hat]
        initial_state_mean = [measurements[0, 0],
                              0,
                              measurements[0, 1],
                              0,
                              measurements[0, 2],
                              0,
                              ]

        # F: prediction matrix x_k = x_k-1
        transition_matrix = [[1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 1]]

        # H: measurements matrix (take x, y, z)
        observation_matrix = [[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]]

        kf = KalmanFilter(transition_matrices=transition_matrix,
                           observation_matrices=observation_matrix,
                           initial_state_mean=initial_state_mean)

        kf = kf.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
        x_smoothed = smoothed_state_means[:, 0]
        y_smoothed = smoothed_state_means[:, 2]
        z_smoothed = smoothed_state_means[:, 4]
        df['x'] = x_smoothed
        df['y'] = y_smoothed
        df['z'] = z_smoothed
        logging.error(df)
        logging.error('\n\n')
        return df
