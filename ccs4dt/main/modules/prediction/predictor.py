class Predictor:
    """
    Predict the position of a given object and timestamp based on the measurements of different sensors.

    :param input_batch_df: Input batch dataframe
    :type input_batch_df: pandas.DataFrame
    """

    def __init__(self, input_batch_df):
        self.__input_batch_df = input_batch_df

    def __predict_position(self, df):
        """
        Predict position for a given object and timestamp based on the measurements of different sensors
        :param df: df for a given object and timestamp containing the measurements of different sensors
        :type df: pd.DataFrame
        :returns: Prediction of position
        :rtype: pd.DataFrame
        """
        return df[['x', 'y', 'z']].mean(axis=0)

    def run(self):
        """
        Run predictor

        :returns: The predicted positions for each object
        :rtype: pd.DataFrame
        """
        df = self.__input_batch_df
        return df.groupby([
            df.index.get_level_values('timestamp'),
            df.index.get_level_values('object_identifier')
        ]).apply(self.__predict_position)
