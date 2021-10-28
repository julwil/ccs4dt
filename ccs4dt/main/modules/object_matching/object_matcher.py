import numpy as np
import uuid


class ObjectMatcher:
    """
    Matching of different object_identifiers that refer to the same physical object and assign a new shared uuid.

    Each sensor refers differently to the objects it records. E.g. camera sensor uses uuids to identify objects
    whereas bluetooth sensor uses MAC address. In reality the different object_identifiers map to the same physical
    object. ObjectMatching groups object_identifiers that most probably belong to the same physical object and re-indexes
    object_identifiers of members of the same cluster with a new shared uuid. This makes further processing easier.

    :param input_batch_df: Input batch dataframe
    :type input_batch_df: pandas.DataFrame
    """

    def __init__(self, input_batch_df):
        self.__input_batch_df = input_batch_df

    def run(self):
        """
        Run object_matching and re-index object_identifiers

        :returns: The re-indexed input_batch_df
        :rtype: pd.DataFrame
        """

        for cluster in self.__compute_clusters(self.__input_batch_df):
            cluster_uuid = str(uuid.uuid4())
            for object_identifier in cluster:
                mask = self.__input_batch_df.index.get_level_values('object_identifier') == object_identifier
                self.__input_batch_df.loc[mask, 'object_identifier'] = cluster_uuid
        return self.__input_batch_df

    def __compute_clusters(self, df):
        """
        Compute clusters of object_identifiers.
        :type df: pd.DataFrame
        :returns: A list of clusters. Each cluster is a set of related object_identifiers that map to the same
         physical object
        :rtype: list
        """

        filtered_df = self.__filter_incomplete_object_identifiers(df)

        if filtered_df.empty:
            raise RuntimeException('Not enough data to perform id matching.')

        filtered_df = filtered_df.groupby('object_identifier').first()
        sensor_identifiers = filtered_df['sensor_identifier'].unique().tolist()
        sensors_dfs = self.__split_df_by_sensor_identifier(filtered_df)

        # iterator_df can be an arbitrary one. Just make sure, they all have the same cardinality.
        iterator_df_key = sensor_identifiers.pop()
        iterator_df = sensors_dfs.pop(iterator_df_key)

        # A cluster is a combination of object_identifiers (each from a different sensor) that best match together.
        # E.g. clusters = [(obj_1_rfid, obj_3_cam, obj_2_wifi)]
        clusters = []
        for i_object_identifier, i_data in iterator_df.iterrows():
            # New cluster contains iterator object_identifier by default.
            cluster = set([i_object_identifier])
            i = np.array([i_data['x'], i_data['y'], i_data['z']])

            # For all other sensors with their df
            for sensor_identifier in sensor_identifiers:
                min_norm = float('inf')
                min_object_identifier = None

                # Find the object_identifier that best fits our iterator object_identifier.
                for j_object_identifier, j_data in sensors_dfs[sensor_identifier].iterrows():
                    j = np.array([j_data['x'], j_data['y'], j_data['z']])

                    # The closer the distance between coordinates, the more likely it is the same physical object.
                    norm = np.linalg.norm(i - j)
                    if norm <= min_norm:
                        min_norm = norm
                        min_object_identifier = j_object_identifier

                cluster.add(min_object_identifier)
            clusters.append(cluster)
        return clusters

    def __filter_incomplete_object_identifiers(self, df):
        """
        Filter df such that it only contains time windows with complete data. This means that for a given time window,
        e.g. 1s, all object_identifiers are present.
        :param df: input dataframe
        :type df: pd.DataFrame
        :rtype: pd.DataFrame
        """
        object_identifiers = df.index.levels[1].unique()

        def should_include(x):
            return object_identifiers.isin(x.index.get_level_values('object_identifier')).all()

        return df.groupby('timestamp').filter(lambda x: should_include(x))

    def __split_df_by_sensor_identifier(self, df):
        """
        Group df by column 'sensor_identifier' and return a dict with grouped dfs.
        :param df: Input dataframe
        :type df: pandas.DataFrame
        :returns: Dict with sensor_identifier as key and dataframe as value
        :rtype: dict
        """

        sensor_identifiers = df['sensor_identifier'].unique().tolist()
        result = {}
        for sensor_identifier in sensor_identifiers:
            result[sensor_identifier] = df.loc[df['sensor_identifier'] == sensor_identifier]
        return result
