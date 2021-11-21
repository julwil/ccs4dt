import itertools
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd


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
        self.__clusters = dict()


    def __get_vectors_by_object_identifiers(self, object_identifiers, df):
        """
        Return a list of vectors of x,y,z, based on a list of object_identifiers.

        :param object_identifiers: list of object_identifiers to retrieve vectors for
        :type object_identifiers: list
        :param df: df to retrieve vectors from
        :type df: pd.DataFrame
        :rtype: list
        """
        vectors = []
        for object_identifier in object_identifiers:
            mask = df.index.get_level_values('object_identifier') == object_identifier
            vector = df.loc[mask, ['x', 'y', 'z']].to_numpy().squeeze()
            vectors.append(vector)
        return vectors

    def __cumulative_norm(self, vectors):
        """
        Compute the cumulative norm / euclidean distance between a list (1,2,3,4,...,n) of vectors.
        Calculates and sums up euclidean distance between all possible pairs of vectors in the tuple.

        :param vectors: A tuple of vectors (x,y,z)
        :type vectors: list
        :rtype: float
        """
        cumulative_norm = 0.

        if len(vectors) == 1:
            return cumulative_norm

        index = [[i] for i in range(len(vectors))]
        index_combinations = list(itertools.product(*index, repeat=2))[0]
        for i in range(0, len(index_combinations) - 1, 2):
            a = index_combinations[i]
            b = index_combinations[i+1]
            cumulative_norm += np.linalg.norm(vectors[a] - vectors[b])
        return cumulative_norm

    def __get_object_identifier_mappings(self, df):
        """
        For a given df, return a mapping from 'new_object_identifier' => ['old_object_identifier1', 'old_object_identifier2', ...].
        The key in the mapping is a newly created, unique identifier for the physical object.
        The value in the mapping is a list of old object_identifiers that most probably all refer to the same physical object.

        :param df: input df
        :type: pd.DataFrame
        :rtype: dict
        """
        def setup(df):
            # key: sensor_identifier, value: set of objet_identifiers
            sensors_object_identifiers = defaultdict(set)

            # key: object_identifier, value: sensor_identifier
            object_identifiers_sensors = dict()

            # Init the two dicts from above
            for sensor_identifier in df.sensor_identifier.unique():
                cluster_object_identifiers = df[df.sensor_identifier == sensor_identifier].index.get_level_values(
                    'object_identifier').unique()
                for object_identifier in cluster_object_identifiers:
                    object_identifiers_sensors[object_identifier] = sensor_identifier
                    sensors_object_identifiers[sensor_identifier].add(object_identifier)
            return sensors_object_identifiers, object_identifiers_sensors

        sensors_object_identifiers, object_identifiers_sensors = setup(df)
        mappings = {}

        # While there are still unmapped object_identifiers, do:
        while sensors_object_identifiers:
            # Build cartesian product between object_identifiers of all sensors
            object_identifier_combinations = itertools.product(*sensors_object_identifiers.values())

            # Store the a 'score' value for all possible clusters
            cluster_scores = dict()
            for object_identifier_tuple in object_identifier_combinations:
                vectors = self.__get_vectors_by_object_identifiers(list(object_identifier_tuple), df)
                key = '.'.join(list(object_identifier_tuple))
                cluster_scores[key] = self.__cumulative_norm(vectors)

            # The best combination is the one with the lowest cumulative euclidean distance
            best_cluster = min(cluster_scores, key=cluster_scores.get)

            # Split the key to retrieve object_identifiers again.
            object_identifier_cluster = best_cluster.split('.')

            # Save the calculated cluster
            cluster_uuid = str(uuid.uuid4())
            mappings[cluster_uuid] = object_identifier_cluster

            # Remove the calculated cluster from the set of possible combinations.
            for object_identifier in object_identifier_cluster:
                sensors_object_identifiers[object_identifiers_sensors[object_identifier]].remove(
                    object_identifier)
                if not sensors_object_identifiers[object_identifiers_sensors[object_identifier]]:
                    del sensors_object_identifiers[object_identifiers_sensors[object_identifier]]
                del object_identifiers_sensors[object_identifier]
        return mappings

    def run(self):
        """
        Run object_matching and re-index object_identifiers

        :returns: The re-indexed input_batch_df
        :rtype: pd.DataFrame
        """
        df = self.__input_batch_df
        all = set(df.index.get_level_values('object_identifier').unique())
        seen = set()
        mapping_list = []

        # Iterate over time windows that contain the highest number unique object_identifiers counts
        for timestamp in self.__get_top_timestamps(df):
            mask = df.index.get_level_values('timestamp') == timestamp
            time_window_df = df.loc[mask, :]

            # Get mapping for defined time_window
            mapping = self.__get_object_identifier_mappings(time_window_df)
            mapping_list.append(mapping)

            # Update which object_identifiers were part of the mapping
            seen.update([item for sublist in mapping.values() for item in sublist])

            # As soon as we have seen all object_identifiers, terminate the process.
            if seen == all:
                break

        self.__clusters = self.__filter_mappings(mapping_list)

        # Finally, replace object_identifier with computed cluster_uuid
        for cluster_uuid, object_identifier_cluster in self.__clusters.items():
            for external_object_identifier in object_identifier_cluster:
                mask = self.__input_batch_df.index.get_level_values('object_identifier') == external_object_identifier
                self.__input_batch_df.loc[mask, 'object_identifier'] = cluster_uuid
        return self.__input_batch_df

    def __filter_mappings(self, mapping_list):
        """
        Filter generated mappings.

        :param mapping_list: list of mappings
        :type mapping_list: list
        :rtype: dict
        """
        # Remove duplicate clusters that were captured in multiple time windows
        final_mapping = {}
        for mapping in mapping_list:
            for cluster_uuid, object_identifier_cluster in mapping.items():
                object_identifier_cluster = set(object_identifier_cluster)
                if object_identifier_cluster in final_mapping.values():
                    continue
                final_mapping[cluster_uuid] = object_identifier_cluster

        # Remove object_identifiers that are part in more than 1 cluster.
        # Sort clusters by length. We consider large clusters more important than small ones.
        cluster_uuids = sorted(final_mapping, key=lambda k: len(final_mapping[k]), reverse=True)
        seen = set()
        for cluster_uuid in cluster_uuids:
            object_identifiers = final_mapping[cluster_uuid]
            object_identifiers = object_identifiers.difference(seen)
            final_mapping[cluster_uuid] = object_identifiers
            seen.update(object_identifiers)

        # At the end, we may be left with empty clusters, filter them and convert to list.
        return {k: list(v) for k, v in final_mapping.items() if v}

    def __get_top_timestamps(self, df):
        """
        Find the time windows for which a high number of unique object_identifiers was recorded.
        Returns a list of timestamps order by count of unique object_identifiers, recorded in this timestamp.

        :param df: input batch df
        :type df: pd.DataFrame
        :rtype: pd.Series
        """

        def count_unique_object_identifiers(x):
            return x.index.get_level_values('object_identifier').unique().size

        return df \
            .groupby([pd.Grouper(freq='1s', level='timestamp')]) \
            .apply(count_unique_object_identifiers).sort_values(ascending=False) \
            .index

    def get_clusters(self):
        """
        Get computed clusters

        :rtype: dict
        """
        return self.__clusters
