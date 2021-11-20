import numpy as np
import pandas as pd
import uuid
from collections import defaultdict
import itertools


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


    def cumulative_norm(self, tuple, df):
        cumulative_norm = 0

        if len(tuple) == 1:
            return cumulative_norm

        object_identifier_combinations = list(itertools.product(*[[i] for i in tuple], repeat=2))[0]
        for i in range(0, int(len(object_identifier_combinations)/2) + 2, 2):
            a_mask = df.index.get_level_values('object_identifier') == object_identifier_combinations[i]
            b_mask = df.index.get_level_values('object_identifier') == object_identifier_combinations[i+1]
            a = df.loc[a_mask,  ['x', 'y', 'z']].to_numpy().squeeze()
            b = df.loc[b_mask, ['x', 'y', 'z']].to_numpy().squeeze()
            cumulative_norm += np.linalg.norm(a-b)
        return cumulative_norm

    def get_object_identifier_mappings(self, df):
        mappings = {}
        object_identifiers_by_sensors = defaultdict(set)
        sensors_by_object_identifiers = dict()
        for sensor_identifier in df.sensor_identifier.unique():
            cluster_object_identifiers = df[df.sensor_identifier == sensor_identifier].index.get_level_values('object_identifier').unique()
            for object_identifier in cluster_object_identifiers:
                sensors_by_object_identifiers[object_identifier] = sensor_identifier
                object_identifiers_by_sensors[sensor_identifier].add(object_identifier)

        while object_identifiers_by_sensors:
            cartesian_product = itertools.product(*object_identifiers_by_sensors.values())
            tuples_score = dict()
            for object_identifier_tuple in cartesian_product:
                key = '.'.join(list(object_identifier_tuple))
                tuples_score[key] = self.cumulative_norm(list(object_identifier_tuple), df)

            best_tuple = min(tuples_score, key=tuples_score.get)
            # Maybe check some condition on the best distance? if it's too high, then maybe different objects?
            # Now we split the key to retrieve object_identifiers again.
            cluster_object_identifiers = best_tuple.split('.')
            cluster_uuid = str(uuid.uuid4())
            mappings[cluster_uuid] = cluster_object_identifiers

            # Remove this combination from the original set
            for object_identifier in cluster_object_identifiers:
                object_identifiers_by_sensors[sensors_by_object_identifiers[object_identifier]].remove(object_identifier)
                if not object_identifiers_by_sensors[sensors_by_object_identifiers[object_identifier]]:
                    del object_identifiers_by_sensors[sensors_by_object_identifiers[object_identifier]]
                del sensors_by_object_identifiers[object_identifier]
        return mappings


    def run(self):
        """
        Run object_matching and re-index object_identifiers

        :returns: The re-indexed input_batch_df
        :rtype: pd.DataFrame
        """
        df = self.__input_batch_df
        top_timestamps = self.__get_top_timestamps(df)
        all = set(df.index.get_level_values('object_identifier').unique())
        seen = set()
        keep = set()
        mapping_list = []

        # Iterate over time windows that contain the most unique object_identifiers
        for timestamp in top_timestamps:
            mask = df.index.get_level_values('timestamp') == timestamp
            filtered_df = df.loc[mask, :]
            mapping = self.get_object_identifier_mappings(filtered_df)
            mapping_list.append(mapping)
            seen.update([item for sublist in mapping.values() for item in sublist])
            if seen == all:
                break


        # Remove duplicate clusters that were captured in multiple time windows
        final_mapping = {}
        for mapping in mapping_list:
            for cluster_uuid, object_identifier_cluster in mapping.items():
                object_identifier_cluster = set(object_identifier_cluster)
                if object_identifier_cluster in final_mapping.values():
                    continue
                final_mapping[cluster_uuid] = object_identifier_cluster

        # Remove object_identifiers that are part in more than 1 cluster.
        seen = set()
        # Sort by length. We consider large clusters more important than small ones.
        cluster_uuids = sorted(final_mapping, key=lambda k: len(final_mapping[k]), reverse=True)
        for cluster_uuid in cluster_uuids:
            object_identifiers = final_mapping[cluster_uuid]
            object_identifiers = object_identifiers.difference(seen)
            final_mapping[cluster_uuid] = object_identifiers
            seen.update(object_identifiers)

        # Filter empty clusters and convert to list.
        self.__clusters = {k: list(v) for k, v in final_mapping.items() if v}

        # Finally, replace object_identifier with computed cluster_uuid
        for cluster_uuid, object_identifier_cluster in self.__clusters.items():
            for external_object_identifier in object_identifier_cluster:
                mask = self.__input_batch_df.index.get_level_values('object_identifier') == external_object_identifier
                self.__input_batch_df.loc[mask, 'object_identifier'] = cluster_uuid
        return self.__input_batch_df


    def __get_top_timestamps(self, df):
        def count_unique_object_identifiers(x):
            return x.index.get_level_values('object_identifier').unique().size

        return df.groupby([pd.Grouper(freq='1s', level='timestamp')])\
            .apply(count_unique_object_identifiers).sort_values(ascending=False)\
            .index

    def get_clusters(self):
        return self.__clusters