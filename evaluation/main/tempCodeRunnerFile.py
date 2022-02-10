print("Explanation: \n \
        'idf1': identification measure - global min-cost F1 score, \n \
        'idp' identification measure - global min-cost precision, \n \
        'idr' identification measure - global min-cost recall,\n \
        'recall' Number of detections over number of objects,\n\
        'precision' Number of detected objects over sum of detected and false positives.,\n\
        GT 'num_unique_objects' Total number of unique object ids encountered.,\n\
        MT 'mostly_tracked' Number of objects tracked for at least 80 percent of lifespan.,\n\
        PT 'partially_tracked' Number of objects tracked between 20 and 80 percent of lifespan.,\n\
        ML 'mostly_lost' Number of objects tracked less than 20 percent of lifespan.,\n\
        FP 'num_false_positives',\n\
        FN 'num_misses' Total number of misses.,\n\
        IDs 'num_switches' Total number of detected objects including matches and switches.,\n\
        FM 'num_fragmentations' Total number of switches from tracked to not tracked.,\n\
        'mota' Multiple object tracker precision.,\n\
        'motp' Multiple object tracker accuracy.,\n\
        IDt 'num_transfer' Total number of track transfer.,\n\
        IDa 'num_ascend' Total number of track ascend.,\n\
        IDm 'num_migrate' Total number of track migrate.")