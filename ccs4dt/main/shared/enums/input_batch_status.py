from enum import Enum

class InputBatchStatus(str, Enum):
    SCHEDULED = 'scheduled'
    PROCESSING = 'processing'
    FINISHED = 'finished'
    FAILED = 'failed'