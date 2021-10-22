from enum import Enum

class InputBatchStatus(str, Enum):
    """Status an input batch process can have"""

    SCHEDULED = 'scheduled'
    PROCESSING = 'processing'
    FINISHED = 'finished'
    FAILED = 'failed'