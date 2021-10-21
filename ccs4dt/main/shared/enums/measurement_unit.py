from enum import Enum

class MeasurementUnit(str, Enum):
    """Supported sensor measurement units"""

    MILLIMETER = 'mm'
    CENTIMETER = 'cm'
    METER = 'm'