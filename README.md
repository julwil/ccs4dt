# ccs4dt



### Synthetic data generation

#### Requirements
##### Attributes
Necessary attributes for the syntethic data (requirements based on original data structure from livealytics):

| Attribute name     | Description     | Other comments | Example |
|--------------------|-----------|------------|------------|
| epc       | device identifier      | e.g. MAC-adress or randomly generated, unique identifier        |        |
| lastSeenTime            | timestamp |TODO: clarify unit of measurement (is too long for UNIX timestamp) | ??? |
| type            | type of entry (new device, update of position)  |        | 'entry','update' |
| xCm            | x-Coordinate of position, relative to readerDevice position  | TODO: clarify unit of measurement | -605 |
| yCm            | y-Coordinate of position, relative to readerDevice position  | TODO: clarify unit of measurement | 37 |
| confidenceWeight            | confidence of measurement  | TODO: Clarify exact meaning of this| '1', '10577' |
| confidenceData            |   | TODO: Clarify purpose and logic behind this field | '[130,1,131,1,132,1,133,1,134,47172,135,1,136,1,137,1,138,0]' |
| filialeId            | Identifier for measurement location  |  | 'schlieren' |
| readerName            | Identifier for readerDevice  |  | 'left', 'central', 'A001' |
| readerDeviceType            | Category of reader device  |  | 'WiFi', 'Camera', 'LIDAR' |