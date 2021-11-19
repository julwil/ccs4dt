class ObjectMatchingService:
    """
    Handle mapping between object_identifiers and external_object_identifiers.
    object_identifiers are generated in the object_matching module. external_object_identifiers are
    the assigned externally by the sensors.

    :param core_db: database connection of core_db
    :type core_db: CoreDB
    """

    def __init__(self, core_db):
        self.__core_db = core_db

    def create(self, input_batch_id, object_identifier, external_object_identifier):
        """
        Store a new relationship.

        :param input_batch_id: id of the input batch
        :type input_batch_id: int
        :param object_identifier: internal identifier
        :type object_identifier: str
        :param external_object_identifier: external identifier
        :type external_object_identifier: str
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''
        INSERT INTO object_identifier_matches 
        (input_batch_id, object_identifier, external_object_identifier)
        VALUES (?, ?, ?)
        '''
        id = connection.cursor().execute(query, (input_batch_id, object_identifier, external_object_identifier)).lastrowid
        connection.commit()
        return self.get_by_id(id)


    def get_by_id(self, id):
        """
        Get matching by id

        :param id: id of the matching
        :type id: int
        :rtype: dict
        """
        connection = self.__core_db.connection()
        query = '''SELECT * FROM object_identifier_matches WHERE id=?'''
        return dict(connection.cursor().execute(query, (id,)).fetchone())


    def get_by_input_batch_id(self, input_batch_id):
        """
        Get all object_identifier mappings by input_batch_id

        :param input_batch_id: id of the input batch
        :type input_batch_id: int
        :rtype: list
        """
        connection = self.__core_db.connection()
        query = '''SELECT * FROM object_identifier_matches WHERE input_batch_id=?'''
        ids = [dict(object_matching)['id'] for object_matching in connection.cursor().execute(query, (input_batch_id,)).fetchall()]
        return [self.get_by_id(id) for id in ids]
