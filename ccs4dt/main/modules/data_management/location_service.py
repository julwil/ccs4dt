class LocationService:
    def __init__(self, core_db):
        self.__core_db = core_db

    def create(self, data):
        location_id = self.__core_db.location_table.insert(data)
        return self.get_by_id(location_id)

    def get_by_id(self, location_id):
        return {
            'id': location_id,
            **dict(self.__core_db.location_table.get(doc_id=location_id))
        }

    def get_all(self):
        return [self.get_by_id(location.doc_id) for location in self.__core_db.location_table.all()]
