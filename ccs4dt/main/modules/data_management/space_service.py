class SpaceService:
    def __init__(self, core_db):
        self.__core_db = core_db

    def create(self, data):
        space_id = self.__core_db.space_table.insert(data)
        return self.get_by_id(space_id)

    def get_by_id(self, space_id):
        return {
            'id': space_id,
            **dict(self.__core_db.space_table.get(doc_id=space_id))
        }

    def get_all(self):
        return [self.get_by_id(space.doc_id) for space in self.__core_db.space_table.all()]

    def update(self, space_id, data):
        self.__core_db.space_table.update(data, doc_ids=[space_id])
        return self.get_by_id(space_id)

    def delete(self, space_id):
        self.__core_db.space_table.remove(doc_ids=[space_id])
