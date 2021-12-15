class OutputBatchService:
    def __init__(self, core_db):
        self.__core_db = core_db

    def create(self, input_batch_id):
        output_batch_id = self.__core_db.output_batch_table.insert({
            "input_batch_id": input_batch_id,
            "positions": []
        })

        return self.get_by_id(output_batch_id)

    def get_by_id(self, output_batch_id):
        return {
            'id': output_batch_id,
            **dict(self.__core_db.output_batch_table.get(doc_id=output_batch_id))
        }
