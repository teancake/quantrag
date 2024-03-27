import sys, os
from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pymilvus import (
    connections, list_collections,
    Collection,
    FieldSchema, CollectionSchema, DataType
)
# note connections is a singleton managing all connections


from utils.starrocks_db_util import StarrocksDbUtil
from rag_base import RagBase

from utils.redis_util import RedisUtil


class QuantRagIndexer(RagBase):
    red = RedisUtil()

    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter()

    def add_collection(self):
        ## add collection
        ## not sure how langchain milvus implemented this, thus use lower-level apis

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="pub_time", dtype=DataType.VARCHAR, max_length=25),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]

        logger.info(f"Create collection...")
        schema = CollectionSchema(fields=fields, description="content embeddings")
        collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"List collections: {list_collections()}")

        logger.info("Add vector index")
        vector_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
        collection.create_index(field_name="vector", index_name="idx_vector", index_params=vector_index_params)
        logger.info("Add scalar index")

        scalar_index_params = {"index_type": "Trie"}
        collection.create_index(field_name="pub_time", index_name="idx_pub_time", index_params=scalar_index_params)
        collection.create_index(field_name="source", index_name="idx_source", index_params=scalar_index_params)

    ## insert data
    def insert_contents(self, ds):
        recs = StarrocksDbUtil().run_sql(f"select * from dwd_quant_rag_di where ds='{ds}'")
        docs = []
        ids = []
        for rec in recs:
            content = rec[5]
            id_ = rec[3]
            source = rec[2]
            pub_time = rec[4]
            metadata = {
                "id": id_,
                "source": source,
                "type": "",
                "pub_time": pub_time
            }
            if not self.red.record_exists(id_) and id_ not in ids:
                docs.append(Document(page_content=content, metadata=metadata))
                ids.append(metadata["id"])
        logger.info(f"{len(recs)} records loaded from database, {len(docs)} records will be inserted into milvus.")
        if len(docs) > 0:
            batch_size = 256
            logger.info(f"len docs {len(docs)}, len ids {len(set(ids))}, batch size {batch_size}")
            for i in range(0, len(docs), batch_size):
                end = min(i + batch_size, len(docs))
                ret = self.vdb.add_documents(docs[i: end], ids=ids[i: end])
                logger.info(f"{len(ret)} records inserted to milvus, starting from {i}")
            self.red.set_records(ids)

    def similarity_search(self, query):
        return self.vdb.similarity_search(query)


if __name__ == "__main__":
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    indexer = QuantRagIndexer()
    indexer.add_collection()
    indexer.insert_contents(ds)
