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

import hashlib


def get_sha256sum(content):
    return str(hashlib.sha256(content.encode("utf-8")).hexdigest())



class QuantRagIndexer(RagBase):

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
        recs = StarrocksDbUtil().run_sql(f"select * from dwd_stock_telegraph_cls_di where ds='{ds}'")
        docs = []
        ids = []
        for rec in recs:
            title = rec[2]
            content = rec[3]
            text_type = "content"
            metadata = {
                "id": get_sha256sum(content),
                "source": "telegraph_cls",
                "type": "content",
                "pub_time": rec[5].strftime("%Y-%m-%d %H:%M:%S")
            }
            docs.append(Document(page_content=content, metadata=metadata))
            ids.append(metadata["id"])
        logger.info(f"{len(docs)} records loaded from database.") 
        if len(docs) > 0:
            ret = self.vdb.add_documents(docs, ids=ids)
            logger.info(f"{len(ret)} records inserted to milvus")

    def similarity_search(self, query):
        return self.vdb.similarity_search(query)


if __name__ == "__main__":
    ds = sys.argv[1]
    logger.info("execute task on ds {}".format(ds))
    indexer = QuantRagIndexer()
    indexer.add_collection()
    indexer.insert_contents(ds)

