#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores.milvus import Milvus

from langchain_community.embeddings.modelscope_hub import ModelScopeEmbeddings

from pymilvus import connections

from utils.config_util import get_rag_config


class RagBase():
    def __init__(self):
        self.emb, self.dim = self.get_emb()
        _, vdb_conf = get_rag_config() 
        vdb_conf["host"] = vdb_conf["server_address"]
        self.collection_name="akshare_rag"
        self.vdb = self.get_vdb(vdb_conf, self.collection_name)

    def get_emb(self) -> (Embeddings, int):
        return ModelScopeEmbeddings(model_id="iic/nlp_corom_sentence-embedding_chinese-base"), 768

    def get_vdb(self, vdb_conf: dict, collection_name) -> VectorStore:
        connections.connect(host=vdb_conf["host"],
                            port=vdb_conf["port"],
                            db_name=vdb_conf["db_name"])
        return Milvus(embedding_function=self.emb, collection_name=collection_name, connection_args=vdb_conf)
