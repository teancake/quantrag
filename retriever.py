#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from loguru import logger
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# from langchain.retrievers.multi_query import MultiQueryRetriever


from langchain.globals import set_debug, set_verbose

set_debug(False)
set_verbose(False)

from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate, AIMessagePromptTemplate,
                                    MessagesPlaceholder
                                    )

from utils.config_util import get_rag_config
from rag_base import RagBase


class QuantRagRetriever(RagBase):
    def __init__(self):
        super().__init__()
        os.environ["VLLM_USE_MODELSCOPE"] = "True"
        self.llm = ChatOpenAI(model="qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
                              openai_api_key="EMPTY",
                              openai_api_base="http://localhost:8000/v1",
                              stop=["<|im_end|>"])
        os.environ["VLLM_USE_MODELSCOPE"] = ""
        openai_conf, _ = get_rag_config()
        self.llm_openai = ChatOpenAI(model_name="gpt-3.5-turbo-0125",
                                     openai_api_key=openai_conf["openai_api_key"],
                                     openai_api_base=openai_conf["openai_api_base"]
                                     )
        #        self.llm = self.llm_openai
        self.chat_chain = self.get_chat_chain()
        logger.info(f"object initialized. chat chain {self.chat_chain}")

    def get_chat_chain(self):
        # use openai client for openai-compatible interfaces
        retriever = self.vdb.as_retriever(search_kwargs={"k": 5})
        # mq_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
        full_chat_prompt = self.get_chat_prompt()
        chat_chain = {
                         "context": itemgetter("query") | retriever,
                         "query": itemgetter("query"),
                         "chat_history": itemgetter("chat_history"),
                     } | full_chat_prompt | self.llm

        return chat_chain

    def get_chat_prompt(self):
        system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
        user_prompt = HumanMessagePromptTemplate.from_template("""
        Answer the question based only on the following context:
        {context}
        Question: {query}
        """)
        full_chat_prompt = ChatPromptTemplate.from_messages([system_prompt,
                                                             MessagesPlaceholder(variable_name="chat_history"),
                                                             user_prompt])
        logger.info(f"full_chat_prompt is {full_chat_prompt}")
        return full_chat_prompt

    def query(self):
        query = input("query:")
        logger.info(f"query {query}")
        response = self.chat_chain.invoke({"query": query, "chat_history": []})
        logger.info(f"query: {query}, response {response}")
        return response.content

    def endless_query(self):
        logger.info("chat")
        chat_history = []
        while True:
            query = input("query:")
            logger.info(f"query {query}")
            response = self.chat_chain.invoke({"query": query, "chat_history": chat_history})
            logger.info(f"response {response}")
            chat_history.extend((HumanMessage(content=query), response))
            print(response.content)
            chat_history = chat_history[-20:]


if __name__ == "__main__":
    ret = QuantRagRetriever()
    logger.info("start chat")
    ret.endless_query()

