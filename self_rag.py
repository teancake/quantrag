#!/usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(False)


from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict

from retriever import QuantRagRetriever



class GraphState(TypedDict):
    keys: Dict[str, any]

class Grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Useful score 'yes' or 'no'")


class QuantSelfRag:
    def __init__(self):
        rag = QuantRagRetriever()
        vectorstore = rag.vdb
        self.retriever = vectorstore.as_retriever()
        self.llm = rag.llm_openai
        self.generation_prompt, self.grade_prompt, self.query_reform_prompt, self.generation_v_question_prompt, self.generation_v_document_prompt = self.get_prompts()
        self.model, self.llm_with_tool, self.parser_tool = self.get_llm_and_parser_tool()


    def get_prompts(self):
        generation_prompt = hub.pull("rlm/rag-prompt")
        logger.info(f"pulling prompts from langchain hub, prompt {generation_prompt}")

        grade_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: {context}
            \n ------- \n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        query_reform_prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for retrieval. \n 
                Look at the input and try to reason about the underlying sematic intent / meaning. \n 
                Here is the initial question: {question} \n
                Formulate an improved question: """,
            input_variables=["question"],
        )

        generation_v_question_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
              Here is the answer: {generation} 
              \n ------- \n
              Here is the question: {question} \n
              Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.""",
            input_variables=["generation", "question"],
        )

        generation_v_document_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
            Here are the facts:  {documents} 
            \n ------- \n
            Here is the answer: {generation} \n
            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts.""",
            input_variables=["generation", "documents"],
        )
        return generation_prompt, grade_prompt, query_reform_prompt, generation_v_question_prompt, generation_v_document_prompt


    def get_llm_and_parser_tool(self):
        model = self.llm
        grade_tool_oai = convert_to_openai_tool(Grade)
        llm_with_tool = model.bind(
            tools=[grade_tool_oai],
            tool_choice={"type": "function", "function": {"name": "Grade"}},
        )
        parser_tool = PydanticToolsParser(tools=[Grade]) | self.get_grade
        return model, llm_with_tool, parser_tool

    def get_grade(self, resp):
        return resp[0].binary_score

    def state_to_variables(self, state):
        state_dict = state["keys"]
        question = state_dict.get("question")
        documents = state_dict.get("documents")
        generation = state_dict.get("generation")
        return question, documents, generation

    def variables_to_state(self, question=None, documents=None, generation=None):
        return {"keys": {"documents": documents, "question": question, "generation": generation}}

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    ### Nodes ###
    def node_retrieve(self, state):
        logger.info("---RETRIEVE---")
        question, documents, generation = self.state_to_variables(state)
        documents = self.retriever.get_relevant_documents(question)
        return self.variables_to_state(question=question, documents=documents)


    def node_grade_documents(self, state):
        logger.info("---CHECK RELEVANCE---")
        question, documents, generation = self.state_to_variables(state)
        chain = self.grade_prompt | self.llm_with_tool | self.parser_tool

        filtered_docs = []
        for d in documents:
            logger.info(f"invoking grade document for document {d}")
            grade = chain.invoke({"question": question, "context": d.page_content})
            if grade == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return self.variables_to_state(question=question, documents=filtered_docs)
    
    def node_generate(self, state):
        logger.info("---GENERATE---")
        question, documents, generation = self.state_to_variables(state)
        rag_chain = self.generation_prompt | self.model | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return self.variables_to_state(question=question, documents=documents, generation=generation)

    def node_transform_query(self, state):
        logger.info("---TRANSFORM QUERY---")
        question, documents, generation = self.state_to_variables(state)
        chain = self.query_reform_prompt | self.model | StrOutputParser()
        better_question = chain.invoke({"question": question})
        return self.variables_to_state(question=better_question, documents=documents)

    def node_prepare_for_final_grade(self, state):
        logger.info("---FINAL GRADE---")
        question, documents, generation = self.state_to_variables(state)
        return self.variables_to_state(question=question, documents=documents, generation=generation)


    ### Routers ###
    def router_decide_to_generate(self, state):
        logger.info("---DECIDE TO GENERATE---")
        question, documents, generation = self.state_to_variables(state)
        if not documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            logger.info("---DECISION: TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            logger.info("---DECISION: GENERATE---")
            return "generate"

    def router_grade_generation_v_documents(self, state):
        logger.info("---GRADE GENERATION vs DOCUMENTS---")
        question, documents, generation = self.state_to_variables(state)
        chain = self.generation_v_document_prompt | self.llm_with_tool | self.parser_tool
        grade = chain.invoke({"generation": generation, "documents": documents})

        if grade == "yes":
            logger.info("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---")
            return "supported"
        else:
            logger.info("---DECISION: NOT SUPPORTED, GENERATE AGAIN---")
            return "not supported"

    def router_grade_generation_v_question(self, state):
        logger.info("---GRADE GENERATION vs QUESTION---")
        question, documents, generation = self.state_to_variables(state)
        chain = self.generation_v_question_prompt | self.llm_with_tool | self.parser_tool
        grade = chain.invoke({"generation": generation, "question": question})

        if grade == "yes":
            logger.info("---DECISION: USEFUL---")
            return "useful"
        else:
            logger.info("---DECISION: NOT USEFUL---")
            return "not useful"


def run_quant_self_rag(query: str):
    workflow = StateGraph(GraphState)
    sr = QuantSelfRag()
    # Define the nodes
    workflow.add_node("retrieve", sr.node_retrieve)  # retrieve
    workflow.add_node("grade_documents", sr.node_grade_documents)  # grade documents
    workflow.add_node("generate", sr.node_generate)  # generatae
    workflow.add_node("transform_query", sr.node_transform_query)  # transform_query
    workflow.add_node("prepare_for_final_grade", sr.node_prepare_for_final_grade)  # passthrough

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        sr.router_decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        sr.router_grade_generation_v_documents,
        {
            "supported": "prepare_for_final_grade",
            "not supported": "generate",
        },
    )
    workflow.add_conditional_edges(
        "prepare_for_final_grade",
        sr.router_grade_generation_v_question,
        {
            "useful": END,
            "not useful": "transform_query",
        },
    )
    app = workflow.compile()
    inputs = {"keys": {"question": f"{query}"}}
    for output in app.stream(inputs):
        for key, value in output.items():
            logger.info(f"Output from node '{key}': {value}")

    # Final generation
    return value["keys"]["generation"]



if __name__ == '__main__':
    #query = "今天有哪些股票跌得很厉害"
    query = input("query: ")
    res = run_quant_self_rag(query)
    logger.info(f"query {query}, result {res}")



