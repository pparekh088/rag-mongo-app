import os

import certifi

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pymongo import MongoClient
import logging
# Set DB
if os.environ.get("MONGO_URI", None) is None:
    raise Exception("Missing `MONGO_URI` environment variable.")
MONGO_URI = os.environ["MONGO_URI"]

DB_NAME = "langchain-test-2"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
logging.warning(MONGO_URI)
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]
logging.warning('At MONGODB_COLLECTION')
# Read from MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
retriever = vectorstore.as_retriever()
logging.warning("retriever : {retriever}")
# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
logging.warning(template)
prompt = ChatPromptTemplate.from_template(template)

# RAG
logging.warning(prompt)

model = ChatOpenAI()
logging.warning(prompt)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    logging.warning('in')
    __root__: str


chain = chain.with_types(input_type=Question)


def _ingest(url: str) -> dict:
    loader = PyPDFLoader(url)
    data = loader.load()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Insert the documents in MongoDB Atlas Vector Search
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    return {}

logging.warning('at the end of chain')
ingest = RunnableLambda(_ingest)
