import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class search_doc:

    def __init__(self,vectorstore:FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_search()

    def setup_search(self):

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # # BM25检索器
        self.tfidf_retriever = TFIDFRetriever.from_documents(
            self.chunks,
            k=3
        )
        # self.tfidf = TfidfVectorizer()


        print("检索器初始化完成")

    def metadata_filter(self,query:str):

        matching_chunks = [
            chunk for chunk in self.chunks if chunk.metadata.get("type", "").lower() in query.lower()
        ]
        if not matching_chunks:
            return None
        else:
            self.tfidf_retriever = TFIDFRetriever.from_documents(
                matching_chunks,
                k=3
            )




    def search(self,query:str,top_k:int = 3)->List[Document]:

        self.metadata_filter(query)
        bm25_docs = self.tfidf_retriever.get_relevant_documents(query)
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        final_docs = bm25_docs[:top_k] + vector_docs[:top_k]

        return final_docs
