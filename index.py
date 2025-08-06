from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List
from pathlib import Path



class Index_build:

    def __init__(self,model_name:str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):

        self.model_name = model_name
        self.index_save = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embedding()

    def setup_embedding(self):
        self.embeddings =HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs ={'device':'cpu'},
            encode_kwargs ={'normalize_embeddings':True}

        )

    def build_vector(self,chunks:List[Document])->FAISS:
        if not chunks:
            print("文档为空")
        self.vectorstore =FAISS.from_documents(
            documents=chunks,
            embedding = self.embeddings
        )
        print("向量化完成")
        return self.vectorstore

    def save_index(self):
        if not self.vectorstore:
            print("无向量索引")
        Path(self.index_save).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save)
        print("向量索引已保存")

    def load_index(self):
        if not self.embeddings:
            self.setup_embedding()

        if not Path(self.index_save).exists():
            print("向量索引路径不存在")

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("向量索引加载成功")
            return self.vectorstore

        except Exception as e:
            print(e)

    def similarity_search(self,query:str,k:int=5)->List[Document]:

        if not self.vectorstore:
            print("无向量索引")

        return self.vectorstore.similarity_search(query, k=k)