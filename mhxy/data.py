from langchain_core.documents import Document
from typing import List, Dict, Any
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from index import Index_build
from search import search_doc
from gen import generation
import os

class Data:

    def __init__(self,pth:str):
        self.data_pth = pth
        self.documents : List[Document] = []

    def load_documents(self) ->List[Document]:
        print("正在加载数据库数据")
        data_path = Path(self.data_pth)
        documents = []

        for txt in data_path.glob("*.txt"):
            try:
                with open(txt,'r',encoding='utf-8') as f:
                    content = f.read()

                # 创建Document对象
                filename = os.path.splitext(os.path.basename(txt))[0]
                doc = Document(
                    page_content=content,
                    metadata={
                        "type": filename  # 标记为父文档
                    }
                )

                documents.append(doc)


            except Exception as e:
                print("读取文件失败")

        self.documents =documents
        print("成功加载数据库数据")
        return documents

    def txt_split(self) -> List[Document]:

        txt_split = CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10
        )

        all_chunks = []

        for doc in self.documents:
            try:
                chunk_type = doc.metadata["type"]
                txt_chunk = txt_split.split_text(doc.page_content)
                for chunk in txt_chunk:
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()  # 保留原始文档的元数据
                    )
                    all_chunks.append(chunk_doc)





            except Exception as e:
                print(e)

        return all_chunks



#

# my_index = Index_build()
# vecstore = my_index.load_index()
# data = Data(r'D:\myrealllm\doc')
# txt = data.load_documents()
# chunk = data.txt_split()
# if vecstore:
#     query = "刘洪在哪里"
#     my_search = search_doc(vecstore,chunk)
#     rel_doc = my_search.search(query)
#     my_gen = generation()
#     response = my_gen.generate_basic_answer(query, rel_doc)
# else:
#     vecstore = my_index.build_vector(chunk)
#     my_index.save_index()
#     query = "请你详细介绍帮派关卡的奖励"
#     my_search = search_doc(vecstore,chunk)
#     rel_doc = my_search.search(query)
#     my_gen = generation()
#     response = my_gen.generate_basic_answer(query, rel_doc)
#
# print(response)



# print(vecstore)