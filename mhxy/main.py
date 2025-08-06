import os
import sys
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig
import data,index,gen,search,config,graph_rag
from index import Index_build
from search import search_doc
from gen import generation
from data import Data
from graph_rag import GraphRAGRetrieval
from openai import OpenAI

#1.加载向量索引
my_index = Index_build()
vecstore = my_index.load_index()

#2.加载文档
data = Data(r'D:\myrealllm\doc')
txt = data.load_documents()

#3.切分文档，用于稀疏向量匹配
chunk = data.txt_split()
print('切分文档成功')

#4.成功加载llm
my_gen = generation()

#5.加载图搜索引擎
os.environ['MOONSHOT_API_KEY'] = 'sk-Kt1hdu1kqPBYlY66B4wCzm13xD0MRrI5cLclAFxoMyAoRX8H'
api_key = os.getenv("MOONSHOT_API_KEY")
config = DEFAULT_CONFIG
client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
my_graph =  GraphRAGRetrieval(
                config=config,
                llm_client=client
            )
print("成功加载图搜索引擎")

#6.问题回答
print('-'*30)
print('请输入你的问题吧')
print('-'*30)
query = input("请输入你的查询：")

if my_gen.search_routing(query) == '1':
   my_graph.initialize()
   rel_doc =  my_graph.graph_rag_search(query)
   response = my_gen.generate_graph_answer(query, rel_doc)
else:
    if vecstore:
        my_search = search_doc(vecstore,chunk)
        rel_doc = my_search.search(query)
        response = my_gen.generate_basic_answer(query, rel_doc)
    else:
        vecstore = my_index.build_vector(chunk)
        my_index.save_index()
        my_search = search_doc(vecstore,chunk)
        rel_doc = my_search.search(query)
        response = my_gen.generate_basic_answer(query, rel_doc)

print('梦幻小助手：',response)

