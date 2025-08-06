import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class generation:

    def __init__(self,model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        os.environ['MOONSHOT_API_KEY'] = 'sk-Kt1hdu1kqPBYlY66B4wCzm13xD0MRrI5cLclAFxoMyAoRX8H'
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            print("无api-key")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )

        print("加载llm成功")

    def generate_basic_answer(self,query:str,contexts_docs: List[Document]) -> str:

        context = self.build_context(contexts_docs)

        prompt =ChatPromptTemplate.from_template(
"""
你是一个梦幻西游的游戏助手。请根据以下的游戏的资料回答用户问题。

用户问题：{question}

相关游戏资料：{context}

请根据相关游戏资料回答，若无法根据游戏资料回答，则回答不知道。

回答：


"""
        )
        chain = (
            {"question":RunnablePassthrough(),'context':lambda _:context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def build_context(self,docs:List[Document],max_length:int = 200)->str:
        if not docs:
            print("暂无相关内容")
        final_text = []

        for doc in docs:
            final_text.append(doc.page_content)

        return final_text

    def search_routing(self, query: str)->str:

        prompt = ChatPromptTemplate.from_template(
            """
            你是一个梦幻西游的游戏助手。请判断下列的游戏问题属于哪一种类型。
            示例：李婆婆在哪里，如何从长安城到东海湾；这些问题属于图搜索问题，则返回1
            
            其余返回0
            
            用户问题：{question}
            
            回答：

            """
        )
        chain = (prompt
                 | self.llm
                 | StrOutputParser()
                 )

        response = chain.invoke(query)
        return response

    def generate_graph_answer(self,query:str,contexts_docs: str) -> str:

        context = contexts_docs

        prompt =ChatPromptTemplate.from_template(
"""
你是一个梦幻西游的游戏助手。请根据以下的游戏的资料回答用户问题。

用户问题：{question}

相关游戏资料：{context}

请根据相关游戏资料回答，若无法根据游戏资料回答，则回答不知道。

回答：


"""
        )
        chain = (
            {"question":RunnablePassthrough(),'context':lambda _:context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

