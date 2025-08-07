import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



def generate_basic_answer(self,query:str) -> str:


        prompt =ChatPromptTemplate.from_template(
"""
我想导入一个节点，其连接着两个地方，在这个节点中包含着很多npc，能把其CQL语句写出来吗

示例：
// 创建建邺城节点
CREATE (jyc:Location {name: '建邺城'})
  
// 创建江南野外和东海湾节点
CREATE (jn:Location {name: '江南野外'})
CREATE (dhw:Location {name: '东海湾'})

// 连接建邺城到江南野外和东海湾
CREATE (jyc)-[:连接]->(jn)
CREATE (jyc)-[:连接]->(dhw)

// 创建 NPC 节点并与建邺城连接
CREATE (npc1:NPC {name: '宠物仙子', location: '63,115'})
CREATE (npc2:NPC {name: '陈长寿', location: '219,123'})
CREATE (npc3:NPC {name: '超级巫医', location: '211,98'})
CREATE (npc4:NPC {name: '超级巫医', location: '104,55'})


下面是文本，将其转换为CQL语句

{question}

回答：


"""
        )
        chain = (
            {"question":RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response



