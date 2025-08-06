import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from langchain_core.documents import Document
from neo4j import GraphDatabase
from config import DEFAULT_CONFIG, GraphRAGConfig
import os

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]


class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        """初始化图RAG检索系统"""

        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Neo4j连接成功")
        except Exception as e:
            print(f"Neo4j连接失败: {e}")
            return

        # 预热：构建实体和关系索引
        # self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引以加速查询"""

        try:
            with self.driver.session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }

                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                print(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")

        except Exception as e:
            print(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
           作为图数据库专家，分析以下查询的图结构意图：

           查询：{query}

           请识别：
           1. 查询类型：
              - entity_relation: 询问实体间的直接关系（如：吴老爹在哪里？）
              - multi_hop: 需要多跳推理（如：从长安城到东海湾怎么走？需要：长安城->江南野外->建邺城->东海湾）


           2. 核心实体：查询中的关键实体名称
           3. 目标实体：期望找到的实体类型
           4. 关系类型：涉及的关系类型


           示例：
           查询："从长安城到东海湾怎么走？"
           分析：这是multi_hop查询，需要通过"长安城->江南野外->建邺城->东海湾"的路径推理

           返回JSON格式：
           {{
               "query_type": "multi_hop",
               "source_entities": ["长安城"],
               "target_entities": ["东海湾"],
               "relation_types": ["连接"],
               "reasoning": "需要多跳推理：长安城->江南野外->建邺城->东海湾"
           }}
           """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content.strip())

            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )

        except Exception as e:
            print(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )

    def multi_hop_traversal(self, graph_query: GraphQuery) -> str:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        print(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")

        paths = []

        if not self.driver:
            print("Neo4j连接未建立")
            return paths

        try:
            with self.driver.session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_entities = graph_query.target_entities or []
                # max_depth = graph_query.max_depth

                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type == QueryType.MULTI_HOP:
                    cypher_query = f"""
                MATCH path = shortestPath((start:Location)-[:连接*]-(end:Location))
                WHERE start.name IN $source_entities AND end.name IN $target_entities
                RETURN path, length(path) AS path_length

                       """

                    result = session.run(cypher_query, {
                        "source_entities": source_entities,
                        "target_entities": target_entities,
                        "relation_types": graph_query.relation_types or []
                    })
                    names=[]

                    # 假设 'result' 是你从 Neo4j 查询返回的结果
                    for record in result:
                        # 获取 path 中的所有节点
                        nodes = record['path'].nodes

                        # 遍历每个节点，提取 'name' 属性
                        for node in nodes:
                            names.append(node.get("name"))

                        str = " -> ".join(names)
                        # print(str)

                    # for record in result:
                    #     path_data = self._parse_neo4j_path(record)
                    #     if path_data:
                    #         paths.append(path_data)
                #
                # elif graph_query.query_type == QueryType.ENTITY_RELATION:
                #     # 实体间关系查询
                #     paths.extend(self._find_entity_relations(graph_query, session))
                #
                # elif graph_query.query_type == QueryType.PATH_FINDING:
                #     # 最短路径查找
                #     paths.extend(self._find_shortest_paths(graph_query, session))

        except Exception as e:
            print(f"多跳遍历失败: {e}")

        print(f"多跳遍历完成")
        return str

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> str:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维
        """
        print(f"提取知识子图: {graph_query.source_entities}")

        if not self.driver:
            print("Neo4j连接未建立")

        try:
            with self.driver.session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_entities = graph_query.target_entities or []

                # 根据查询类型选择不同的遍历策略
                cypher_query = """
        MATCH (location:Location)-[:包含]->(npc:NPC)
        WHERE npc.name IN $npc_names
        RETURN location.name AS location_name, npc.name AS npc_name
                       """
                text="""MATCH (location:Location)-[:包含]->(npc:NPC {name: '孙婆婆'})
RETURN location.name AS location_name, npc.name AS npc_name
                """
                result = session.run(cypher_query, {"npc_names": source_entities})
                # result = session.run(text)

                names=[]

                for record in result:
                    location_name = record["location_name"]  # 提取字段值
                    names.append(location_name)

                str = " , ".join(names)

                str = f"包含{source_entities}的地点有：" + str

        except Exception as e:
            print(f"子图提取失败: {e}")

        # 降级方案：简单邻居查询
        return str

    def graph_rag_search(self, query: str, top_k: int = 5) -> str:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        print(f"开始图RAG检索: {query}")

        if not self.driver:
            print("Neo4j连接未建立，返回空结果")
            return '[]'

        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        print(f"查询类型: {graph_query.query_type.value}")


        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type == QueryType.MULTI_HOP:
                # 多跳遍历
                prompt = self.multi_hop_traversal(graph_query)
                print(f"图RAG检索完成，返回相关资料")
                return prompt

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询
                prompt = self.extract_knowledge_subgraph(graph_query)
                print(f"图RAG检索完成，返回相关资料")
                return prompt



        except Exception as e:
            print(f"图RAG检索失败: {e}")
            return '[]'

    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            print("图RAG检索系统已关闭")





# graph_rag_retrieval = GraphRAGRetrieval(
#                 config=config,
#                 llm_client=client
#             )
# graph_rag_retrieval.initialize()
#
# query='怎么从东海湾到长安城'
#
# rel_docs = graph_rag_retrieval.graph_rag_search(query)


# rel_docs = graph_rag_retrieval.multi_hop_traversal(graph_query)
# rel_docs = graph_rag_retrieval.extract_knowledge_subgraph(graph_query)
# print(rel_docs)


# data_dir = r'D:\myrealllm\map'
#
# for filename in os.listdir(data_dir):
#     # 检查文件是否是txt文件
#     if filename.endswith(".txt"):
#         file_path = os.path.join(data_dir, filename)
#
#         # 打开并读取文件内容
#         with open(file_path, 'r', encoding='utf-8') as file:
#             CQL = file.read()  # 直接读取文件内容，不去除前后空格
#
#             with graph_rag_retrieval.driver.session() as session:
#                 # 构建实体索引 - 修复Neo4j语法兼容性问题
#                 entity_query = f"""
#                 {CQL}
#                 """
#                 session.run(entity_query)
#
# # with graph_rag_retrieval.driver.session() as session:
# #                 # 构建实体索引 - 修复Neo4j语法兼容性问题
# #         entity_query = ""
# #     session.run(entity_query)


