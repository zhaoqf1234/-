import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import re

load_dotenv()



os.environ['DEEPSEEK_API_KEY'] = 'sk-9cbd8b29dc614faf8f083c538771ef1d'

# 提示词模板


data_dir = r'D:\myrealllm\map'
cql_pattern = re.compile(r'```cypher\n(.*?)```', re.DOTALL)

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

for filename in os.listdir(data_dir):
    # 检查文件是否是txt文件
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)

        example = """
// 使用 MERGE 检查并创建 "东海湾" 节点，如果不存在则创建
MERGE (jyc:Location {name: '建邺城'})
  
// 使用 MERGE 检查并创建 "建邺城" 节点，如果不存在则创建
MERGE (jn:Location {name: '江南野外'})
MERGE (dhw:Location {name: '东海湾'})

// 连接建邺城到江南野外和东海湾
CREATE (jyc)-[:连接]->(jn)
CREATE (jyc)-[:连接]->(dhw)

// 创建 NPC 节点并与建邺城连接
CREATE (npc1:NPC {name: '宠物仙子', location: '63,115'})
CREATE (npc2:NPC {name: '陈长寿', location: '219,123'})
CREATE (npc3:NPC {name: '超级巫医', location: '211,98'})
CREATE (npc4:NPC {name: '超级巫医', location: '104,55'})

// 建立建邺城与所有NPC的"包含"关系
CREATE (jyc)-[:包含]->(npc1)
CREATE (jyc)-[:包含]->(npc2)
CREATE (jyc)-[:包含]->(npc3)
CREATE (jyc)-[:包含]->(npc4)


"""


        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            question = file.read()  # 直接读取文件内容，不去除前后空格
            prompt = f"""
             我想导入一个节点，其连接着两个地方，在这个节点中包含着很多npc，能把其CQL语句写出来吗

             示例：{example}

             下面是文本，将其转换为CQL语句,其中的关系包含连接包含两种

             {question}

             回答：


             """
            answer = llm.invoke(prompt)
            print(answer)
            cql_code = cql_pattern.search(answer.content)
            if cql_code:
                cql_code_str = cql_code.group(1)

                # 将 CQL 语句保存到文件
                output_filename = f"CQL_{filename}"
                output_file_path = os.path.join(data_dir, output_filename)

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cql_code_str)

                print(f"Saved CQL to {output_filename}")
            else:
                print(f"No CQL code found in {filename}")


