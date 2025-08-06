import os
from neo4j import GraphDatabase


data_dir = r'D:\myrealllm\map'

for filename in os.listdir(data_dir):
    # 检查文件是否是txt文件
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)

        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            question = file.read()  # 直接读取文件内容，不去除前后空格

