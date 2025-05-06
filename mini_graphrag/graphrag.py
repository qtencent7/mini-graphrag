import os 
import json
import hashlib # 新增导入 hashlib 模块
from openai import OpenAI


class GraphRAG:
    """
    初始化包括两块内容：
    1，加载各种配置参数
    2，加载各种功能进来：
        2.1，加载openai客户端
        2.2，加载语料库进内存
        2
    """
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, working_dir: str = "graphrag"):
        self.chunk_size = chunk_size # 预料块大小
        self.overlap = overlap # 块重叠大小
        self.working_dir = working_dir # 集合名称
        self.corpus_collection = working_dir + "_corpus.json" # 语料库集合名称, 它是 .json 文件
        self.corpus = [] # 初始化 self.corpus 为一个空列表
        self._loadConfig() # 在初始化时调用加载配置


    def _loadConfig(self):
        """
        1，加载openai客户端
        """

        self.llm_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_URL"),
        )
        
        """
        2，加载各种工作目录
        """
        # 确保工作目录存在，如果不存在则创建
        if not os.path.exists(self.working_dir):
            try:
                os.makedirs(self.working_dir)
                print(f"工作目录 {self.working_dir} 已创建。")
            except OSError as e:
                print(f"创建工作目录 {self.working_dir} 失败: {e}")
                # 根据需要，这里可以抛出异常或进行其他错误处理
                return # 如果目录创建失败，可能后续操作无法进行


        """
        3，加载语料库进内存
        """
        # 构建语料库文件的完整路径
        corpus_file_path = os.path.join(self.working_dir, self.corpus_collection)

        # 检查文件是否存在
        if os.path.exists(corpus_file_path):
            try:
                with open(corpus_file_path, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f) # 从JSON文件加载数据
            except json.JSONDecodeError:
                print(f"警告: 文件 {corpus_file_path} 不是有效的JSON格式。self.corpus 将保持为空列表。")
                self.corpus = [] # 如果JSON解析失败，则设置为空列表
            except Exception as e:
                print(f"警告: 加载文件 {corpus_file_path} 时发生错误: {e}。self.corpus 将保持为空列表。")
                self.corpus = [] # 其他异常也设置为空列表
        else:
            print(f"警告: 语料库文件 {corpus_file_path} 未找到。self.corpus 将保持为空列表。")
            self.corpus = [] # 如果文件不存在，则设置为空列表
        
    """
        保存语料库到文件
    """
    def _saveCorpus(self, contentKey: str, content: str):
        self.corpus.append({
            "contentKey": contentKey,
            "content": content
        })
        # 构建语料库文件的完整路径
        corpus_file_path = os.path.join(self.working_dir, self.corpus_collection)

        # 尝试写入文件
        try:
            with open(corpus_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=4)
            print(f"语料库已成功保存到 {corpus_file_path}")
        except Exception as e:
            print(f"保存语料库到 {corpus_file_path} 时发生错误: {e}")

    def insert(self, corpus_content: str):
        """
        把corpus_content插入语料库
        """
        # 1. 为输入内容生成指纹
        hasher = hashlib.sha256()
        hasher.update(corpus_content.encode('utf-8'))
        content_fingerprint = hasher.hexdigest()

        # 2. 如果指纹不存在或我们选择继续处理已存在的内容，则保存（如果不存在）
        # 检查是否真的需要保存（即内容是否是全新的）
        is_new_content = True
        if isinstance(self.corpus, list):
            for item in self.corpus:
                if isinstance(item, dict) and item.get("contentKey") == content_fingerprint:
                    is_new_content = False
                    break
        
        if is_new_content:
            print(f"内容指纹 '{content_fingerprint}' 不存在，正在添加到语料库...")
            self._saveCorpus(contentKey=content_fingerprint, content=corpus_content)
        else:
            print(f"内容指纹 '{content_fingerprint}' 已存在，跳过保存步骤。")


        """
        将corpus_content 切分成chunks
        """
        chunks = self._split_into_chunks(corpus_content)
        print(f"内容已切分为 {len(chunks)} 个块:")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: \"{chunk}\"")

        # 对每个文本块进行关系提取
        extracted_relations = []
        for i, chunk in enumerate(chunks):
            print(f"正在处理块 {i+1}/{len(chunks)}...")
            relations = self._extract_relations_from_chunk(chunk)
            if relations:
                extracted_relations.extend(relations)
                print(f"  从块 {i+1} 中提取了 {len(relations)} 个关系")
            else:
                print(f"  块 {i+1} 中未找到关系")
        
        # 如果提取到了关系，则保存到Neo4j
        if extracted_relations:
            self._save_relations_to_neo4j(extracted_relations, content_fingerprint)
            print(f"已将 {len(extracted_relations)} 个关系保存到Neo4j数据库")
        else:
            print("未从文本中提取到任何关系")

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        将文本按指定的块大小和重叠大小切分成块。
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 计算下一个块的起始位置
            # 如果这是最后一个可能的块，或者剩余文本不足以形成一个有意义的重叠块，则停止
            if end >= text_len:
                break
            
            next_start = start + self.chunk_size - self.overlap
            
            # 如果 next_start 与 start 相同（例如 overlap >= chunk_size），
            # 或者 next_start 超出文本长度，则需要调整或停止以避免死循环或无效操作。
            # 简单处理：如果 next_start 没有前进，则强制前进一点，或者直接结束。
            # 这里我们选择如果 next_start <= start，则说明步进有问题，直接结束循环。
            if next_start <= start:
                 # 如果重叠导致无法前进，则移动一个字符，避免死循环
                 # 或者，如果 chunk_size <= overlap，这会导致问题
                 # 更稳健的做法是确保 step > 0
                step = self.chunk_size - self.overlap
                if step <= 0: # 防止死循环或无效的重叠
                    print(f"警告: chunk_size ({self.chunk_size}) 小于或等于 overlap ({self.overlap})。将以 chunk_size 作为步长。")
                    start += self.chunk_size 
                    if start >= end : # 确保start确实前进了
                        break
                else:
                    start = next_start

            else:
                start = next_start
                
        return chunks

    """
        借用大语言模型对chunk进行关系提取
    """
    def _extract_relations_from_chunk(self, chunk: str) -> list[dict]:
        """
        使用大语言模型从文本块中提取实体关系
        
        返回格式为：
        [
            {
                "source": "实体1",
                "relation": "关系类型",
                "target": "实体2"
            },
            ...
        ]
        """
        try:
            # 构建提示词，要求模型提取关系
            prompt = f"""
            请分析以下文本，提取其中的实体及实体之间的关系。
            以JSON格式返回结果，格式为：
            [
                {{
                    "source": "实体1",
                    "relation": "关系类型",
                    "target": "实体2"
                }},
                ...
            ]
            
            如果没有找到任何实体关系，请返回空数组 []。
            
            文本内容：
            {chunk}
            """
            
            # 调用大语言模型
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat", # 根据实际使用的模型调整
                messages=[
                    {"role": "system", "content": "你是一个专业的知识图谱构建助手，擅长从文本中提取实体及其关系。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # 低温度以获得更确定性的结果
                max_tokens=2000
            )
            
            # 解析响应
            result_text = response.choices[0].message.content
            
            # 尝试从响应中提取JSON部分
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
            else:
                # 如果没有找到JSON数组格式，尝试直接解析整个响应
                result_json = json.loads(result_text)
            
            # 验证结果格式
            if not isinstance(result_json, list):
                print(f"警告: 模型返回的不是列表格式: {result_json}")
                return []
            
            # 验证每个关系项的格式
            valid_relations = []
            for item in result_json:
                if isinstance(item, dict) and "source" in item and "relation" in item and "target" in item:
                    valid_relations.append(item)
                else:
                    print(f"警告: 跳过格式不正确的关系项: {item}")
            
            return valid_relations
            
        except json.JSONDecodeError as e:
            print(f"解析模型返回的JSON时出错: {e}")
            print(f"模型返回的原始文本: {result_text}")
            return []
        except Exception as e:
            print(f"从文本块提取关系时发生错误: {e}")
            return []
    
    """
        将提取到的关系存放到neo4j图数据库中
    """
    def _save_relations_to_neo4j(self, relations: list[dict], content_fingerprint: str):
        """
        将提取的关系保存到Neo4j图数据库
        
        参数:
        - relations: 关系列表，每个关系是一个字典，包含source, relation, target
        - content_fingerprint: 内容的唯一标识符，用于关联关系来源
        """
        # 这里应该实现Neo4j连接和数据存储逻辑
        # 由于这需要Neo4j的依赖和配置，这里提供一个示例框架
        
        print("注意: Neo4j存储功能尚未完全实现。")
        print(f"将要存储的关系: {relations}")
        
        # Neo4j连接示例（需要安装neo4j库: pip install neo4j）
        from neo4j import GraphDatabase
        
        # 从环境变量或配置中获取Neo4j连接信息
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            # 创建Neo4j驱动
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # 使用会话执行事务
            with driver.session() as session:
                # 为每个关系创建或更新节点和关系
                for relation in relations:
                    # 使用MERGE确保不创建重复节点
                    result = session.run('''
                        MERGE (source:Entity {name: $source_name})
                        MERGE (target:Entity {name: $target_name})
                        MERGE (source)-[r:RELATES {type: $relation_type, source_doc: $doc_id}]->(target)
                        RETURN source, r, target
                    ''', {
                        'source_name': relation['source'],
                        'target_name': relation['target'],
                        'relation_type': relation['relation'],
                        'doc_id': content_fingerprint
                    })
                    
                    # 可以处理结果，例如计数或记录
                    summary = result.consume()
                    print(f"创建了 {summary.counters.nodes_created} 个节点和 {summary.counters.relationships_created} 个关系")
            
            # 关闭驱动
            driver.close()
            print("成功将关系存储到Neo4j")
            
        except Exception as e:
            print(f"连接或存储到Neo4j时发生错误: {e}")
        