import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.join(os.path.dirname(__file__))))

from mini_graphrag.graphrag import GraphRAG
from mini_graphrag.corpus_loader import CorpusLoader
def main():
    from dotenv import load_dotenv
    load_dotenv()
    corpus = CorpusLoader('./pdf/huawei.pdf').load()
    graphrag = GraphRAG()
    graphrag.insert(corpus)

if __name__ == '__main__':
    main()