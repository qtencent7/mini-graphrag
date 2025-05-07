import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.join(os.path.dirname(__file__))))

from mini_graphrag.graphrag import GraphRAG

def main():
    from dotenv import load_dotenv
    load_dotenv()
    graphrag = GraphRAG()
    ans = graphrag.query("任正非是谁？")
    print(ans)

if __name__ == '__main__':
    main()