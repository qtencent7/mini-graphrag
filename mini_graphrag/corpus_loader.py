class CorpusLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        """
        加载语料库文件。
        如果文件是PDF格式，则读取前10页内容。
        否则尝试作为普通文本文件读取。
        
        返回:
            str: 文件内容
        """
        import os
        
        # 获取文件扩展名
        _, file_extension = os.path.splitext(self.path)
        
        # 检查文件是否为PDF
        if file_extension.lower() == '.pdf':
            return self._load_pdf()
        else:
            # 对于非PDF文件，尝试作为文本文件读取
            try:
                with open(self.path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f"读取文件时出错: {e}")
                return ""
    
    def _load_pdf(self):
        """
        读取PDF文件的前10页内容。
        
        返回:
            str: PDF文件前10页的文本内容
        """
        try:
            # 尝试导入PyPDF2库
            try:
                import PyPDF2
            except ImportError:
                print("未安装PyPDF2库。请使用命令安装: pip install pdfplumber")
                return ""
            
            # 打开PDF文件
            with open(self.path, 'rb') as file:
                # 创建PDF阅读器对象
                pdf_reader = PyPDF2.PdfReader(file)

                # 获取PDF页数
                num_pages = len(pdf_reader.pages)
                print(f"PDF文件总页数: {num_pages}")
                # 确定要读取的页数（从第20页开始，最多读10页）
                start_page = 19  # 索引从0开始，所以第20页是索引19
                pages_to_read = 2
                
                # 如果起始页超出了PDF的总页数，则返回空字符串
                if start_page >= num_pages:
                    print(f"警告: PDF只有{num_pages}页，无法从第20页开始读取")
                    return ""
                
                # 提取文本
                text = ""
                for page_num in range(start_page, start_page + pages_to_read):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                print(f"已成功读取PDF文件的前{pages_to_read}页")
                return text
                
        except Exception as e:
            print(f"读取PDF文件时出错: {e}")
            return ""
        
    