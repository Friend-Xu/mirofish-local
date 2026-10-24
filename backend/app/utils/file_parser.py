"""
文件解析工具
支持PDF、Markdown、TXT文件的文本提取
"""

import os
from pathlib import Path
from typing import List, Optional


class TextDecodingError(ValueError):
    """文本解码失败。

    历史上这里有四层"回退链"（UTF-8 → charset_normalizer → chardet →
    UTF-8 + `errors='replace'`），最后一步会把解不出来的字节替换成 U+FFFD：
    文档内容被静默改掉，而调用方拿到的是"解析成功"。缺了编码判断就报错，
    让用户换一个能解码的文件或显式声明编码。
    """


# 显式编码优先级：先按列表逐个"严格解码"，都不成再让检测库给一个候选，
# 候选也必须能严格解码；全部失败即抛错（不再有 errors='replace'）。
_PREFERRED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16")


def _strict_decode(data: bytes, encoding: str) -> Optional[str]:
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return None


def _detect_encoding(data: bytes) -> Optional[str]:
    """让检测库给出候选编码（不负责解码）。"""

    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best and best.encoding:
            return str(best.encoding)
    except Exception:  # noqa: BLE001 - 检测库缺失/异常都不影响下一步
        pass

    try:
        import chardet

        result = chardet.detect(data)
        encoding = result.get("encoding") if result else None
        if encoding:
            return str(encoding)
    except Exception:  # noqa: BLE001
        pass

    return None


def _read_text_with_fallback(file_path: str) -> str:
    """读取文本文件：按显式编码优先级严格解码，解不出来就报错。

    顺序：`_PREFERRED_ENCODINGS` → 检测库候选（同样严格解码）。
    不再使用 `errors='replace'`：把解不出的字节换成 U+FFFD 属于**改写文档内容**，
    而下游（图谱抽取、报告引用）无法区分它和原文。
    """

    data = Path(file_path).read_bytes()
    attempted: List[str] = []

    for encoding in _PREFERRED_ENCODINGS:
        decoded = _strict_decode(data, encoding)
        if decoded is not None:
            return decoded
        attempted.append(encoding)

    detected = _detect_encoding(data)
    if detected:
        attempted.append(detected)
        decoded = _strict_decode(data, detected)
        if decoded is not None:
            return decoded

    sample = data[:24]
    raise TextDecodingError(
        f"无法解码文件 {file_path}（{len(data)} 字节）："
        f"已尝试 {', '.join(attempted)}；开头字节 {sample!r}。"
        "请把文件转成 UTF-8/GB18030，或显式声明编码后再上传——"
        "用 errors='replace' 会把解不出的字节悄悄换成 U+FFFD，等于改写文档内容"
    )


class FileParser:
    """文件解析器"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        检查文件是否为支持的格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果文件格式受支持则返回 True
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        从文件中提取文本
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的文本内容
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"无法处理的文件格式: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """从PDF提取文本"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("需要安装PyMuPDF: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """从Markdown提取文本，支持自动编码检测"""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """从TXT提取文本，支持自动编码检测"""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        从多个文件提取文本并合并
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            合并后的文本
            
        Raises:
            Exception: 任意一个文件提取失败即抛出（不把失败写成文本里的占位行——
                那样"少了一份文档"会看起来像"文档都在，只是有一行提示"）
        """
        all_texts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
            except Exception as error:
                message = (
                    f"第 {i} 个文档提取失败（{Path(file_path).name}）：{error}"
                )
                # 能在构造时接字符串的类型就保留类型（API 依赖 ValueError/OSError 区分 400/404），
                # 其余原样抛出——原始异常里已经带了文件名。
                if isinstance(error, (OSError, ValueError, ImportError, RuntimeError)):
                    raise type(error)(message) from error
                raise
            filename = Path(file_path).name
            all_texts.append(f"=== 文档 {i}: {filename} ===\n{text}")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    将文本分割成小块
    
    Args:
        text: 原始文本
        chunk_size: 每块的字符数
        overlap: 重叠字符数
        
    Returns:
        文本块列表
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界处分割
        if end < len(text):
            # 查找最近的句子结束符
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下一个块从重叠位置开始
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

