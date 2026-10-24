"""输入边界契约测试：解不出来的字节必须报错，不能被"替换"成 U+FFFD。

对应规约：`MiroFish/AGENTS.md §1.1`（边界上发明默认值 / 空状态与失败不可区分）；
掩盖台账条目：F5（`file_parser.py` 的 `errors='replace'` 与四层回退链）。
"""

import pytest

from app.utils.file_parser import (
    FileParser,
    TextDecodingError,
    _read_text_with_fallback,
)


def test_utf8_text_is_read(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes("中文内容\n第二行".encode("utf-8"))
    assert _read_text_with_fallback(str(path)) == "中文内容\n第二行"


def test_gb18030_text_is_read(tmp_path):
    path = tmp_path / "b.txt"
    path.write_bytes("国标编码的文档内容".encode("gb18030"))
    assert "国标编码" in _read_text_with_fallback(str(path))


# 这些字节在 utf-8(-sig) / gb18030 / big5 / utf-16 下都无法严格解码
_UNDECODABLE = b"\xff\xfe\x00\x81\x8d\x90\xff"


def test_undecodable_bytes_raise_instead_of_replacement_characters(tmp_path):
    path = tmp_path / "c.txt"
    path.write_bytes(_UNDECODABLE)
    with pytest.raises(TextDecodingError) as error:
        _read_text_with_fallback(str(path))
    message = str(error.value)
    assert "无法解码" in message
    assert "errors='replace'" in message, "错误里要说明为什么不用 replace"


def test_decode_error_keeps_the_attempted_encodings(tmp_path):
    path = tmp_path / "d.txt"
    path.write_bytes(_UNDECODABLE)
    with pytest.raises(TextDecodingError) as error:
        _read_text_with_fallback(str(path))
    assert "utf-8" in str(error.value)


def test_multiple_documents_fail_loudly_when_one_is_broken(tmp_path):
    """一个文档解不出来就是失败：不能写成"提取失败"的占位行继续跑。"""

    good = tmp_path / "good.txt"
    good.write_bytes("正常文档".encode("utf-8"))
    bad = tmp_path / "bad.txt"
    bad.write_bytes(_UNDECODABLE)

    with pytest.raises(Exception) as error:
        FileParser.extract_from_multiple([str(good), str(bad)])
    assert "bad.txt" in str(error.value)
    assert "第 2 个文档" in str(error.value)


def test_multiple_documents_are_joined(tmp_path):
    first = tmp_path / "1.txt"
    first.write_bytes("第一份".encode("utf-8"))
    second = tmp_path / "2.txt"
    second.write_bytes("第二份".encode("utf-8"))
    merged = FileParser.extract_from_multiple([str(first), str(second)])
    assert "第一份" in merged and "第二份" in merged
    assert "=== 文档 1: 1.txt ===" in merged
