from __future__ import annotations

from pathlib import Path

import pytest

COMMENT_BLOCK = "\n".join(
    [
        "#中華經典古籍精校",
        "#下載最新版本，請訪問: https://github.com/bgc2017/chtxt",
        "#如果您發現錯字，請反饋至: .../issues",
        "#創建日期: 2021/02/15",
        "#更新日期: 2021/02/15",
        "",
    ]
)


def write_book(root: Path, relpath: str, body: str) -> Path:
    """Create a corpus file with the standard 5-line comment header + body."""
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(COMMENT_BLOCK + "\n" + body.lstrip("\n"), encoding="utf-8")
    return p


@pytest.fixture
def make_book():
    """Expose the corpus-file writer to tests."""
    return write_book


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A miniature corpus reproducing the named format quirks."""
    root = tmp_path / "chtxt"

    # 唐詩三百首: '##' used as the book head line + '## 体裁' groups + ○ poems
    write_book(
        root,
        "n.蒙學/唐詩三百首.txt",
        "## 唐詩三百首 〔清〕蘅塘退士 孫洙 編\n\n"
        "## 五言古詩\n\n"
        "○ 感遇(四首之一) / 張九齡\n\n"
        "孤鴻海上來，池潢不敢顧。\n"
        "側見雙翠鳥，巢在三珠樹。\n\n"
        "○ 下終南山 / 李白\n\n"
        "暮從碧山下，山月隨人歸。\n",
    )

    # 說文解字: multi-〔〕 head + tab-continuation dictionary entries
    write_book(
        root,
        "m.字書訓詁/說文解字.txt",
        "# 說文解字〔漢〕許慎 撰〔南唐〕徐鍇、徐鉉 注 〔清〕段玉裁 注\n\n"
        "㐁\t\n"
        "\t【大徐本】 他念切。舌皃。\n"
        "\t【小徐本】 他暗反。舌貌。\n"
        "敝\t【段注本】 帗也。\n",
    )

    # 康熙字典 sharded into p1 + p2 -> must merge into one book
    write_book(
        root,
        "m.字書訓詁/康熙字典_p1.txt",
        "# 康熙字典 〔清〕張玉書、陳廷敬等 編撰 (前半部分)\n\n"
        "敝\t【卯集下】【攴字部】 敝【廣韻】毘祭切，音幣。\n",
    )
    write_book(
        root,
        "m.字書訓詁/康熙字典_p2.txt",
        "# 康熙字典 〔清〕張玉書、陳廷敬等 編撰 (後半部分)\n\n"
        "龍\t【亥集下】【龍字部】 龍【廣韻】力鍾切。\n",
    )

    # 南齊書: no '# ' head, first '## ' is a section (序), NOT the title
    write_book(
        root,
        "j.史書/南齊書.txt",
        "## 序\n\n"
        "史臣曰：序者，緒也。\n\n"
        "## 卷一‧高帝紀第一\n\n"
        "太祖高皇帝諱道成，字紹伯，姓蕭氏。\n\n"
        "生而神異，及長，博學能文。\n",
    )

    # 史記: '# ' head + '## 卷' hard boundaries + blank-line paragraphs
    write_book(
        root,
        "j.史書/史記.txt",
        "# 史記〔漢〕司馬遷\n\n"
        "## 卷一‧五帝本紀第一\n\n"
        + "".join(f"甲{i}段" + "字" * 120 + "。\n\n" for i in range(6))
        + "## 卷二‧夏本紀第二\n\n"
        + "乙一段" + "字" * 120 + "。\n",
    )

    # 詩經 (real) — verse collection with ○ poems, no '## ' groups
    write_book(
        root,
        "0.詩詞/a.先秦兩漢篇/詩經.txt",
        "# 詩經\n\n"
        "○ 國風·周南·關雎\n\n"
        "關關雎鳩，在河之洲。\n"
        "窈窕淑女，君子好逑。(一章)\n\n"
        "○ 國風·周南·葛覃\n\n"
        "葛之覃兮，施于中谷。\n",
    )

    # 詩經_symbolic — pointer stub, must be is_stub and excluded from passages
    write_book(root, "a.儒家/詩經_symbolic.txt", "# 詩經\n\n內容見 0.詩詞/a.先秦兩漢篇/詩經.txt\n")

    # empty poetry file
    write_book(root, "0.詩詞/b.魏晉篇/b.魏晉篇.txt", "")

    return root
