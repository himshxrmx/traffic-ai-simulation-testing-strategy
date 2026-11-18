"""Small helper around python-docx so every internship report has the same clean look."""
import re

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x2E, 0x86, 0x7E)
GREY = RGBColor(0x59, 0x59, 0x59)


class Report:
    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.12
        for lvl, size in [(1, 16), (2, 13), (3, 11.5)]:
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.color.rgb = NAVY if lvl < 3 else TEAL
            h.font.bold = True
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        sec = self.doc.sections[0]
        sec.left_margin = sec.right_margin = Cm(2.2)
        sec.top_margin = sec.bottom_margin = Cm(2.0)
        self._page_numbers()
        self.fig_n = 0
        self.tab_n = 0
        self.words = 0

    def _page_numbers(self):
        p = self.doc.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(9)
        run.font.color.rgb = GREY

    def _count(self, text):
        self.words += len(re.findall(r"[A-Za-z0-9₹%][\w%₹.,'/-]*", text))

    def title_page(self, title, subtitle, meta_lines):
        for _ in range(5):
            self.doc.add_paragraph()
        p = self.doc.add_paragraph()
        r = p.add_run(title)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(24), NAVY
        p = self.doc.add_paragraph()
        r = p.add_run(subtitle)
        r.font.size, r.font.color.rgb = Pt(14), TEAL
        self.doc.add_paragraph()
        for line in meta_lines:
            p = self.doc.add_paragraph()
            r = p.add_run(line)
            r.font.size, r.font.color.rgb = Pt(11), GREY
            p.paragraph_format.space_after = Pt(2)
        self.doc.add_page_break()

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def h3(self, text):
        self.doc.add_heading(text, level=3)

    def p(self, text, bold_lead=None, italic=False):
        para = self.doc.add_paragraph()
        if bold_lead:
            r = para.add_run(bold_lead + " ")
            r.bold = True
            self._count(bold_lead)
        r = para.add_run(text)
        r.italic = italic
        self._count(text)
        return para

    def bullets(self, items, numbered=False):
        style = "List Number" if numbered else "List Bullet"
        for it in items:
            para = self.doc.add_paragraph(style=style)
            if isinstance(it, tuple):
                r = para.add_run(it[0] + " ")
                r.bold = True
                para.add_run(it[1])
                self._count(it[0] + " " + it[1])
            else:
                para.add_run(it)
                self._count(it)
            para.paragraph_format.space_after = Pt(3)

    def table(self, header, rows, caption=None, widths=None, font=9.5):
        if caption:
            self.tab_n += 1
            cp = self.doc.add_paragraph()
            r = cp.add_run(f"Table {self.tab_n}. {caption}")
            r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), NAVY
            cp.paragraph_format.space_after = Pt(2)
            cp.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(h))
            r.bold, r.font.size = True, Pt(font)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(font)
                self._count(str(v))
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        # repeat the header row on a new page, never split a row, and keep short tables on one page
        trpr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trpr.append(el)
        for ri, row in enumerate(t.rows):
            rp = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); rp.append(cs)
            if len(t.rows) <= 18 and ri < len(t.rows) - 1:
                for c in row.cells:
                    for para in c.paragraphs:
                        para.paragraph_format.keep_with_next = True
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path, caption, width_cm=15.5):
        self.fig_n += 1
        self.doc.add_picture(path, width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(f"Figure {self.fig_n}. {caption}")
        r.italic, r.font.size, r.font.color.rgb = True, Pt(9), GREY
        self._count(caption)

    def code(self, text):
        para = self.doc.add_paragraph()
        r = para.add_run(text)
        r.font.name, r.font.size = "Consolas", Pt(9)
        para.paragraph_format.left_indent = Cm(0.6)

    def page_break(self):
        self.doc.add_page_break()

    # ---- reuse of text blocks made by docx2blocks.py -------------------------------------------
    def emit(self, blocks, images=None, rename=None, drop=()):
        """Write blocks. rename maps old heading text -> new heading text; drop lists headings whose
        whole section (until the next heading of the same or higher level) is skipped."""
        rename = rename or {}
        skip_level = None
        for b in blocks:
            if b[0] == "h":
                if skip_level is not None and b[1] <= skip_level:
                    skip_level = None
                if b[2] in drop:
                    skip_level = b[1]
                    continue
            if skip_level is not None:
                continue
            kind = b[0]
            if kind == "h":
                {1: self.h1, 2: self.h2}.get(b[1], self.h3)(rename.get(b[2], b[2]))
            elif kind == "p":
                self.p(b[1], bold_lead=b[2])
            elif kind in ("b", "n"):
                self.bullets([tuple(i) if isinstance(i, list) else i for i in b[1]], numbered=kind == "n")
            elif kind == "t":
                self.table(b[2], b[3], caption=b[1])
            elif kind == "img" and images and b[1] < len(images) and images[b[1]]:
                self.figure(images[b[1]], b[2] or "")
            elif kind == "code":
                self.code(b[1])


    def save(self, path):
        self.doc.save(path)
        return self.words


def section(blocks, heading, include_heading=True):
    """Blocks from `heading` up to the next heading of the same or higher level."""
    out, level = [], None
    for b in blocks:
        if level is None:
            if b[0] == "h" and b[2] == heading:
                level = b[1]
                if include_heading:
                    out.append(b)
            continue
        if b[0] == "h" and b[1] <= level:
            break
        out.append(b)
    if level is None:
        raise KeyError(heading)
    return out
