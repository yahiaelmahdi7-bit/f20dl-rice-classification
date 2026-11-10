# Generated with AI assistance (Claude Sonnet, 2026-11-10).
# Prompt (summary): build a five-slide 16:9 mini-viva deck for F20DL Group 16 (rice variety
# classification) from the finished report and its figures. One claim-style title, one key
# number, at most four short bullets and one real-size figure per slide. Every number must come
# from report/FACTS.md. Text stays editable (real text boxes, pictures and tables).
# Run: python docs/viva/make_deck.py   (needs python-pptx and Pillow)
import os
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "figures")
INK = RGBColor(0x14, 0x17, 0x1F)
ACC = RGBColor(0x1F, 0x6F, 0x5C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
M = 0.6
SW, SH = 13.333, 7.5
COL_X = 6.4
COL_W = SW - M - COL_X
LEFT_W = 5.4
FIG_Y = 1.65

prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
blank = prs.slide_layouts[6]


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb, tf


def run(p, text, font, size, color=INK, bold=False):
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return r


def rule(slide, x, y, w):
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.shadow.inherit = False
    ln.line.color.rgb = ACC
    ln.line.width = Pt(0.75)


def bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.22)))
    pPr.set("indent", str(-Inches(0.22)))
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•")


def picture(slide, name, x, y, maxw, maxh, crop_top=0, crop_bottom=0):
    path = os.path.join(FIG, name)
    w, h = Image.open(path).size
    vh = h - crop_top - crop_bottom
    s = min(maxw / w, maxh / vh)
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w * s), Inches(vh * s))
    pic.crop_top = crop_top / h
    pic.crop_bottom = crop_bottom / h
    return w * s, vh * s


def table(slide, x, y, w, rows, col_w, row_h, size=11):
    n, c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n, c, Inches(x), Inches(y), Inches(w), Inches(row_h * n))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            run(p, val, "Arial", size, INK, bold=(i == 0))
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag))
                if tag == "a:lnB":
                    ln.set("w", "9525" if i else "12700")
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    clr = etree.SubElement(sf, qn("a:srgbClr"))
                    clr.set("val", "1F6F5C" if i == 0 else "D8DCE2")
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            # border elements must precede the fill in tcPr
            fill = tcPr.find(qn("a:solidFill"))
            if fill is not None:
                tcPr.remove(fill)
                tcPr.append(fill)


def base(n, title, key, caption, bullets):
    s = prs.slides.add_slide(blank)
    _, tf = textbox(s, M, 0.45, SW - 2 * M, 0.95, MSO_ANCHOR.BOTTOM)
    run(tf.paragraphs[0], title, "Georgia", 28, INK, bold=True)
    rule(s, M, 1.5, SW - 2 * M)
    _, tf = textbox(s, M, 1.65, LEFT_W, 0.95, MSO_ANCHOR.MIDDLE)
    run(tf.paragraphs[0], key, "Arial", 54, ACC, bold=True)
    _, tf = textbox(s, M, 2.7, LEFT_W, 0.7)
    run(tf.paragraphs[0], caption, "Arial", 13, INK)
    _, tf = textbox(s, M, 3.65, LEFT_W, 3.1)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        bullet(p)
        run(p, b, "Arial", 15, INK)
    _, tf = textbox(s, M, 7.0, SW - 2 * M, 0.25)
    run(tf.paragraphs[0], f"F20DL Group 16 · slide {n}/5 · Speaker {n}: ______", "Arial", 10, INK)
    return s


# 1
s = base(1, "Numbers and pictures both sort five rice varieties near perfectly.",
         "74703",
         "grains left after linking 75000 images to 106 numeric columns and dropping 297 duplicate pairs",
         ["Five varieties: Arborio, Basmati, Ipsala, Jasmine, Karacadag.",
          "Question: do measured numbers or raw images classify the same grains better?",
          "Split 70/15/15 by class and lot: train 52292, validation 11205, test 11206.",
          "We also ask what near-perfect scores depend on."])
picture(s, "eda_examples.png", COL_X, FIG_Y, COL_W, 5.2, crop_top=50, crop_bottom=8)

# 2
s = base(2, "A late Jasmine lot hides inside the data.",
         "×1.81",
         "median grain area of late Jasmine against early Jasmine (3283 grains, tint ×2.64). Cause not established.",
         ["All 75000 row-to-image links rebuilt one-to-one.",
          "297 exact duplicate pairs, all Ipsala, dropped.",
          "k-means at k=6 puts a share 0.9896 of the late lot in one cluster.",
          "Best GMM k=5, ARI 0.9437; best k-means k=6, ARI 0.8934."])
w, h = picture(s, "eda_jasmine_break.png", COL_X, FIG_Y, COL_W, 3.0)
picture(s, "cluster_overview.png", COL_X, FIG_Y + h + 0.2, COL_W, 2.5)

# 3
s = base(3, "A depth-four tree, readable by hand, still scores 99.03 percent.",
         "99.03%",
         "depth-four tree, test accuracy. Best baseline, HGB: 99.88% (13 errors in 11206 test grains)",
         ["Tuned on validation only, code frozen, exactly 3 logged test reads.",
          "Shape alone: HGB 97.99. Colour alone: HGB 99.14.",
          "Tree splits on SHAPEFACTOR_3, MAJOR_AXIS, PERIMETER: pixel-scale size, so camera scale matters.",
          "Use: readable first stage with a familiarity check, recalibrated per camera."])
w, h = picture(s, "baselines_tree_depth4.png", COL_X, FIG_Y, COL_W, 2.8)
rows = [["Model (test accuracy, %)", "all106"],
        ["HGB", "99.88"], ["MLP, seed mean", "99.92"], ["CNN on images, seed mean", "99.69"],
        ["Logistic regression", "99.83"], ["kNN", "99.79"], ["Perceptron", "99.67"],
        ["Naive Bayes", "99.05"], ["Decision tree, depth four", "99.03"]]
table(s, COL_X, FIG_Y + h + 0.15, COL_W, rows, [COL_W - 1.2, 1.2], 0.25)

# 4
s = base(4, "MLP and HGB tie; the CNN gap depends on the seed.",
         "99.92%",
         "MLP test accuracy, seed mean (11, 9, 8 errors), against HGB 99.88%",
         ["HGB against MLP: no significant difference on any seed (exact McNemar, Holm).",
          "CNN makes 15 to 50 errors across three seeds; mean 99.69%.",
          "HGB beats the CNN for seeds zero and two, not seed one.",
          "Routing answers a share 0.9797 of test grains with 2 errors; optimistic."])
w, h = picture(s, "report_learning_curves.png", COL_X, FIG_Y, COL_W, 3.65)
rows = [["Holm-adjusted p", "Seed 0", "Seed 1", "Seed 2"],
        ["HGB vs MLP", "0.7266", "0.7895", "0.2266"],
        ["HGB vs CNN", "4.74e-7", "0.8318", "1.38e-4"]]
table(s, COL_X, FIG_Y + h + 0.2, COL_W, rows, [COL_W - 3.3, 1.1, 1.1, 1.1], 0.33, size=12)

# 5
s = base(5, "Test accuracy hides a lot the models have never seen.",
         "1.42%",
         "HGB recall on 492 late Jasmine test grains when the lot is held out (mean confidence 0.9263)",
         ["CNN seeds recover 0.41%, 1.42%, 16.06%; size-free HGB keeps 84.55%.",
          "Removing tint cuts CNN recall from 99.93% to 4.00%; colour and size stay mixed.",
          "Limits: one camera, cause unknown, routing floors set in-sample.",
          "Takeaway: add a familiarity check; do not trust test accuracy alone."])
picture(s, "report_late_lot_recall.png", COL_X, FIG_Y, COL_W, 5.2, crop_top=44, crop_bottom=4)

prs.save(os.path.join(HERE, "deck.pptx"))
