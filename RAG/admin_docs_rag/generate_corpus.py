"""Generate the synthetic admin-docs corpus and eval set from world.yaml.

Writes:
  corpus/*.pdf             one PDF per document (some rendered as image-only scans)
  corpus/documents.jsonl   ground-truth metadata per document
  questions.jsonl          eval questions with reference answers and gold evidence

Gold evidence is a list of facts the answer needs. Each fact lists the
(doc_id, page) pairs where it is printed; retrieving any one of them counts.

Usage: python generate_corpus.py [--world world.yaml] [--out .]
"""

import argparse
import calendar
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pymupdf as fitz
import numpy as np
import yaml
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

EUR = "€"


# ---------------------------------------------------------------- formatting


def D(x) -> Decimal:
    return Decimal(str(x))


def money(x) -> Decimal:
    return D(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt_money(x) -> str:
    return f"{EUR}{money(x):,.2f}"


def parse_date(s) -> date:
    return s if isinstance(s, date) else date.fromisoformat(str(s))


def fmt_date(d) -> str:
    d = parse_date(d)
    return f"{d.day} {d.strftime('%B %Y')}"


def fmt_period(start, end) -> str:
    s, e = parse_date(start), parse_date(end)
    return f"{s.day} {s.strftime('%B')} {s.year} to {e.day} {e.strftime('%B')} {e.year}"


def period_phrase(start, end) -> str:
    """'January to March 2025' for question text."""
    s, e = parse_date(start), parse_date(end)
    if s.year == e.year:
        return f"{s.strftime('%B')} to {e.strftime('%B')} {e.year}"
    return f"{s.strftime('%B %Y')} to {e.strftime('%B %Y')}"


def month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


# ---------------------------------------------------------------- PDF building


class Mark(Flowable):
    """Zero-size flowable that records which page it lands on."""

    def __init__(self, key, sink):
        super().__init__()
        self.key, self.sink = key, sink

    def wrap(self, *_):
        return (0, 0)

    def draw(self):
        self.sink.setdefault(self.key, set()).add(self.canv.getPageNumber())


@dataclass
class Style:
    font: str = "Helvetica"
    bold: str = "Helvetica-Bold"
    color: colors.Color = colors.HexColor("#1f4e79")
    tagline: str = ""
    footer: str = ""


STYLES = {
    "BrightSpark Energy": Style(color=colors.HexColor("#e8710a"), tagline="Powering homes since 1998",
                                footer="BrightSpark Energy Ltd, 40 Dynamo Way, Lindenford LF1 2BS. Registered no. 0458812."),
    "Northfield Gas": Style("Times-Roman", "Times-Bold", colors.HexColor("#0b6e4f"), "Warmth you can count on",
                            footer="Northfield Gas plc, Northfield House, Harrowgate Road, Lindenford LF9 4NG."),
    "Lindenford Water": Style(color=colors.HexColor("#1565c0"), tagline="Your local water and wastewater company",
                              footer="Lindenford Water Services, Reservoir Lane, Lindenford LF3 8WW."),
    "Nimbus Telecom": Style(color=colors.HexColor("#6a1b9a"), tagline="Connected, everywhere",
                            footer="Nimbus Telecom Ltd, 9 Signal Street, Lindenford LF1 6NT. Customer Care: 0700 900 100."),
    "Okafor Illustration": Style("Helvetica", "Helvetica-Bold", colors.HexColor("#c2185b"), "Illustration for books, brands and museums",
                                 footer="Okafor Illustration, Flat 3, 14 Quill Lane, Lindenford LF2 7QA. hello@okafor-illustration.example"),
    "Pixelforge Software": Style(color=colors.HexColor("#283593"), footer="Pixelforge Software BV, Kade 12, Port Arden."),
    "The Loom Coworking": Style("Times-Roman", "Times-Bold", colors.HexColor("#5d4037"), "Work, woven together",
                                footer="The Loom Coworking, 2 Mill Yard, Lindenford LF2 1LM."),
    "Kestrel Computers": Style("Courier", "Courier-Bold", colors.HexColor("#37474f"), footer="Kestrel Computers, Unit 7, Falcon Park, Lindenford LF6 3KC."),
    "Inkwell Print Studio": Style("Times-Roman", "Times-Bold", colors.HexColor("#00695c"), footer="Inkwell Print Studio, 31 Press Row, Lindenford LF2 9IP."),
    "Hartley & Finch Accountants": Style("Times-Roman", "Times-Bold", colors.HexColor("#4e342e"), "Chartered accountants",
                                         footer="Hartley & Finch, 5 Ledger Court, Lindenford LF1 4HF."),
    "Harbourview Lettings": Style("Times-Roman", "Times-Bold", colors.HexColor("#00838f"), "Lettings and property management",
                                  footer="Harbourview Lettings, 88 Quay Street, Lindenford LF4 2HL."),
    "Shieldwell Insurance": Style(color=colors.HexColor("#ad1457"), tagline="Home insurance, simply",
                                  footer="Shieldwell Insurance Company Ltd. Authorised and regulated by the Lindenford Financial Authority."),
    "Arrowline Motor Insurance": Style("Courier", "Courier-Bold", colors.HexColor("#bf360c"),
                                       footer="Arrowline Motor Insurance, 1 Junction Road, Lindenford LF7 1AR."),
    "Lindenford Revenue Office": Style("Times-Roman", "Times-Bold", colors.HexColor("#212121"), "Self Assessment",
                                       footer="Lindenford Revenue Office, PO Box 1200, Lindenford LF1 0RO."),
    "Calder Bank": Style(color=colors.HexColor("#1a237e"), tagline="Banking for Lindenford",
                         footer="Calder Bank plc, 1 Calder Square, Lindenford LF1 1CB."),
}


class Doc:
    """One PDF: a list of flowables plus the page each marked fact lands on."""

    def __init__(self, vendor, title):
        self.vendor, self.title = vendor, title
        self.st = STYLES[vendor]
        self.story, self.marks = [], {}
        f, b = self.st.font, self.st.bold
        self.p = ParagraphStyle("p", fontName=f, fontSize=9.5, leading=13)
        self.small = ParagraphStyle("s", fontName=f, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
        self.h1 = ParagraphStyle("h1", fontName=b, fontSize=16, leading=20, textColor=self.st.color, spaceAfter=6)
        self.h2 = ParagraphStyle("h2", fontName=b, fontSize=11.5, leading=15, textColor=self.st.color, spaceBefore=8, spaceAfter=4)
        self.big = ParagraphStyle("big", fontName=b, fontSize=20, leading=24, textColor=self.st.color)
        self.right = ParagraphStyle("r", parent=self.p, alignment=TA_RIGHT)

    # flowable helpers -------------------------------------------------------
    def mark(self, key, *flowables):
        self.story.append(KeepTogether([Mark(key, self.marks), *flowables]))

    def add(self, *flowables):
        self.story.extend(flowables)

    def para(self, text, style=None, key=None):
        f = Paragraph(text, style or self.p)
        self.mark(key, f) if key else self.add(f)

    def table(self, rows, widths, header=True, key=None, zebra=True):
        t = Table([[Paragraph(str(c), self.p) for c in r] for r in rows], colWidths=widths)
        cmds = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), self.st.color),
                     ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
            t = Table([[Paragraph(f"<font color='white'><b>{c}</b></font>", self.p) for c in rows[0]]]
                      + [[Paragraph(str(c), self.p) for c in r] for r in rows[1:]], colWidths=widths)
        if zebra:
            cmds += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")])]
        t.setStyle(TableStyle(cmds))
        self.mark(key, t, Spacer(1, 4)) if key else self.add(t, Spacer(1, 4))

    def kv(self, pairs, key=None, widths=(55 * mm, 110 * mm)):
        self.table([[f"<b>{k}</b>", v] for k, v in pairs], widths, header=False, key=key, zebra=False)

    # rendering --------------------------------------------------------------
    def _decorate(self, canv, doc):
        w, h = A4
        canv.saveState()
        canv.setFillColor(self.st.color)
        canv.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.white)
        canv.setFont(self.st.bold, 15)
        canv.drawString(18 * mm, h - 13 * mm, self.vendor)
        if self.st.tagline:
            canv.setFont(self.st.font, 8.5)
            canv.drawString(18 * mm, h - 18 * mm, self.st.tagline)
        canv.setFillColor(colors.HexColor("#666666"))
        canv.setFont(self.st.font, 7.5)
        canv.drawString(18 * mm, 10 * mm, self.st.footer)
        canv.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")
        canv.restoreState()

    def build(self, path):
        doc = SimpleDocTemplate(str(path), pagesize=A4, topMargin=30 * mm, bottomMargin=18 * mm,
                                leftMargin=18 * mm, rightMargin=18 * mm, title=self.title, author=self.vendor)
        doc.build(self.story, onFirstPage=self._decorate, onLaterPages=self._decorate)
        with fitz.open(path) as pdf:
            return pdf.page_count


def to_scan(path: Path, rng: random.Random):
    """Rasterise a PDF and degrade it lightly, producing an image-only PDF."""
    src = fitz.open(path)
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, expand=False, fillcolor=255)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 10, arr.shape)
        arr = np.clip(arr * 0.95 + 8, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=55)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    src.close()
    out.save(path, garbage=4, deflate=True)
    out.close()


# ---------------------------------------------------------------- world -> documents


@dataclass
class Question:
    type: str
    question: str
    answer: str | None
    evidence: list = field(default_factory=list)  # [[(doc_id, key), ...], ...]
    value: float | None = None
    near_miss: bool = False


class Corpus:
    def __init__(self, world, out: Path):
        self.w, self.out = world, out
        self.rng = random.Random(world["seed"])
        self.docs, self.marks, self.questions = [], {}, []
        self.cur = world["currency"]
        self.hh = world["household"]

    def emit(self, doc_id, d: Doc, meta, scanned=False):
        path = self.out / "corpus" / f"{doc_id}.pdf"
        pages = d.build(path)
        if scanned:
            to_scan(path, self.rng)
        self.marks[doc_id] = d.marks
        self.docs.append({"doc_id": doc_id, "file": f"corpus/{doc_id}.pdf", "title": d.title,
                          "vendor": d.vendor, "pages": pages, "scanned": scanned, **meta})

    def q(self, qtype, text, answer, evidence, value=None, near_miss=False):
        self.questions.append(Question(qtype, text, answer, evidence, value, near_miss))

    def pick(self, *variants):
        return self.rng.choice(variants)

    def address_block(self, d: Doc, name):
        d.para("<br/>".join([name, *self.hh["address"]]))
        d.add(Spacer(1, 6))

    # -------------------------------------------------------------- electricity
    def electricity(self):
        e = self.w["electricity"]
        read = e["periods"][0]["read_start"]
        bills = []
        for i, p in enumerate(e["periods"]):
            start, end = parse_date(p["start"]), parse_date(p["end"])
            tariff = next(t for t in e["tariffs"] if parse_date(t["start"]) <= start <= parse_date(t["end"]))
            days = (end - start).days + 1
            energy = money(D(p["kwh"]) * D(tariff["unit_rate"]))
            standing = money(days * D(tariff["standing"]))
            sub = energy + standing
            vat = money(sub * D(e["vat_rate"]))
            total = sub + vat
            bill_date = end + timedelta(days=6)
            dd_date = bill_date + timedelta(days=14)
            doc_id = f"elec-{start.year}-q{(start.month - 1) // 3 + 1}"
            bill_no = f"BSE-B-{end.year}{end.month:02d}-{4410 + i}"

            d = Doc(e["vendor"], f"Electricity bill {period_phrase(start, end)}")
            d.para("Your electricity bill", d.h1)
            self.address_block(d, "Robin Marlowe")
            d.kv([("Account number", e["account"]), ("Bill number", bill_no), ("Bill date", fmt_date(bill_date)),
                  ("Billing period", fmt_period(start, end))], key="header")
            d.para("Your bill at a glance", d.h2)
            d.kv([("Balance brought forward", fmt_money(0)), ("Charges for this period (incl. VAT)", fmt_money(total)),
                  ("<b>Total amount due</b>", f"<b>{fmt_money(total)}</b>"),
                  ("Payment", f"We will collect {fmt_money(total)} by Direct Debit on or after {fmt_date(dd_date)}.")], key="total")
            d.para("Ways to save", d.h2)
            d.para("Running a washing machine at 30 degrees instead of 40 can cut its energy use by around a third. "
                   "Draught-proofing windows and doors is one of the cheapest ways to keep heat in. "
                   "Ask us about a free smart meter upgrade.")
            d.para("Moving home?", d.h2)
            d.para("Tell us at least 2 working days before you move and give us a final meter reading on the day. "
                   "We will send a final bill to your new address within 14 days.")
            d.add(PageBreak())
            d.para("Your charges explained", d.h1)
            read_end = read + p["kwh"]
            d.table([["Meter", "Previous reading", "Current reading", "Units used"],
                     [e["meter"], f"{read:,} (actual)", f"{read_end:,} (actual)", f"{p['kwh']:,} kWh"]],
                    [35 * mm, 45 * mm, 45 * mm, 40 * mm], key="usage")
            d.table([["Description", "Quantity", "Rate", "Amount"],
                     ["Electricity used", f"{p['kwh']:,} kWh", f"{D(tariff['unit_rate']) * 100:.2f}p per kWh", fmt_money(energy)],
                     ["Standing charge", f"{days} days", f"{D(tariff['standing']) * 100:.2f}p per day", fmt_money(standing)],
                     ["Subtotal", "", "", fmt_money(sub)],
                     [f"VAT at {int(e['vat_rate'] * 100)}%", "", "", fmt_money(vat)],
                     ["<b>Total charges</b>", "", "", f"<b>{fmt_money(total)}</b>"]],
                    [60 * mm, 30 * mm, 40 * mm, 35 * mm], key="charges")
            d.para("Your tariff", d.h2)
            d.para(f"You are on <b>{tariff['name']}</b>, a fixed-price tariff. Your unit rate and standing charge are fixed "
                   f"until <b>{fmt_date(tariff['end'])}</b>. There is no exit fee if you switch in the last 49 days of the tariff.",
                   key="tariff")
            d.para("Complaints", d.h2)
            d.para("If something has gone wrong, contact us first. If we cannot resolve it within 8 weeks, you can ask the "
                   "Energy Ombudsman to look at your complaint for free.", d.small)
            self.emit(doc_id, d, {"doc_type": "utility_bill", "doc_date": str(bill_date), "period_start": str(start),
                                  "period_end": str(end), "reference": e["account"]})
            read = read_end
            bills.append(dict(doc_id=doc_id, start=start, end=end, total=total, kwh=p["kwh"], tariff=tariff, days=days))

            ph = period_phrase(start, end)
            self.q("lookup", self.pick(f"What was the total amount due on the electricity bill for {ph}?",
                                       f"How much was the BrightSpark electricity bill covering {ph}?",
                                       f"What did we pay for electricity for {ph}?"),
                   fmt_money(total), [[(doc_id, "total"), (doc_id, "charges")]], float(total))
            self.q("lookup", self.pick(f"How many kWh of electricity did we use between {fmt_date(start)} and {fmt_date(end)}?",
                                       f"What was our electricity consumption in kWh for {ph}?"),
                   f"{p['kwh']:,} kWh", [[(doc_id, "usage"), (doc_id, "charges")]], float(p["kwh"]))
            if i in (0, 2):
                self.q("lookup", f"What unit rate per kWh did BrightSpark charge on the electricity bill for {ph}?",
                       f"{D(tariff['unit_rate']) * 100:.2f}p per kWh ({tariff['name']})", [[(doc_id, "charges")]])
            if i == 1:
                self.q("identifier", f"What is the bill number of the BrightSpark electricity bill for {ph}?",
                       bill_no, [[(doc_id, "header")]])

        b25 = [b for b in bills if b["start"].year == 2025]
        fixed24 = [b for b in bills if b["tariff"]["name"] == "SparkFix 24"]
        self.q("date_deadline", "When does our SparkFix 24 electricity tariff end?", fmt_date(fixed24[0]["tariff"]["end"]),
               [[(b["doc_id"], "tariff") for b in fixed24]])
        self.q("identifier", f"Which supplier is account number {e['account']} with, and for what service?",
               f"BrightSpark Energy, electricity", [[(b["doc_id"], "header") for b in bills]])
        tot = sum(b["total"] for b in b25)
        self.q("cross_document", "How much did we spend on electricity in total in 2025?", fmt_money(tot),
               [[(b["doc_id"], "total"), (b["doc_id"], "charges")] for b in b25], float(tot))
        kwh = sum(b["kwh"] for b in b25)
        self.q("cross_document", "How many kWh of electricity did we use across all of 2025?", f"{kwh:,} kWh",
               [[(b["doc_id"], "usage"), (b["doc_id"], "charges")] for b in b25], float(kwh))
        top = max(b25, key=lambda b: b["total"])
        self.q("cross_document", "Which quarter of 2025 had the highest electricity bill, and how much was it?",
               f"{period_phrase(top['start'], top['end'])}: {fmt_money(top['total'])}",
               [[(b["doc_id"], "total"), (b["doc_id"], "charges")] for b in b25], float(top["total"]))
        a, z = bills[0], bills[-1]
        self.q("cross_document", "How did the January to March 2026 electricity bill compare with January to March 2025?",
               f"{fmt_money(z['total'])} in 2026 vs {fmt_money(a['total'])} in 2025, a difference of {fmt_money(z['total'] - a['total'])}",
               [[(a["doc_id"], "total"), (a["doc_id"], "charges")], [(z["doc_id"], "total"), (z["doc_id"], "charges")]],
               float(z["total"] - a["total"]))
        self.q("date_deadline", "When will the direct debit for the January to March 2026 electricity bill be collected?",
               fmt_date(z["end"] + timedelta(days=20)), [[(z["doc_id"], "total")]])
        self.elec_bills = bills

    # -------------------------------------------------------------- gas
    def gas(self):
        g = self.w["gas"]
        read = g["periods"][0]["read_start"]
        bills = []
        for i, p in enumerate(g["periods"]):
            start, end = parse_date(p["start"]), parse_date(p["end"])
            rate = [r for r in g["rates"] if parse_date(r["from"]) <= start][-1]
            days = (end - start).days + 1
            kwh = round(p["m3"] * g["conversion"])
            energy = money(D(kwh) * D(rate["unit_rate"]))
            standing = money(days * D(rate["standing"]))
            sub = energy + standing
            vat = money(sub * D(g["vat_rate"]))
            total = sub + vat
            bill_date = end + timedelta(days=9)
            pay_by = bill_date + timedelta(days=21)
            doc_id = f"gas-{start.year}-q{(start.month - 1) // 3 + 1}"
            read_end = read + p["m3"]

            d = Doc(g["vendor"], f"Gas statement {period_phrase(start, end)}")
            d.para("Gas statement", d.h1)
            self.address_block(d, "R. Marlowe &amp; S. Okafor")
            d.kv([("Customer account", g["account"]), ("Statement date", fmt_date(bill_date)),
                  ("Period", fmt_period(start, end))], key="header")
            d.mark("total", Paragraph(f"Amount to pay: {fmt_money(total)}", d.big),
                   Paragraph(f"Please pay by {fmt_date(pay_by)}. You pay by monthly direct debit, so this amount will be "
                             f"balanced against your payments.", d.p))
            d.para("Your gas use this period", d.h2)
            d.table([["Meter", "Opening read (m³)", "Closing read (m³)", "Volume (m³)", "Energy (kWh)"],
                     [g["meter"], f"{read:,}", f"{read_end:,}", f"{p['m3']:,}", f"{kwh:,}"]],
                    [28 * mm, 35 * mm, 35 * mm, 30 * mm, 30 * mm], key="usage")
            d.para(f"We convert the volume of gas into kilowatt hours using a factor of {g['conversion']} kWh per m³, "
                   "based on the average calorific value in your area.", d.small)
            d.para("Gas safety", d.h2)
            d.para("If you smell gas, leave the property, do not use electrical switches and call the emergency line on "
                   "0800 111 999 at any time.")
            d.add(PageBreak())
            d.para("How your charges are calculated", d.h1)
            d.table([["Item", "Detail", "Charge"],
                     ["Gas used", f"{kwh:,} kWh at {D(rate['unit_rate']) * 100:.2f}p/kWh", fmt_money(energy)],
                     ["Daily standing charge", f"{days} days at {D(rate['standing']) * 100:.2f}p/day", fmt_money(standing)],
                     ["Charges before VAT", "", fmt_money(sub)],
                     [f"VAT ({int(g['vat_rate'] * 100)}%)", "", fmt_money(vat)],
                     ["<b>Total for this period</b>", "", f"<b>{fmt_money(total)}</b>"]],
                    [55 * mm, 70 * mm, 40 * mm], key="charges")
            d.para("Your tariff", d.h2)
            d.para(f"You are on Northfield Standard Variable. Prices can go up or down; we will give you 30 days' notice of any "
                   f"change. Your current unit rate is {D(rate['unit_rate']) * 100:.2f}p per kWh and your standing charge is "
                   f"{D(rate['standing']) * 100:.2f}p per day.", key="rates")
            d.para("Get help paying", d.h2)
            d.para("If you are worried about paying, contact us. We can set up a payment plan or point you to grants.", d.small)
            self.emit(doc_id, d, {"doc_type": "utility_bill", "doc_date": str(bill_date), "period_start": str(start),
                                  "period_end": str(end), "reference": g["account"]})
            read = read_end
            bills.append(dict(doc_id=doc_id, start=start, end=end, total=total, kwh=kwh, m3=p["m3"], rate=rate))

            ph = period_phrase(start, end)
            self.q("lookup", self.pick(f"How much was the gas bill for {ph}?",
                                       f"What was the amount to pay on the Northfield Gas statement for {ph}?"),
                   fmt_money(total), [[(doc_id, "total"), (doc_id, "charges")]], float(total))
            if i in (0, 2, 4):
                self.q("lookup", f"How much gas did we use in {ph}, in cubic metres and kWh?",
                       f"{p['m3']} m³, which is {kwh:,} kWh", [[(doc_id, "usage")]], float(kwh))

        last = bills[-1]
        self.q("lookup", "What is the current gas unit rate and standing charge with Northfield Gas?",
               f"{D(last['rate']['unit_rate']) * 100:.2f}p per kWh and {D(last['rate']['standing']) * 100:.2f}p per day",
               [[(last["doc_id"], "rates"), (last["doc_id"], "charges")]])
        b25 = [b for b in bills if b["start"].year == 2025]
        tot = sum(b["total"] for b in b25)
        self.q("cross_document", "What was our total gas spend for 2025?", fmt_money(tot),
               [[(b["doc_id"], "total"), (b["doc_id"], "charges")] for b in b25], float(tot))
        e25 = [b for b in self.elec_bills if b["start"].year == 2025]
        energy = tot + sum(b["total"] for b in e25)
        self.q("cross_document", "How much did we spend on energy (electricity and gas together) in 2025?", fmt_money(energy),
               [[(b["doc_id"], "total"), (b["doc_id"], "charges")] for b in b25 + e25], float(energy))
        a, z = bills[0], bills[-1]
        self.q("cross_document", "Did we use more gas in January to March 2026 than in the same quarter of 2025?",
               f"Yes: {z['kwh']:,} kWh ({z['m3']} m³) in 2026 vs {a['kwh']:,} kWh ({a['m3']} m³) in 2025",
               [[(a["doc_id"], "usage")], [(z["doc_id"], "usage")]])

    # -------------------------------------------------------------- water
    def water(self):
        wt = self.w["water"]
        totals = []
        for b in wt["bills"]:
            start, end, bd = parse_date(b["start"]), parse_date(b["end"]), parse_date(b["bill_date"])
            total = money(D(b["supply"]) + D(b["wastewater"]) + D(b["standing"]))
            inst = money(total / 6)
            doc_id = f"water-{start.year}-h{1 if start.month == 1 else 2}"
            d = Doc(wt["vendor"], f"Water bill {period_phrase(start, end)}")
            d.para("Your water and wastewater bill", d.h1)
            self.address_block(d, "The Occupier")
            d.kv([("Account", wt["account"]), ("Bill date", fmt_date(bd)), ("Charges for", fmt_period(start, end)),
                  ("Charging basis", "Assessed charge (no meter fitted)")], key="header")
            d.table([["Charge", "Amount"], ["Water supply", fmt_money(b["supply"])], ["Wastewater collection and treatment", fmt_money(b["wastewater"])],
                     ["Fixed standing charge", fmt_money(b["standing"])], ["<b>Total for the period</b>", f"<b>{fmt_money(total)}</b>"]],
                    [110 * mm, 50 * mm], key="charges")
            d.para(f"You pay in 6 monthly instalments of {fmt_money(inst)} by direct debit, starting on the 1st of the "
                   f"first month of the period. No VAT is charged on household water services.", key="instalments")
            d.para("Could a water meter save you money?", d.h2)
            d.para("Households with fewer people than bedrooms often pay less with a meter. Meters are fitted free, and you "
                   "can switch back within 24 months if you are not better off.")
            self.emit(doc_id, d, {"doc_type": "utility_bill", "doc_date": str(bd), "period_start": str(start),
                                  "period_end": str(end), "reference": wt["account"]}, scanned=True)
            totals.append((doc_id, total))
            ph = period_phrase(start, end)
            self.q("lookup", self.pick(f"How much is the water bill for {ph}?", f"What do we pay Lindenford Water for {ph}?"),
                   fmt_money(total), [[(doc_id, "charges")]], float(total))
            if start.year == 2026:
                self.q("lookup", f"What is the monthly water instalment for {ph}?", fmt_money(inst), [[(doc_id, "instalments")]], float(inst))
                self.q("lookup", f"How much is the wastewater charge on the water bill for {ph}?", fmt_money(b["wastewater"]),
                       [[(doc_id, "charges")]], float(b["wastewater"]))
        tot = totals[0][1] + totals[1][1]
        self.q("cross_document", "How much were our water charges for the whole of 2025?", fmt_money(tot),
               [[(totals[0][0], "charges")], [(totals[1][0], "charges")]], float(tot))

    # -------------------------------------------------------------- telecom
    def telecom(self):
        t = self.w["telecom"]
        totals = []
        for i, b in enumerate(t["bills"]):
            y, m = map(int, b["month"].split("-"))
            bill_date = date(y, m, 1)
            month_name = bill_date.strftime("%B %Y")
            extras = sum((D(x["amount"]) for x in b["extras"]), D(0))
            total = money(D(t["monthly"]) + extras)
            vat = money(total - total / (1 + D(t["vat_rate"])))
            doc_id = f"nimbus-{b['month']}"
            d = Doc(t["vendor"], f"Nimbus bill {month_name}")
            d.para(f"Your Nimbus bill for {month_name}", d.h1)
            self.address_block(d, "Robin Marlowe")
            d.kv([("Account number", t["account"]), ("Bill date", fmt_date(bill_date)), ("Plan", t["plan"]),
                  ("Payment", f"Direct Debit on {fmt_date(bill_date + timedelta(days=9))}")], key="header")
            d.table([["Summary", "Amount (incl. VAT)"], ["Monthly plan charge", fmt_money(t["monthly"])],
                     ["Extras and one-off charges", fmt_money(extras)], ["<b>Total this month</b>", f"<b>{fmt_money(total)}</b>"]],
                    [110 * mm, 50 * mm], key="total")
            d.para("Did you know?", d.h2)
            d.para("You can check your data use and set spending caps in the Nimbus app. Spending caps stop extra charges "
                   "for calls, texts and data outside your allowance.")
            d.add(PageBreak())
            d.para("Itemised charges", d.h1)
            rows = [["Item", "Amount"], [f"{t['plan']}, {month_name}", fmt_money(t["monthly"])]]
            rows += [[x["item"], fmt_money(x["amount"])] for x in b["extras"]] or [["No extra charges this month", fmt_money(0)]]
            rows += [["<b>Total</b>", f"<b>{fmt_money(total)}</b>"], [f"Includes VAT at {int(t['vat_rate'] * 100)}%", fmt_money(vat)]]
            d.table(rows, [120 * mm, 40 * mm], key="items")
            d.para("Roaming", d.h2)
            d.para("Use in the Europe Zone is included. Outside the Europe Zone you pay the rates in our Roaming Price Guide "
                   "unless you buy a travel pass before you go.", d.small)
            self.emit(doc_id, d, {"doc_type": "phone_bill", "doc_date": str(bill_date), "period_start": str(bill_date),
                                  "period_end": str(month_end(y, m)), "reference": t["account"]})
            totals.append((doc_id, total, extras, b))
            if i in (0, 2, 3, 5):
                self.q("lookup", self.pick(f"How much was the Nimbus bill for {month_name}?",
                                           f"What was the total on our phone and broadband bill for {month_name}?"),
                       fmt_money(total), [[(doc_id, "total"), (doc_id, "items")]], float(total))
        nov = next(x for x in totals if x[3]["month"] == "2025-11")
        self.q("lookup", "How much did we pay in roaming charges on the November 2025 Nimbus bill?", fmt_money(nov[2]),
               [[(nov[0], "items"), (nov[0], "total")]], float(nov[2]))
        dec = next(x for x in totals if x[3]["month"] == "2025-12")
        self.q("lookup", "What extra charge was on the December 2025 phone bill?",
               f"International calls to Nigeria, 37 min: {fmt_money(dec[2])}", [[(dec[0], "items")]], float(dec[2]))
        feb = next(x for x in totals if x[3]["month"] == "2026-02")
        self.q("lookup", "Why was the February 2026 Nimbus bill higher than usual?",
               f"A 20 GB data add-on for mobile 2 ({fmt_money(feb[2])})", [[(feb[0], "items")]], float(feb[2]))
        q4 = totals[:3]
        tot = sum(x[1] for x in q4)
        self.q("cross_document", "What did we pay Nimbus in total for October, November and December 2025?", fmt_money(tot),
               [[(x[0], "total"), (x[0], "items")] for x in q4], float(tot))
        ex = sum(x[2] for x in totals)
        self.q("cross_document", "How much have we paid in Nimbus extra charges, on top of the plan, since the contract started?",
               fmt_money(ex), [[(x[0], "items"), (x[0], "total")] for x in totals if x[2] > 0], float(ex))

    # -------------------------------------------------------------- invoices
    def invoice_doc(self, vendor, number, inv_date, bill_to, items, vat_rate, due=None, iban=None, extra_header=()):
        lines = [(desc, qty, money(price), money(D(qty) * D(price))) for desc, qty, price in items]
        net = sum(x[3] for x in lines)
        vat = money(net * D(vat_rate))
        total = net + vat
        d = Doc(vendor, f"Invoice {number}")
        d.para("INVOICE", d.h1)
        d.para("<br/>".join(bill_to))
        d.add(Spacer(1, 6))
        d.kv([("Invoice number", number), ("Invoice date", fmt_date(inv_date)), *extra_header,
              *([("Payment due", fmt_date(due))] if due else [])], key="header")
        rows = [["Description", "Qty", "Unit price", "Amount"]]
        rows += [[desc, str(qty), fmt_money(p), fmt_money(a)] for desc, qty, p, a in lines]
        rows += [["Net total", "", "", fmt_money(net)], [f"VAT at {int(vat_rate * 100)}%", "", "", fmt_money(vat)],
                 ["<b>Total due</b>", "", "", f"<b>{fmt_money(total)}</b>"]]
        d.table(rows, [85 * mm, 15 * mm, 30 * mm, 35 * mm], key="lines")
        if iban:
            d.para(f"Please pay by bank transfer to IBAN {iban}, quoting the invoice number.", key="payment")
        return d, net, vat, total, lines

    def client_invoices(self):
        ci, biz = self.w["client_invoices"], self.hh["business"]
        clients = {"Juniper Books": ["Accounts Payable", "Juniper Books", "22 Paper Mill Road", "Port Arden PA4 1JB"],
                   "Calloway Coffee Co.": ["Finance team", "Calloway Coffee Co.", "5 Roastery Lane", "Lindenford LF5 2CC"],
                   "Meridian Museum": ["Purchase Ledger", "Meridian Museum of Maritime History", "The Esplanade", "Lindenford LF4 0MM"],
                   "Tidewater Games": ["Studio Operations", "Tidewater Games", "Dock 3, Harbour Works", "Port Arden PA2 7TG"]}
        out = []
        for i, inv in enumerate(ci["invoices"]):
            d_inv = parse_date(inv["date"])
            terms = ci["terms_days"][inv["client"]]
            due = d_inv + timedelta(days=terms)
            d, net, vat, total, lines = self.invoice_doc(
                "Okafor Illustration", inv["number"], d_inv, clients[inv["client"]], inv["items"], ci["vat_rate"], due,
                biz["iban"], extra_header=[("Supplier VAT number", biz["vat_number"]), ("Terms", f"{terms} days")])
            d.para("Thank you for your business. Usage rights are granted on receipt of full payment.", d.small)
            doc_id = f"inv-{inv['number'].lower()}"
            self.emit(doc_id, d, {"doc_type": "invoice_issued", "doc_date": str(d_inv), "period_start": None,
                                  "period_end": None, "reference": inv["number"], "counterparty": inv["client"]})
            out.append(dict(doc_id=doc_id, inv=inv, total=total, net=net, vat=vat, due=due, lines=lines, d=d_inv))
            n = inv["number"]
            if i in (0, 2, 3, 6, 7, 9, 10, 11):
                self.q("identifier", self.pick(f"What is the total including VAT on invoice {n}?", f"How much is invoice {n} for in total?"),
                       fmt_money(total), [[(doc_id, "lines")]], float(total))
            if i in (5, 8):
                self.q("identifier", f"Who was invoice {n} issued to, and for what?",
                       f"{inv['client']}: " + "; ".join(x[0] for x in lines), [[(doc_id, "header"), (doc_id, "lines")]])
            if i in (1, 4, 8):
                self.q("date_deadline", f"When is payment due on invoice {n}?", fmt_date(due), [[(doc_id, "header")]])
        # line-item lookups by description, no invoice number given
        for idx, li in ((4, 1), (2, 1), (5, 1), (7, 1)):
            x = out[idx]
            desc, qty, price, amt = x["lines"][li]
            self.q("lookup", f"How much did Sam charge {x['inv']['client']} for '{desc}'?",
                   f"{fmt_money(amt)} net ({qty} x {fmt_money(price)})", [[(x["doc_id"], "lines")]], float(amt))
        y25 = [x for x in out if x["d"].year == 2025]
        for client in ("Juniper Books", "Tidewater Games", "Meridian Museum", "Calloway Coffee Co."):
            xs = [x for x in y25 if x["inv"]["client"] == client]
            net = sum(x["net"] for x in xs)
            self.q("cross_document", f"How much did Sam invoice {client} in 2025, before VAT?", fmt_money(net),
                   [[(x["doc_id"], "lines")] for x in xs], float(net))
        net = sum(x["net"] for x in y25)
        self.q("cross_document", "What was Okafor Illustration's total invoiced revenue in 2025, excluding VAT?", fmt_money(net),
               [[(x["doc_id"], "lines")] for x in y25], float(net))
        q4 = [x for x in y25 if x["d"].month >= 10]
        vat = sum(x["vat"] for x in q4)
        self.q("cross_document", "How much VAT did Sam charge clients on invoices dated October to December 2025?", fmt_money(vat),
               [[(x["doc_id"], "lines")] for x in q4], float(vat))
        self.client_out = out

    def supplier_invoices(self):
        to = ["Sam Okafor", "Okafor Illustration", *self.hh["address"]]
        out = []
        for inv in self.w["supplier_invoices"]:
            d_inv = parse_date(inv["date"])
            due = d_inv + timedelta(days=14)
            d, net, vat, total, lines = self.invoice_doc(inv["vendor"], inv["number"], d_inv, to, inv["items"], inv["vat_rate"], due)
            d.para("Payment received with thanks if already paid by card or direct debit.", d.small)
            doc_id = "sinv-" + inv["number"].lower().replace("/", "-")
            self.emit(doc_id, d, {"doc_type": "invoice_received", "doc_date": str(d_inv), "period_start": None, "period_end": None,
                                  "reference": inv["number"], "counterparty": "Okafor Illustration"},
                      scanned=inv.get("scanned", False))
            out.append(dict(doc_id=doc_id, inv=inv, net=net, vat=vat, total=total, lines=lines))
        by = {x["inv"]["number"]: x for x in out}
        k = by["KC/INV/55031"]
        self.q("lookup", "How much was the Kestrel Computers invoice in total, including VAT?", fmt_money(k["total"]),
               [[(k["doc_id"], "lines")]], float(k["total"]))
        self.q("lookup", "How much did the extended warranty from Kestrel Computers cost?", fmt_money(k["lines"][2][3]) + " net",
               [[(k["doc_id"], "lines")]], float(k["lines"][2][3]))
        self.q("identifier", "What was invoice KC/INV/55031 for?", "; ".join(x[0] for x in k["lines"]), [[(k["doc_id"], "lines")]])
        h = by["HF-2025-118"]
        self.q("lookup", "What did the accountants charge for the 2024-25 year-end accounts and tax return?",
               f"{fmt_money(h['net'])} plus VAT, {fmt_money(h['total'])} in total", [[(h["doc_id"], "lines")]], float(h["total"]))
        ink = by["IPS-30419"]
        self.q("lookup", "How much did the portfolio prints from Inkwell Print Studio cost?",
               f"{fmt_money(ink['total'])} including VAT (15 prints at {fmt_money(18)})", [[(ink["doc_id"], "lines")]], float(ink["total"]))
        s = by["LOOM-2025-0907"]
        self.q("identifier", "Why is coworking invoice LOOM-2025-0907 higher than the usual membership fee?",
               f"It includes 3 hours of meeting room overage at {fmt_money(15)} per hour ({fmt_money(45)} net)",
               [[(s["doc_id"], "lines")]], 45.0)
        p1, p2 = by["PF-88213"], by["PF-91877"]
        self.q("identifier", "How much VAT was charged on Pixelforge invoice PF-91877?", fmt_money(p2["vat"]), [[(p2["doc_id"], "lines")]], float(p2["vat"]))
        self.q("lookup", "How much did the Pixelforge Creative Suite licence cost for 2025-26?", fmt_money(p1["total"]) + " including VAT",
               [[(p1["doc_id"], "lines")]], float(p1["total"]))
        self.q("cross_document", "By how much did the Pixelforge annual licence go up from 2025-26 to 2026-27?",
               f"{fmt_money(p2['total'] - p1['total'])} including VAT ({fmt_money(p1['total'])} to {fmt_money(p2['total'])})",
               [[(p1["doc_id"], "lines")], [(p2["doc_id"], "lines")]], float(p2["total"] - p1["total"]))
        loom = [x for x in out if x["inv"]["vendor"] == "The Loom Coworking"]
        l25 = [x for x in loom if x["inv"]["date"].startswith("2025")]
        tot = sum(x["total"] for x in l25)
        self.q("cross_document", "How much did we pay The Loom Coworking on the 2025 invoices we have?", fmt_money(tot),
               [[(x["doc_id"], "lines")] for x in l25], float(tot))
        eq = [k, p1]
        tot = sum(x["total"] for x in eq)
        self.q("cross_document", "What did Sam spend in 2025 on computer equipment and design software, including VAT?",
               fmt_money(tot), [[(x["doc_id"], "lines")] for x in eq], float(tot))

    # -------------------------------------------------------------- contracts & letters
    def contracts(self):
        for c in self.w["contracts"]:
            d = Doc(c["vendor"], c["title"])
            d.para(c["title"], d.h1)
            d.para(f"Document date: {fmt_date(c['doc_date'])}. Reference: {c['reference']}.", key="dated")
            d.para(c["intro"])
            d.add(Spacer(1, 4))
            d.para("Summary", d.h2)
            d.kv(c["summary"], key="summary")
            d.para("Terms and conditions", d.h2)
            for j, cl in enumerate(c["clauses"]):
                key = f"clause{j}"
                d.mark(key, Paragraph(cl["title"], ParagraphStyle("ct", parent=d.p, fontName=d.st.bold, spaceBefore=5)),
                       Paragraph(cl["text"], d.p))
                for q in cl.get("questions", []):
                    self.q(q["type"], q["q"], q["a"], [[(c["doc_id"], key)]], near_miss=q.get("near_miss", False))
            dtype = {"Harbourview Lettings": "lease", "Nimbus Telecom": "contract", "The Loom Coworking": "contract"}.get(c["vendor"], "insurance_policy")
            self.emit(c["doc_id"], d, {"doc_type": dtype, "doc_date": c["doc_date"], "period_start": None, "period_end": None,
                                       "reference": c["reference"]}, scanned=c.get("scanned", False))
        ev = lambda doc: [(doc, "summary")]
        self.q("lookup", "How much is the annual premium for the home contents insurance?", "€214.80 a year (€17.90 a month)",
               [ev("contract-home-insurance")], 214.80)
        self.q("identifier", "Which policy is ARW-77310-M, and what does it cover?",
               "The Arrowline Motor Insurance comprehensive policy for the 2019 Corvala Hatch (LF19 KRW)", [ev("contract-car-insurance")])
        self.q("identifier", "What is the policy number of our home contents insurance?", "SHW-HC-204418", [ev("contract-home-insurance")])
        self.q("lookup", "How much is the contents sum insured on the Shieldwell policy?", "€45,000", [ev("contract-home-insurance")], 45000.0)
        self.q("date_deadline", "When does the home contents insurance period end?", "31 August 2026", [ev("contract-home-insurance")])
        self.q("lookup", "How many years of no-claims discount does Robin have on the car insurance?", "6 years (protected)",
               [ev("contract-car-insurance")], 6.0)
        self.q("date_deadline", "As of 1 April 2026, which of our insurance policies or contracts ends or renews next?",
               "The Arrowline car insurance, which ends on 19 May 2026",
               [[("contract-car-insurance", "summary")], [("contract-home-insurance", "summary")], [("contract-lease", "summary")]])
        self.q("lookup", "How much is the deposit on the flat?", "€2,900.00", [[("contract-lease", "summary"), ("contract-lease", "clause2")]], 2900.0)
        self.q("date_deadline", "When did the coworking membership start?", "1 April 2025", [ev("contract-coworking")])

    def letters(self):
        for L in self.w["letters"]:
            d = Doc(L["vendor"], L["title"])
            d.para(f"{L['to']}<br/>" + "<br/>".join(self.hh["address"]))
            d.add(Spacer(1, 8))
            d.para(f"Date: {fmt_date(L['doc_date'])}<br/>Reference: {L['reference']}", key="header")
            d.para(L["title"], d.h1)
            for par in L["paragraphs"]:
                if isinstance(par, dict):
                    d.para(par["text"], key=par["key"])
                else:
                    d.para(par)
                    d.add(Spacer(1, 3))
            self.emit(L["doc_id"], d, {"doc_type": "letter", "doc_date": L["doc_date"], "period_start": None, "period_end": None,
                                       "reference": L["reference"]}, scanned=L.get("scanned", False))
            for q in L["questions"]:
                self.q(q["type"], q["q"], q["a"], [[(L["doc_id"], q["key"])]], q.get("value"))

    def unanswerable(self):
        for text in self.w["unanswerable"]:
            self.q("unanswerable", text, None, [])

    # -------------------------------------------------------------- output
    def resolve(self, evidence):
        out = []
        for alternatives in evidence:
            pages = sorted({(doc, p) for doc, key in alternatives for p in self.marks[doc][key]})
            out.append([[doc, p] for doc, p in pages])
        return out

    def write(self):
        with open(self.out / "corpus" / "documents.jsonl", "w") as f:
            for doc in self.docs:
                f.write(json.dumps(doc) + "\n")
        with open(self.out / "questions.jsonl", "w") as f:
            for i, q in enumerate(self.questions, 1):
                f.write(json.dumps({"qid": f"q{i:03d}", "type": q.type, "question": q.question, "answer": q.answer,
                                    "answer_value": q.value, "evidence": self.resolve(q.evidence),
                                    "near_miss": q.near_miss}, ensure_ascii=False) + "\n")

    def run(self):
        (self.out / "corpus").mkdir(parents=True, exist_ok=True)
        for old in (self.out / "corpus").glob("*.pdf"):
            old.unlink()
        for step in (self.electricity, self.gas, self.water, self.telecom, self.client_invoices,
                     self.supplier_invoices, self.contracts, self.letters, self.unanswerable):
            step()
        self.write()
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="world.yaml")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    world = yaml.safe_load(open(args.world))
    c = Corpus(world, Path(args.out)).run()
    pages = sum(d["pages"] for d in c.docs)
    scans = sum(d["scanned"] for d in c.docs)
    print(f"{len(c.docs)} documents, {pages} pages ({scans} scanned documents), {len(c.questions)} questions")


if __name__ == "__main__":
    main()
