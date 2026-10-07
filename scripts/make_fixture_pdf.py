"""Generate tests/fixtures/handbook.pdf: a 42-page "Acme Store Employee Handbook" fixture.

Every planted fact sits on a known page and is listed in tests/fixtures/planted_facts.json, so
tests (and the Phase 3 golden set) can check that retrieval and citations land on the right page.
The output is deterministic: running the script twice produces the same text on the same pages.

    python scripts/make_fixture_pdf.py [--out-dir tests/fixtures]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

PAGES = 42
TITLE = "Acme Store Employee Handbook"
WIDTH, HEIGHT = LETTER
MARGIN = 72
FONT, SIZE, LEADING, PARA_GAP = "Helvetica", 11, 15, 12

# Required by the Phase 1 plan: (page, id, exact sentence as printed, keywords a correct answer has)
WARRANTY_P20 = "To file a warranty claim, an employee must first collect the original receipt"
WARRANTY_P21 = (
    "and then submit form WC-2 to the Customer Service Desk within 14 days of the fault "
    "being reported."
)
TABLE_ROWS = [
    ["Shift", "Hours", "Allowance"],
    ["Early morning", "5 AM - 9 AM", "$2.50 per hour"],
    ["Evening", "6 PM - 10 PM", "$1.75 per hour"],
    ["Overnight", "10 PM - 5 AM", "$4.00 per hour"],
    ["Sunday", "All day", "$3.00 per hour"],
    ["Public holiday", "All day", "1.5x base pay"],
]

FACTS = [
    dict(
        id="store_hours",
        page=3,
        keywords=["9 AM", "9 PM", "Monday", "Saturday"],
        text="The store is open from 9 AM to 9 PM, Monday through Saturday.",
    ),
    dict(
        id="refund_policy",
        page=12,
        keywords=["30 days", "unused"],
        text="Refunds are accepted within 30 days of purchase as long as the item is unused.",
    ),
    dict(
        id="employee_discount",
        page=18,
        keywords=["15%"],
        text="All employees receive a 15% discount on regular-priced merchandise.",
    ),
    dict(
        id="warranty_claim",
        pages=[20, 21],
        keywords=["WC-2", "14 days", "receipt"],
        text=f"{WARRANTY_P20} {WARRANTY_P21}",
    ),
    dict(
        id="shift_allowances",
        page=27,
        kind="table",
        keywords=["$4.00", "Overnight"],
        text="Shift allowance table: " + "; ".join(" | ".join(r) for r in TABLE_ROWS[1:]),
    ),
    dict(id="blank_page", page=33, kind="near_empty", keywords=[], text="Notes"),
    dict(
        id="data_retention",
        page=40,
        keywords=["7 years"],
        text="Employee records are retained for 7 years after the end of employment.",
    ),
    # Extra facts for the Phase 3 golden set (paraphrase targets, one per page).
    dict(
        id="probation",
        page=6,
        keywords=["90 days"],
        text="New employees complete a probation period of 90 days.",
    ),
    dict(
        id="dress_code",
        page=8,
        keywords=["navy", "closed-toe"],
        text="Staff must wear the navy Acme polo shirt and closed-toe shoes on the sales floor.",
    ),
    dict(
        id="meal_break",
        page=10,
        keywords=["30-minute", "six hours"],
        text="Employees who work more than six hours receive an unpaid 30-minute meal break.",
    ),
    dict(
        id="time_off_notice",
        page=14,
        keywords=["two weeks"],
        text="Vacation requests must be submitted at least two weeks in advance.",
    ),
    dict(
        id="overtime_rate",
        page=16,
        keywords=["1.5 times", "40 hours"],
        text="Overtime is paid at 1.5 times the base rate for hours worked beyond 40 hours "
        "in a week.",
    ),
    dict(
        id="sick_days",
        page=23,
        keywords=["eight", "sick days"],
        text="Full-time employees receive eight paid sick days per calendar year.",
    ),
    dict(
        id="incident_report",
        page=25,
        keywords=["24 hours"],
        text="Any workplace injury must be reported to a manager within 24 hours.",
    ),
    dict(
        id="lost_and_found",
        page=29,
        keywords=["60 days"],
        text="Items left in the lost and found are held for 60 days before being donated.",
    ),
    dict(
        id="cash_drawer",
        page=31,
        keywords=["$5"],
        text="A cash drawer discrepancy of more than $5 must be reported to the shift lead.",
    ),
    dict(
        id="training_hours",
        page=35,
        keywords=["12 hours"],
        text="Every new hire completes 12 hours of paid training before working alone.",
    ),
    dict(
        id="parking",
        page=37,
        keywords=["Lot C"],
        text="Employees must park in Lot C behind the loading dock.",
    ),
]

SECTIONS = {
    1: "Welcome to Acme",
    2: "How to Use This Handbook",
    3: "Store Hours",
    4: "Our History",
    5: "Mission and Values",
    6: "Starting Your Employment",
    7: "Equal Opportunity",
    8: "Dress Code",
    9: "Attendance",
    10: "Breaks and Meals",
    11: "Customer Service",
    12: "Returns and Refunds",
    13: "Scheduling",
    14: "Time Off",
    15: "Holidays",
    16: "Pay and Overtime",
    17: "Payroll Questions",
    18: "Employee Discount",
    19: "Product Knowledge",
    20: "Product Warranties",
    21: "Warranty Claims",
    22: "Health Benefits",
    23: "Sick Leave",
    24: "Workplace Safety",
    25: "Reporting Injuries",
    26: "Emergency Procedures",
    27: "Shift Allowances",
    28: "Loss Prevention",
    29: "Lost and Found",
    30: "Cash Handling",
    31: "Register Procedures",
    32: "Inventory and Stockroom",
    33: "Notes",
    34: "Performance Reviews",
    35: "Training and Development",
    36: "Communication Channels",
    37: "Parking and Transit",
    38: "Technology Use",
    39: "Confidential Information",
    40: "Records and Privacy",
    41: "Leaving Acme",
    42: "Acknowledgement",
}

# Topic-neutral filler: no numbers, no policies, so it can never contradict a planted fact.
FILLER = [
    "This section explains how {topic} works at every Acme location.",
    "Managers are expected to apply the guidance on {topic} consistently and fairly.",
    "If anything about {topic} is unclear, speak with your shift supervisor before acting.",
    "Our approach to {topic} reflects the values described at the start of this handbook.",
    "Team members who have questions about {topic} can also contact the People Team.",
    "The guidance on {topic} may be updated, and the latest version is posted in the break room.",
    "Good judgement matters, and the rules on {topic} are not a substitute for common sense.",
    "Local laws sometimes add requirements on {topic}, and those always take precedence.",
    "We review our practices on {topic} regularly and welcome suggestions from the team.",
    "New team members are introduced to {topic} during their first week on the floor.",
    "Consistency on {topic} helps customers trust us and helps colleagues work together.",
    "Store leaders keep written notes whenever an exception related to {topic} is made.",
    "Everyone shares responsibility for {topic}, whatever their role or seniority.",
    "Questions raised in good faith about {topic} will never lead to retaliation.",
    "Department leads may add store-specific detail on {topic} in their own briefings.",
    "Please read this part of the handbook carefully, because {topic} affects daily work.",
    "Examples in this section on {topic} are illustrations rather than an exhaustive list.",
    "Where this handbook is silent on {topic}, managers will follow the spirit of our values.",
    "Feedback from customers often shapes how we think about {topic} over time.",
    "Clear communication is the most reliable way to avoid misunderstandings about {topic}.",
    "Acme invests in tools and coaching so that {topic} never becomes a burden on the team.",
    "Colleagues are encouraged to help one another understand the expectations on {topic}.",
    "The People Team keeps a short frequently asked questions sheet covering {topic}.",
    "Repeated or serious problems involving {topic} are handled through the normal review process.",
]


def page_paragraphs(page: int) -> list[str]:
    """Body paragraphs for one page: the planted fact (if any) plus deterministic filler."""
    topic = SECTIONS[page].lower()
    rng = random.Random(page)
    pool = FILLER[:]
    rng.shuffle(pool)
    filler = [s.format(topic=topic) for s in pool]
    paras = [" ".join(filler[i : i + 4]) for i in range(0, 16, 4)]
    fact = next((f for f in FACTS if f.get("page") == page and not f.get("kind")), None)
    if fact:
        paras.insert(1, f"{fact['text']} {filler[16]} {filler[17]}")
    return paras


class Writer:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=LETTER, invariant=1)
        self.c.setTitle(TITLE)
        self.c.setAuthor("Acme Store (fixture)")
        self.y = HEIGHT - MARGIN

    def heading(self, text: str) -> None:
        self.c.setFont("Helvetica-Bold", 16)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= 30

    def paragraph(self, text: str) -> None:
        self.c.setFont(FONT, SIZE)
        for line in simpleSplit(text, FONT, SIZE, WIDTH - 2 * MARGIN):
            self.c.drawString(MARGIN, self.y, line)
            self.y -= LEADING
        self.y -= PARA_GAP

    def table(self, rows: list[list[str]]) -> None:
        table = Table(rows, colWidths=[150, 130, 150])
        table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), FONT, SIZE),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", SIZE),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            )
        )
        _, h = table.wrapOn(self.c, WIDTH - 2 * MARGIN, self.y)
        table.drawOn(self.c, MARGIN, self.y - h)
        self.y -= h + PARA_GAP + LEADING

    def new_page(self) -> None:
        self.c.showPage()
        self.y = HEIGHT - MARGIN

    def save(self) -> None:
        self.c.save()


def build(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "handbook.pdf"
    w = Writer(pdf_path)
    for page in range(1, PAGES + 1):
        if page == 33:  # nearly blank: a heading only, below the loader's empty-page threshold
            w.c.setFont(FONT, SIZE)
            w.c.drawString(MARGIN, w.y, "Notes")
        elif page == 21:  # the warranty sentence continues from page 20, so no heading first
            w.paragraph(WARRANTY_P21)
            for para in page_paragraphs(page):
                w.paragraph(para)
        else:
            w.heading(SECTIONS[page] if page != 1 else TITLE)
            paras = page_paragraphs(page)
            if page == 27:
                w.paragraph(paras[0])
                w.paragraph("Shift allowances are paid on top of the base hourly rate:")
                w.table(TABLE_ROWS)
                paras = paras[1:]
            for para in paras:
                w.paragraph(para)
            if page == 20:
                w.paragraph(WARRANTY_P20)
        if page < PAGES:
            w.new_page()
    w.save()

    facts_path = out_dir / "planted_facts.json"
    facts = []
    for f in FACTS:
        entry = {"id": f["id"], "pages": f.get("pages", [f.get("page")])}
        entry |= {"kind": f.get("kind", "text"), "text": f["text"], "keywords": f["keywords"]}
        facts.append(entry)
    payload = {"pdf": pdf_path.name, "title": TITLE, "pages": PAGES, "facts": facts}
    facts_path.write_text(json.dumps(payload, indent=2) + "\n")
    return pdf_path, facts_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    parser.add_argument("--out-dir", type=Path, default=default)
    args = parser.parse_args()
    pdf_path, facts_path = build(args.out_dir)
    print(f"wrote {pdf_path} and {facts_path}")


if __name__ == "__main__":
    main()
