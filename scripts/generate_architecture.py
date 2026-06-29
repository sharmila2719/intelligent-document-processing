"""
generate_architecture.py
========================
Generates the IDP architecture diagram and saves it to
docs/architecture/idp_architecture.png

Run:
    python scripts/generate_architecture.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_service_box(ax, x, y, width, height, label, sublabel, color, icon_char=""):
    """Draw a rounded rectangle service box with label."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)

    # Service icon / short label
    if icon_char:
        ax.text(
            x + width / 2, y + height * 0.65,
            icon_char,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white", zorder=4,
        )

    # Service name
    ax.text(
        x + width / 2, y + height * 0.35,
        label,
        ha="center", va="center",
        fontsize=6.5, fontweight="bold",
        color="white", zorder=4,
        wrap=True,
    )

    # Sub-label
    if sublabel:
        ax.text(
            x + width / 2, y + height * 0.12,
            sublabel,
            ha="center", va="center",
            fontsize=5.5,
            color="white", zorder=4,
            alpha=0.9,
        )


def draw_phase_label(ax, x, y, width, height, phase_num, phase_name, bg_color):
    """Draw a phase container background with label."""
    bg = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03",
        facecolor=bg_color,
        edgecolor="#cccccc",
        linewidth=1,
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(bg)
    ax.text(
        x + width / 2, y + height - 0.18,
        f"Phase {phase_num}",
        ha="center", va="top",
        fontsize=7, fontweight="bold",
        color="#333333", zorder=2,
    )
    ax.text(
        x + width / 2, y + height - 0.38,
        phase_name,
        ha="center", va="top",
        fontsize=6,
        color="#555555", zorder=2,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", label=""):
    """Draw a flow arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.5,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom", fontsize=5.5, color=color)


def generate_architecture_diagram(output_path: str = "docs/architecture/idp_architecture.png"):
    """Generate and save the IDP architecture diagram."""

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(
        10, 10.6,
        "Intelligent Document Processing (IDP) — AWS AI Services Architecture",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#232F3E",
    )
    ax.text(
        10, 10.25,
        "Amazon S3  ·  Amazon Textract  ·  Amazon Comprehend  ·  "
        "Amazon Bedrock  ·  LangChain  ·  Amazon A2I",
        ha="center", va="center",
        fontsize=9, color="#666666",
    )

    # ─────────────────────────────────────────────────────────────────
    # AWS colour palette
    # ─────────────────────────────────────────────────────────────────
    AWS_ORANGE   = "#FF9900"
    AWS_BLUE     = "#232F3E"
    S3_GREEN     = "#3F8624"
    TEXTRACT_TEAL= "#01A88D"
    COMPREHEND_B = "#1A64AC"
    BEDROCK_PURP = "#7B2D8B"
    A2I_RED      = "#C7131F"
    LAMBDA_GOLD  = "#D45B07"
    PHASE_COLORS = [
        "#FFF3E0", "#E8F5E9", "#E3F2FD",
        "#F3E5F5", "#E8EAF6", "#FCE4EC",
    ]

    # ─────────────────────────────────────────────────────────────────
    # Phase containers
    # ─────────────────────────────────────────────────────────────────
    phase_defs = [
        (0.3,  1.2, 2.8, 7.8,  "1", "Data Capture"),
        (3.4,  1.2, 2.8, 7.8,  "2", "Classification"),
        (6.5,  1.2, 2.8, 7.8,  "3", "Extraction"),
        (9.6,  1.2, 2.8, 7.8,  "4", "Enrichment"),
        (12.7, 1.2, 2.8, 7.8,  "5", "GenAI Q&A"),
        (15.8, 1.2, 3.9, 7.8,  "6", "Human Review"),
    ]
    for i, (px, py, pw, ph, pnum, pname) in enumerate(phase_defs):
        draw_phase_label(ax, px, py, pw, ph, pnum, pname, PHASE_COLORS[i])

    # ─────────────────────────────────────────────────────────────────
    # Service boxes
    # ─────────────────────────────────────────────────────────────────
    bw, bh = 2.2, 1.0   # box width / height
    cy = 8.4             # top row y

    # ── Phase 1: Data Capture ────────────────────────────────────────
    draw_service_box(ax, 0.6,  cy,      bw, bh, "Amazon S3",          "Document Storage",  S3_GREEN,     "🗄")
    draw_service_box(ax, 0.6,  cy-1.5,  bw, bh, "Amazon S3",          "PDF / PNG / TIFF",  S3_GREEN,     "📄")
    draw_service_box(ax, 0.6,  cy-3.0,  bw, bh, "Amazon SQS",         "Event Queue",       COMPREHEND_B, "📬")
    draw_service_box(ax, 0.6,  cy-4.5,  bw, bh, "AWS Lambda",         "Trigger Handler",   LAMBDA_GOLD,  "λ")

    # ── Phase 2: Classification ──────────────────────────────────────
    draw_service_box(ax, 3.7,  cy,      bw, bh, "Amazon Textract",    "DetectDocumentText", TEXTRACT_TEAL,"🔍")
    draw_service_box(ax, 3.7,  cy-1.5,  bw, bh, "Amazon Comprehend",  "Custom Classifier",  COMPREHEND_B, "🏷")
    draw_service_box(ax, 3.7,  cy-3.0,  bw, bh, "Amazon Comprehend",  "Real-time Endpoint", COMPREHEND_B, "⚡")
    draw_service_box(ax, 3.7,  cy-4.5,  bw, bh, "Amazon S3",          "Classified Output",  S3_GREEN,     "📁")

    # ── Phase 3: Extraction ──────────────────────────────────────────
    draw_service_box(ax, 6.8,  cy,      bw, bh, "Amazon Textract",    "AnalyzeDocument",    TEXTRACT_TEAL,"📋")
    draw_service_box(ax, 6.8,  cy-1.5,  bw, bh, "Amazon Textract",    "Tables & Forms",     TEXTRACT_TEAL,"📊")
    draw_service_box(ax, 6.8,  cy-3.0,  bw, bh, "Amazon Textract",    "AnalyzeExpense",     TEXTRACT_TEAL,"🧾")
    draw_service_box(ax, 6.8,  cy-4.5,  bw, bh, "Amazon Textract",    "QUERIES Feature",   TEXTRACT_TEAL,"❓")

    # ── Phase 4: Enrichment ──────────────────────────────────────────
    draw_service_box(ax, 9.9,  cy,      bw, bh, "Amazon Comprehend",  "NER Entities",       COMPREHEND_B, "🔤")
    draw_service_box(ax, 9.9,  cy-1.5,  bw, bh, "Amazon Comprehend",  "PII Detection",      COMPREHEND_B, "🔒")
    draw_service_box(ax, 9.9,  cy-3.0,  bw, bh, "Amazon Comprehend",  "Custom NER Model",   COMPREHEND_B, "🏗")
    draw_service_box(ax, 9.9,  cy-4.5,  bw, bh, "Comprehend Medical", "PHI / ICD / RxNorm", COMPREHEND_B, "🏥")

    # ── Phase 5: GenAI Q&A ───────────────────────────────────────────
    draw_service_box(ax, 13.0, cy,      bw, bh, "Amazon Textract",    "PDF Loader",         TEXTRACT_TEAL,"📄")
    draw_service_box(ax, 13.0, cy-1.5,  bw, bh, "Amazon Bedrock",     "Titan Embeddings",   BEDROCK_PURP, "🔢")
    draw_service_box(ax, 13.0, cy-3.0,  bw, bh, "FAISS",              "Vector Store",       AWS_BLUE,     "📐")
    draw_service_box(ax, 13.0, cy-4.5,  bw, bh, "Amazon Bedrock",     "Claude 3 (LLM)",     BEDROCK_PURP, "🤖")

    # ── Phase 6: Human Review (A2I) ──────────────────────────────────
    draw_service_box(ax, 16.1, cy,      bw, bh, "Amazon A2I",         "Flow Definition",    A2I_RED,      "👁")
    draw_service_box(ax, 16.1, cy-1.5,  bw, bh, "Amazon A2I",         "Human Loop",         A2I_RED,      "👤")
    draw_service_box(ax, 16.1, cy-3.0,  bw, bh, "Amazon SageMaker",   "Workforce / UI",     AWS_ORANGE,   "🖥")
    draw_service_box(ax, 16.1, cy-4.5,  bw, bh, "Amazon S3",          "Reviewed Output",    S3_GREEN,     "✅")

    # ─────────────────────────────────────────────────────────────────
    # Flow arrows between phases (horizontal)
    # ─────────────────────────────────────────────────────────────────
    arrow_y = cy + 0.5
    for x_start, x_end in [
        (2.8,  3.7),   # Phase 1 → Phase 2
        (5.9,  6.8),   # Phase 2 → Phase 3
        (9.0,  9.9),   # Phase 3 → Phase 4
        (12.1, 13.0),  # Phase 4 → Phase 5
        (15.2, 16.1),  # Phase 5 → Phase 6
    ]:
        draw_arrow(ax, x_start, arrow_y, x_end, arrow_y, color=AWS_ORANGE)

    # ─────────────────────────────────────────────────────────────────
    # Vertical arrows within each phase
    # ─────────────────────────────────────────────────────────────────
    for phase_x in [1.7, 4.8, 7.9, 11.0, 14.1, 17.2]:
        for row_y in [cy - 0.05, cy - 1.55, cy - 3.05]:
            draw_arrow(ax, phase_x, row_y, phase_x, row_y - 0.4, color="#888888")

    # ─────────────────────────────────────────────────────────────────
    # Bottom legend
    # ─────────────────────────────────────────────────────────────────
    legend_items = [
        (S3_GREEN,     "Amazon S3"),
        (TEXTRACT_TEAL,"Amazon Textract"),
        (COMPREHEND_B, "Amazon Comprehend"),
        (BEDROCK_PURP, "Amazon Bedrock"),
        (A2I_RED,      "Amazon A2I"),
        (LAMBDA_GOLD,  "AWS Lambda"),
        (AWS_ORANGE,   "Amazon SageMaker"),
    ]
    lx, ly, lw, lh = 0.5, 0.2, 1.0, 0.35
    for i, (color, label) in enumerate(legend_items):
        box = FancyBboxPatch(
            (lx + i * 2.75, ly), lw + 0.1, lh,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=1, zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            lx + i * 2.75 + lw + 0.2, ly + lh / 2,
            label,
            va="center", fontsize=6.5, color="#333333",
        )

    # ─────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Architecture diagram saved to: {output_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
