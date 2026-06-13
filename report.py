import io
import os
import tempfile

from fpdf import FPDF


def build_pdf(model_type, summary, report, figs):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Fraud Detection Report", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Model: {model_type}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)

    pdf.set_font("Arial", "", 11)
    for key, value in summary.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Classification Report", ln=True)

    pdf.set_font("Arial", "", 10)
    for label in ["0", "1"]:
        if label in report:
            pdf.cell(
                0,
                7,
                f"Class {label} | Precision: {report[label]['precision']:.4f} | "
                f"Recall: {report[label]['recall']:.4f} | "
                f"F1-score: {report[label]['f1-score']:.4f}",
                ln=True,
            )

    temp_files = []
    try:
        for name, fig in figs.items():
            if fig is None:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.savefig(tmp.name, bbox_inches="tight")
                temp_files.append(tmp.name)

                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, name.replace("_", " ").title(), ln=True)
                pdf.ln(4)
                pdf.image(tmp.name, w=180)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        return pdf_bytes

    finally:
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)
