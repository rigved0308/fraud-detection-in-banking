import tempfile
from fpdf import FPDF
import matplotlib.pyplot as plt


def _save_fig(fig: plt.Figure) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", dpi=150)
    tmp.close()
    return tmp.name


def _safe_text(text):
    return (
        str(text)
        .replace("📈", "")
        .replace("🔎", "")
        .replace("🤖", "")
        .replace("🔮", "")
        .replace("💾", "")
        .replace("🟢", "Green")
        .replace("🔴", "Red")
        .replace("🟡", "Yellow")
        .replace("²", "2")
        .replace("–", "-")
        .replace("—", "-")
    )


def build_pdf(ticker, period, metrics, summary_stats, figs: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(12)
    pdf.cell(0, 10, "Stock Price Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, _safe_text(f"Ticker: {ticker}  |  Period: {period}"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 8, "Summary Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for k, v in summary_stats.items():
        pdf.set_x(10)
        pdf.multi_cell(190, 6, _safe_text(f"{k}: {v}"))
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 8, "LSTM Model Evaluation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for k, v in metrics.items():
        pdf.set_x(10)
        pdf.multi_cell(190, 6, _safe_text(f"{k}: {v}"))
    pdf.ln(3)

    chart_titles = {
        "price_ma": "Price & Moving Averages",
        "volume": "Volume Chart",
        "bollinger": "Bollinger Bands",
        "rsi": "RSI Indicator",
        "macd": "MACD",
        "prediction": "LSTM Prediction vs Actual",
        "forecast": "Future Price Forecast",
    }

    for key, title in chart_titles.items():
        if key in figs and figs[key] is not None:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(190, 8, title, new_x="LMARGIN", new_y="NEXT")
            path = _save_fig(figs[key])
            pdf.image(path, x=10, w=185)

    return bytes(pdf.output())