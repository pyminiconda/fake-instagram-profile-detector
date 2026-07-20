"""
ReportGenerator — PDF report generation using ReportLab.

Generates professional PDF reports for:
  • Single profile analysis
  • Batch analysis results
  • Search history export
"""

import io
import os
import tempfile
from datetime import datetime
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# Colors
PRIMARY = colors.HexColor("#1a73e8")
DANGER = colors.HexColor("#dc3545")
SUCCESS = colors.HexColor("#28a745")
WARNING = colors.HexColor("#ffc107")
DARK = colors.HexColor("#212529")
LIGHT_GREY = colors.HexColor("#f8f9fa")
BORDER_GREY = colors.HexColor("#dee2e6")


class ReportGenerator:
    """Generate professional PDF reports."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Add custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Heading1"],
            fontSize=22,
            textColor=PRIMARY,
            spaceAfter=20,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            "SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=14,
            textColor=DARK,
            spaceBefore=16,
            spaceAfter=8,
            borderWidth=1,
            borderColor=PRIMARY,
            borderPadding=4,
        ))
        self.styles.add(ParagraphStyle(
            "Label_Fake",
            parent=self.styles["Normal"],
            fontSize=16,
            textColor=DANGER,
            alignment=TA_CENTER,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            "Label_Genuine",
            parent=self.styles["Normal"],
            fontSize=16,
            textColor=SUCCESS,
            alignment=TA_CENTER,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            "FooterStyle",
            parent=self.styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    @staticmethod
    def _add_footer(canvas, doc):
        """Draw footer on every page."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            A4[0] / 2, 0.5 * inch,
            f"Fake Instagram Profile Detection System  •  Page {doc.page}  •  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        canvas.restoreState()

    # ------------------------------------------------------------------
    # Single profile report
    # ------------------------------------------------------------------
    def generate_single_report(self, profile: dict, prediction: dict,
                                generated_by: str = "System") -> bytes:
        """
        Generate a PDF report for a single profile analysis.

        Args:
            profile: dict with profile snapshot data
            prediction: dict from PredictionEngine.predict()
            generated_by: username of the reporting user

        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=1 * inch, bottomMargin=1 * inch)
        elements = []

        # Title
        elements.append(Paragraph(
            "Profile Analysis Report", self.styles["ReportTitle"]
        ))
        elements.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} by {generated_by}",
            self.styles["Normal"]
        ))
        elements.append(Spacer(1, 20))

        # Profile Snapshot
        elements.append(Paragraph("Profile Snapshot", self.styles["SectionHeader"]))
        profile_data = [
            ["Field", "Value"],
            ["Username", f"@{profile.get('username', 'N/A')}"],
            ["Followers", f"{profile.get('followersCount', 0):,}"],
            ["Following", f"{profile.get('followingCount', 0):,}"],
            ["Posts", f"{profile.get('postsCount', 0):,}"],
            ["Profile Picture", "Yes" if profile.get("hasProfilePicture") else "No"],
            ["Private", "Yes" if profile.get("isPrivate") else "No"],
            ["Verified", "Yes" if profile.get("isVerified") else "No"],
        ]
        bio = profile.get("biography", "")
        if bio:
            profile_data.append(["Bio", bio[:100] + ("..." if len(bio) > 100 else "")])

        t = Table(profile_data, colWidths=[2 * inch, 4 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Prediction Result
        elements.append(Paragraph("Prediction Result", self.styles["SectionHeader"]))
        label = prediction.get("label", "unknown")
        confidence = prediction.get("confidence", 0)
        style = self.styles["Label_Fake"] if label == "fake" else self.styles["Label_Genuine"]
        elements.append(Paragraph(
            f"<b>{label.upper()}</b>  —  Confidence: {confidence:.1%}",
            style
        ))
        if prediction.get("low_confidence"):
            elements.append(Paragraph(
                "⚠️ Low confidence prediction — results may not be reliable.",
                ParagraphStyle("Warning", parent=self.styles["Normal"],
                               textColor=WARNING, fontSize=10)
            ))
        elements.append(Spacer(1, 16))

        # Risk Profile
        elements.append(Paragraph("Risk Profile", self.styles["SectionHeader"]))
        risk_flags = prediction.get("risk_flags", {})
        risk_data = [["Feature", "Status", "Assessment"]]
        for fname, info in risk_flags.items():
            risk_data.append([fname, info["flag"], info["note"]])

        rt = Table(risk_data, colWidths=[2 * inch, 0.8 * inch, 3.2 * inch])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(rt)
        elements.append(Spacer(1, 16))

        # SHAP Values table
        elements.append(Paragraph("Feature Importance (SHAP)", self.styles["SectionHeader"]))
        shap_vals = prediction.get("shap_values", {})
        if shap_vals:
            shap_sorted = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
            shap_data = [["Feature", "SHAP Value", "Direction"]]
            for fname, val in shap_sorted:
                direction = "→ Fake" if val > 0 else "→ Genuine"
                shap_data.append([fname, f"{val:.4f}", direction])

            st = Table(shap_data, colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(st)
        else:
            elements.append(Paragraph("SHAP values not available.", self.styles["Normal"]))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Batch report
    # ------------------------------------------------------------------
    def generate_batch_report(self, results: List[dict],
                               generated_by: str = "System") -> bytes:
        """
        Generate a PDF for batch analysis results.

        Args:
            results: list of dicts, each with keys:
                username, label, confidence, risk_level, timestamp
            generated_by: username

        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=1 * inch, bottomMargin=1 * inch)
        elements = []

        total = len(results)
        fake_count = sum(1 for r in results if r.get("label") == "fake")
        genuine_count = total - fake_count

        # Cover page
        elements.append(Spacer(1, 100))
        elements.append(Paragraph(
            "Batch Analysis Report", self.styles["ReportTitle"]
        ))
        elements.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            ParagraphStyle("Centered", parent=self.styles["Normal"], alignment=TA_CENTER)
        ))
        elements.append(Paragraph(
            f"Generated by: {generated_by}",
            ParagraphStyle("Centered2", parent=self.styles["Normal"], alignment=TA_CENTER)
        ))
        elements.append(Spacer(1, 40))

        # Summary stats
        summary_data = [
            ["Metric", "Value"],
            ["Total Profiles Analyzed", str(total)],
            ["Fake Profiles", str(fake_count)],
            ["Genuine Profiles", str(genuine_count)],
            ["Fake Percentage", f"{fake_count / max(total, 1) * 100:.1f}%"],
        ]
        st = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(st)
        elements.append(PageBreak())

        # Results table
        elements.append(Paragraph("Detailed Results", self.styles["SectionHeader"]))
        results_data = [["#", "Username", "Label", "Confidence", "Timestamp"]]
        for i, r in enumerate(results, 1):
            results_data.append([
                str(i),
                f"@{r.get('username', 'N/A')}",
                r.get("label", "N/A").upper(),
                f"{r.get('confidence', 0):.1%}",
                r.get("timestamp", "N/A"),
            ])

        rt = Table(results_data, colWidths=[0.5 * inch, 1.8 * inch, 1 * inch, 1.2 * inch, 1.5 * inch])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(rt)

        doc.build(elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # History report
    # ------------------------------------------------------------------
    def generate_history_report(self, records: List[dict],
                                 username: str) -> bytes:
        """Generate a PDF export of search history records."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=1 * inch, bottomMargin=1 * inch)
        elements = []

        elements.append(Paragraph("Search History Report", self.styles["ReportTitle"]))
        elements.append(Paragraph(
            f"User: {username}  •  Records: {len(records)}  •  "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            self.styles["Normal"]
        ))
        elements.append(Spacer(1, 20))

        if records:
            data = [["Username", "Label", "Confidence", "Date"]]
            for r in records:
                data.append([
                    r.get("queriedUsername", "N/A"),
                    r.get("resultLabel", "N/A").upper(),
                    f"{r.get('confidenceScore', 0):.1%}",
                    r.get("predictedAt", "N/A")[:19],
                ])

            t = Table(data, colWidths=[2 * inch, 1.2 * inch, 1.2 * inch, 1.8 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("No records found.", self.styles["Normal"]))

        doc.build(elements, onFirstPage=self._add_footer, onLaterPages=self._add_footer)
        return buffer.getvalue()
