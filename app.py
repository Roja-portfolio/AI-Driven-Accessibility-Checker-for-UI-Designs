import streamlit as st
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from ml_model import predict_ml_score

from rules.rules import (
    check_contrast_simple,
    check_use_of_color,
    check_text_resize,
    check_alt_text
)
from rules.scoring import compute_score

st.set_page_config(page_title="AI Accessibility Checker", layout="wide")

st.title("🤖 AI-Driven Accessibility Checker for UI Screenshots")
st.write("Upload a UI screenshot (e.g., browsing window or landing page) for accessibility analysis.")

# --- Final Screenshot Filter ---
def is_probably_ui_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    if width < 400 or height < 200:
        return False

    std_dev = np.std(image_np)
    if std_dev < 10 or std_dev > 90:
        return False

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (height * width)
    if edge_density < 0.01 or edge_density > 0.15:
        return False

    top_slice = gray[0:int(0.10 * height), :]
    top_edges = cv2.Canny(top_slice, 100, 200)
    top_edge_density = np.sum(top_edges > 0) / (top_slice.shape[0] * top_slice.shape[1])
    if top_edge_density < 0.01:
        return False

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    doc_edges = cv2.Canny(blur, 75, 200)
    contours, _ = cv2.findContours(doc_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_like_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 300 and cv2.boundingRect(cnt)[3] > 300]
    if len(rect_like_contours) >= 1:
        return False

    return True

# --- Upload and Validate ---
uploaded_file = st.file_uploader("Upload a UI screenshot (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    img_path = "temp_image.png"

    if not is_probably_ui_image(img_path):
        st.error("⚠️ This image does not appear to be a valid webpage screenshot. Please include the browser window (tabs or top bar).")
    else:
        img = Image.open(img_path)
        st.image(img, caption="Uploaded UI Screenshot", use_container_width=True)

        results = {}
        results.update(check_contrast_simple(img_path))
        results.update(check_use_of_color(img_path))
        results.update(check_text_resize(img_path))
        results.update(check_alt_text(img_path))

        rule_score = compute_score(results)

        # --- ML Score with Safety Handling ---
        try:
            ml_score = float(predict_ml_score(img_path))
            if np.isnan(ml_score) or ml_score < 0 or ml_score > 1000:
                ml_score = 0.0
            else:
                ml_score = min(max(ml_score, 0), 100)
        except:
            ml_score = 0.0

        st.subheader("🎯 Accessibility Scores")
        col1, col2 = st.columns(2)
        col1.metric("🧠 Rule-Based Score", f"{rule_score}/100")
        col2.metric("🤖 ML-Predicted Score", f"{ml_score}/100")

        st.subheader("📋 WCAG Evaluation Results")
        st.markdown(f"- **Contrast Score**: {results['contrast']}")
        st.markdown(f"- **Use of Color**: {'Passed' if results['color'] else 'Failed'}")
        st.markdown(f"- **Text Resize Support**: {'Passed' if results['text_resize'] else 'Failed'}")
        st.markdown(f"- **Alt Text Presence**: {'Present' if results['alt_text'] else 'Missing'}")

        st.subheader("🛠 Suggestions")
        suggestions = []
        if results["alt_text"] == 0:
            suggestions.append("Add alt text to all images for screen reader compatibility.")
        if results["contrast"] < 4.5:
            suggestions.append("Improve contrast between text and background (min 4.5:1).")
        if results["color"] == 0:
            suggestions.append("Avoid using color as the only way to convey information.")
        if results["text_resize"] == 0:
            suggestions.append("Ensure text can be resized up to 200% without loss of content.")
        for s in suggestions:
            st.markdown(f"- {s}")

        # --- PDF Report Generation ---
        pdf_path = "accessibility_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("AI Accessibility Report", styles['Title']))
        story.append(Spacer(1, 12))
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(f"Generated on: {now}", styles['Normal']))
        story.append(Spacer(1, 12))

        img_rgb = img.convert("RGB")
        img_rgb_path = "resized_image.jpg"
        img_rgb.save(img_rgb_path)
        story.append(RLImage(img_rgb_path, width=6.5 * inch, height=3.5 * inch))
        story.append(Spacer(1, 16))

        story.append(Paragraph("WCAG Evaluation Results:", styles['Heading2']))
        story.append(Paragraph(f"- Contrast Score: {results['contrast']}", styles['Normal']))
        story.append(Paragraph(f"- Use of Color: {'Passed' if results['color'] else 'Failed'}", styles['Normal']))
        story.append(Paragraph(f"- Text Resize Support: {'Passed' if results['text_resize'] else 'Failed'}", styles['Normal']))
        story.append(Paragraph(f"- Alt Text Presence: {'Present' if results['alt_text'] else 'Missing'}", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"🧠 Rule-Based Score: {rule_score}/100", styles['Normal']))
        story.append(Paragraph(f"🤖 ML-Predicted Score: {ml_score}/100", styles['Normal']))
        story.append(Spacer(1, 12))

        if suggestions:
            story.append(Paragraph("Suggestions:", styles['Heading2']))
            for s in suggestions:
                story.append(Paragraph(f"• {s}", styles['Normal']))

        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="accessibility_report.pdf", mime="application/pdf")
