import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ YOLOv8 Object Detection")
st.caption(
    "Zero-Shot inference on MS COCO (80 classes) — "
    "people, vehicles, electronics, animals and more"
)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_size = st.sidebar.selectbox(
    "Model Size",
    ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    index=0,
    help="Nano=fastest, eXtra Large=most accurate"
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05,
    help="Minimum confidence to show detection"
)

st.sidebar.divider()
st.sidebar.subheader("📊 Model Info")
model_info = {
    "yolov8n": ("Nano", "~6MB", "⚡⚡⚡⚡⚡", "⭐⭐"),
    "yolov8s": ("Small", "~22MB", "⚡⚡⚡⚡", "⭐⭐⭐"),
    "yolov8m": ("Medium", "~52MB", "⚡⚡⚡", "⭐⭐⭐⭐"),
    "yolov8l": ("Large", "~88MB", "⚡⚡", "⭐⭐⭐⭐⭐"),
    "yolov8x": ("XLarge", "~136MB", "⚡", "⭐⭐⭐⭐⭐"),
}
name, size, speed, acc = model_info[model_size]
st.sidebar.info(f"""
**{name}** ({size})
Speed: {speed}
Accuracy: {acc}
Classes: 80 (MS COCO)
""")

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    os.makedirs("models", exist_ok=True)
    return YOLO(f"{model_name}.pt")

with st.spinner(f"Loading {model_size}... 🤖"):
    model = load_model(model_size)

# ── File uploader ─────────────────────────────────────────
st.subheader("📸 Upload Image")
uploaded_file = st.file_uploader(
    "Drop an image here (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# ── Sample images ─────────────────────────────────────────
st.caption("Or try with a sample image:")
s1, s2, s3, s4 = st.columns(4)

sample_urls = {
    "🚗 Traffic": "https://ultralytics.com/images/bus.jpg",
    "🏃 People": "https://ultralytics.com/images/zidane.jpg",
}

use_sample = None
with s1:
    if st.button("🚗 Traffic Scene"):
        use_sample = "https://ultralytics.com/images/bus.jpg"
with s2:
    if st.button("🏃 People"):
        use_sample = "https://ultralytics.com/images/zidane.jpg"

# ── Process image ─────────────────────────────────────────
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif use_sample:
    import requests
    from io import BytesIO
    response = requests.get(use_sample)
    image = Image.open(BytesIO(response.content))

if image is not None:
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original")
        st.image(image, use_column_width=True)
        w, h = image.size
        st.caption(f"Size: {w}×{h}px")

    with col2:
        st.subheader("🤖 AI Vision")
        with st.spinner(f"Running {model_size} inference... 🔍"):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=img_cv2,
                conf=conf_threshold,
                verbose=False
            )

            res = results[0]
            annotated_bgr = res.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True)

    # ── Stats ─────────────────────────────────────────────
    st.divider()
    boxes = res.boxes

    if len(boxes) == 0:
        st.warning("No objects detected. Try lowering the confidence threshold.")
    else:
        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("🎯 Objects Found", len(boxes))
        with k2:
            avg_conf = float(boxes.conf.mean())
            st.metric("📊 Avg Confidence", f"{avg_conf:.1%}")
        with k3:
            unique_classes = len(set([int(b.cls[0]) for b in boxes]))
            st.metric("🏷️ Unique Classes", unique_classes)
        with k4:
            st.metric("🤖 Model", model_size.upper())

        # Count classes
        found_classes = {}
        class_confs = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            found_classes[class_name] = found_classes.get(class_name, 0) + 1
            if class_name not in class_confs:
                class_confs[class_name] = []
            class_confs[class_name].append(conf)

        # Sort by count
        sorted_classes = sorted(
            found_classes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Detection Results")
            labels = [x[0].capitalize() for x in sorted_classes]
            counts = [x[1] for x in sorted_classes]
            avg_confs = [
                np.mean(class_confs[x[0]]) for x in sorted_classes
            ]

            fig = go.Figure(go.Bar(
                x=counts,
                y=labels,
                orientation='h',
                marker=dict(
                    color=avg_confs,
                    colorscale='Blues',
                    colorbar=dict(title="Avg Conf")
                ),
                text=[
                    f"{c} ({a:.0%})"
                    for c, a in zip(counts, avg_confs)
                ],
                textposition='outside'
            ))
            fig.update_layout(
                height=max(200, len(labels) * 40),
                margin=dict(l=0, r=80, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📋 Detailed Report")
            for name, count in sorted_classes:
                avg_c = np.mean(class_confs[name])
                bar = "█" * int(avg_c * 10) + "░" * (10 - int(avg_c * 10))
                st.write(
                    f"**{name.capitalize()}** — {count} detected | "
                    f"`{bar}` {avg_c:.0%}"
                )

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "👁️ YOLOv8 Object Detector | "
    "Ultralytics | MS COCO (80 classes) | "
    "Zero-Shot Inference | "
    "Built by Rashid Nurbekov 🇰🇿"
)