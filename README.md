# 👁️ YOLOv8 Object Detection

> **Real-time object detection web application — Zero-Shot inference on MS COCO (80 classes)**
> Model selector from Nano to XLarge | Confidence threshold | Plotly analytics

---

## 🚀 Live Demo

🔗 **[yolov8-detector.streamlit.app](https://5jqwsy2zmlp6dmztnkhtwk.streamlit.app)**

---

## 📊 Model Performance

| Model | Size | Speed | mAP50-95 | Best for |
|---|---|---|---|---|
| **yolov8n** | ~6MB | ⚡⚡⚡⚡⚡ | 37.3 | Real-time |
| **yolov8s** | ~22MB | ⚡⚡⚡⚡ | 44.9 | Balanced |
| **yolov8m** | ~52MB | ⚡⚡⚡ | 50.2 | Accurate |
| **yolov8l** | ~88MB | ⚡⚡ | 52.9 | High accuracy |
| **yolov8x** | ~136MB | ⚡ | 53.9 | Maximum accuracy |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| **Computer Vision** | Ultralytics YOLOv8 |
| **Image Processing** | OpenCV (`opencv-python-headless`) |
| **Data Processing** | NumPy, Pillow (PIL) |
| **Visualization** | Plotly (bar charts + confidence coloring) |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 🔑 Key Features

### 1. Zero-Shot Inference
Использование предобученных весов на датасете **MS COCO (80 классов)** для мгновенного распознавания без дополнительного Fine-Tuning:
- 👤 People & poses
- 🚗 Vehicles (car, truck, bus, bicycle)
- 🐕 Animals (dog, cat, bird)
- 📱 Electronics (laptop, phone, TV)
- 🍕 Food & household objects

### 2. Speed vs Accuracy Trade-off
Выбор модели прямо в sidebar — от `yolov8n` (Nano, 6MB, real-time) до `yolov8x` (eXtra Large, 136MB, максимальная точность). Один клик — разные характеристики без перезагрузки страницы.

### 3. Confidence Threshold Slider
Настраиваемый порог уверенности (0.1–0.9) — фильтрует слабые детекции и убирает шум в сложных сценах.

### 4. Analytics Dashboard
- **KPI cards** — Objects Found, Avg Confidence, Unique Classes, Model Used
- **Plotly bar chart** — количество объектов с цветовой шкалой по уверенности
- **Detailed report** — ASCII progress bars для каждого класса

### 5. Sample Images
Два встроенных тестовых изображения от Ultralytics — traffic scene и people detection. Работают без загрузки файла.

---

## ⚙️ Architecture

```
User uploads image (JPG/PNG)
        │
        ▼
PIL Image → NumPy Array → OpenCV BGR conversion
        │
        ▼
YOLOv8 model.predict(conf=threshold)
  • Downloads weights automatically on first run
  • Runs on CPU (no GPU required)
  • Returns: Bounding Boxes, Class IDs, Confidence scores
        │
        ▼
res.plot() → Annotated image with Bounding Boxes
        │
        ▼
Statistics parsing:
  • Count per class
  • Average confidence per class
  • Plotly visualization
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/RaNurbekov/yolov8-object-detection.git
cd yolov8-object-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
> YOLOv8 weights (~6MB for Nano) download automatically on first run.

---

## 📁 Project Structure

```
yolov8-object-detection/
├── src/                    # Source utilities
├── app.py                  # Streamlit app + YOLOv8 inference
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Roadmap

| Feature | Description |
|---|---|
| **Video support** | Process MP4/webcam stream frame by frame |
| **Custom Fine-Tuning** | Train YOLOv8 on custom dataset (e.g. KZ traffic) |
| **Export results** | Download annotated image + CSV report |
| **Batch processing** | Upload multiple images at once |

---

## 🔗 Related Projects

Part of a Fintech & AI ecosystem:

- [**fraud-gnn**](https://github.com/RaNurbekov/fraud-gnn) — Graph Neural Networks for fraud detection
- [**pytorch-bank-churn**](https://github.com/RaNurbekov/pytorch-bank-churn) — Custom PyTorch MLP for churn prediction
- [**bank-transaction-categorizer**](https://github.com/RaNurbekov/Transaction-Categorizer-Deep-Learning-PyTorch-Hugging-Face-NLP-) — DistilBERT fine-tuning for NLP classification

> 💡 **Deep Learning progression:** NLP classification (DistilBERT) → Tabular (PyTorch MLP) → Computer Vision (YOLOv8). Three different domains of neural networks.

---

## 📫 Author

**Rashid Nurbekov** — ML Engineer | Fintech & Generative AI | Almaty, Kazakhstan 🇰🇿

[![Telegram](https://img.shields.io/badge/Telegram-@RaNurbek-2CA5E0?style=flat&logo=telegram&logoColor=white)](https://t.me/RaNurbek)
[![Email](https://img.shields.io/badge/Email-nurbekovrashidjob@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nurbekovrashidjob@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-RaNurbekov-181717?style=flat&logo=github&logoColor=white)](https://github.com/RaNurbekov)
