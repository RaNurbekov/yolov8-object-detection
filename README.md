# 👁️ AI Computer Vision: Object Detection (YOLOv8)

Веб-приложение для видеоаналитики и обнаружения объектов на изображениях в реальном времени с использованием архитектуры State-of-the-Art (SOTA) в области Computer Vision.

## 🛠 Стек технологий
* **Computer Vision:** Ultralytics YOLOv8, OpenCV (`opencv-python-headless`)
* **Data Processing:** Numpy, Pillow (PIL)
* **Frontend:** Streamlit

## ⚙️ Особенности проекта
1. **Zero-Shot Inference:** Использование предобученных весов (на датасете MS COCO - 80 классов) для мгновенного распознавания людей, транспорта, электроники и животных без дополнительного Fine-Tuning.
2. **Speed vs Accuracy Trade-off:** Поддержка масштабирования моделей от `yolov8n` (Nano - для сверхбыстрого real-time инференса) до `yolov8x` (eXtra Large - для максимальной точности на сложных зашумленных сценах).
3. **Data Extraction:** Автоматический парсинг Bounding Boxes и генерация бизнес-отчета (статистика по количеству каждого найденного класса).

## 🚀 Как запустить

### 1. Подготовка
Установите зависимости (включая OpenCV и движок Ultralytics):
```bash
pip install -r requirements.txt

2. Запуск приложения
Запустите веб-интерфейс (модель скачается автоматически в кэш при первом запуске):
code
Bash
streamlit run app.py