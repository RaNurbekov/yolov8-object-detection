import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# 1. Настройка страницы
st.set_page_config(page_title="AI Object Detector", page_icon="👁️", layout="wide")
st.title("👁️ Система Видеоаналитики (Computer Vision)")
st.write("Загрузите фотографию, и нейросеть (YOLOv8) мгновенно распознает все объекты на ней.")

# 2. Загрузка легкой и быстрой модели (YOLOv8 Nano)
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    # При первом запуске YOLO сама скачает веса yolov8n.pt (~6 МБ) из интернета!
    model = YOLO('yolov8x.pt') 
    return model

with st.spinner("Загрузка оптических сенсоров (YOLOv8)... 🤖"):
    model = load_model()

# 3. Интерфейс для загрузки картинки
uploaded_file = st.file_uploader("Загрузите фото (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4. Превращаем загруженный файл в картинку, понятную Питону
    image = Image.open(uploaded_file)
    st.markdown("---")
    
    # Создаем две колонки: До и После
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Оригинал")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Взгляд ИИ")
        with st.spinner("Анализирую пиксели... 🔍"):
            # Конвертируем PIL Image в формат OpenCV (матрица numpy)
            img_array = np.array(image)
            
            # YOLO ждет цвета в формате BGR, а мы загрузили в RGB. Меняем местами!
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 5. МАГИЯ ЗДЕСЬ: Инференс нейросети
            # conf=0.25 значит, что мы просим игнорировать объекты, в которых ИИ не уверен
            results = model.predict(source=img_cv2, conf=0.25)
            
            # Достаем первую картинку из результатов (у нас она одна)
            res = results[0]
            
            # .plot() автоматически рисует красивые рамки (Bounding Boxes) на картинке!
            annotated_img_bgr = res.plot()
            
            # Конвертируем обратно в RGB для показа на сайте
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_img_rgb, use_column_width=True)
            
    # 6. Подводим итоги (Статистика для бизнеса)
    st.markdown("---")
    st.subheader("📊 Отчет системы:")
    
    # Достаем список того, что нашла нейросеть
    boxes = res.boxes
    if len(boxes) == 0:
        st.warning("Объекты не найдены.")
    else:
        st.success(f"Найдено объектов: **{len(boxes)}**")
        
        # Считаем, сколько каких объектов найдено
        found_classes = {}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id] # Имя объекта (например, 'person', 'car')
            found_classes[class_name] = found_classes.get(class_name, 0) + 1
            
        # Красиво выводим статистику
        for name, count in found_classes.items():
            st.write(f"- **{name.capitalize()}**: {count} шт.")