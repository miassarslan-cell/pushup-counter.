import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Заголовок
st.set_page_config(page_title="AI Gym", layout="centered")
st.title("💪 Твой AI Счетчик")

# --- САМЫЙ ПРОСТОЙ ИМПОРТ В МИРЕ ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Инициализация модели (без кеша, чтобы не глючило)
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Переменные счета
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'stage' not in st.session_state:
    st.session_state.stage = "up"

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Камера
img_file = st.camera_input("Встань перед камерой боком и сделай фото")

if img_file:
    # Читаем фото
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Нейросеть любит RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        # Рисуем скелет поверх изображения
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
        # Точки для правой стороны: плечо(12), локоть(14), кисть(16)
        s = [lm[12].x, lm[12].y]
        e = [lm[14].x, lm[14].y]
        w = [lm[16].x, lm[16].y]
        
        angle = get_angle(s, e, w)
        
        # Логика отжимания
        if angle > 160: 
            st.session_state.stage = "up"
        if angle < 90 and st.session_state.stage == "up":
            st.session_state.stage = "down"
            st.session_state.count += 1
            st.balloons()

    # Показываем результат (конвертируем обратно в RGB для Streamlit)
    res_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(res_img)
    st.header(f"Отжиманий: {st.session_state.count}")
