import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("💪 AI Счетчик (Simple Mode)")

# Максимально простой импорт
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

if 'count' not in st.session_state:
    st.session_state.count = 0
if 'stage' not in st.session_state:
    st.session_state.stage = "up"

# Виджет камеры
img_file = st.camera_input("Сделай фото")

if img_file:
    # Читаем картинку
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Переводим в RGB для нейронки
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Берем только координаты локтя (точка 14)
        elbow_y = lm[14].y 
        
        # Упрощенная логика: если локоть ниже плеча — значит отжался
        if elbow_y < 0.5: # Условно "вверху"
            st.session_state.stage = "up"
        if elbow_y > 0.8 and st.session_state.stage == "up": # Условно "внизу"
            st.session_state.count += 1
            st.session_state.stage = "down"
            st.balloons()

    st.header(f"Счет: {st.session_state.count}")
