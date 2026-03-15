import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Настройка страницы
st.set_page_config(page_title="AI Gym Trainer", layout="centered")
st.title("💪 Твой AI Счетчик Отжиманий")

# Функция для инициализации MediaPipe (кешируем, чтобы не тормозило)
@st.cache_resource
def load_pose():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose = load_pose()
mp_drawing = mp.solutions.drawing_utils

# Переменные для счета (хранятся в сессии браузера)
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Виджет камеры
img_file = st.camera_input("Сделай фото в нижней или верхней точке отжимания")

if img_file:
    # Декодируем изображение
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обработка нейросетью
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Рисуем скелет
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        try:
            landmarks = results.pose_landmarks.landmark
            # Координаты (плечо, локоть, запястье)
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Логика счета
            if angle > 160:
                st.session_state.stage = "ВВЕРХУ"
            if angle < 90 and st.session_state.stage == "ВВЕРХУ":
                st.session_state.stage = "ВНИЗУ"
                st.session_state.counter += 1
                st.balloons() # Эффект праздника при каждом отжимании!

        except Exception as e:
            pass

    # Вывод результата
    st.image(image_rgb, caption="Результат обработки")
    st.metric(label="Количество отжиманий", value=st.session_state.counter)
    st.write(f"Текущая стадия: **{st.session_state.stage}**")

if st.button("Сбросить счетчик"):
    st.session_state.counter = 0
    st.session_state.stage = None
    st.rerun()
