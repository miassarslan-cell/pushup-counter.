import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="AI Gym", layout="centered")
st.title("💪 Твой AI Счетчик")

# --- БРОНЕБОЙНЫЙ ИМПОРТ ---
try:
    # Пробуем стандартный путь
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Если не вышло, лезем во внутренности (для новых версий)
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Инициализация модели
@st.cache_resource
def load_model():
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose = load_model()

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
img_file = st.camera_input("Встань перед камерой боком")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
        # Точки: плечо(11), локоть(13), кисть(15)
        s = [lm[11].x, lm[11].y]
        e = [lm[13].x, lm[13].y]
        w = [lm[15].x, lm[15].y]
        
        angle = get_angle(s, e, w)
        
        if angle > 160: st.session_state.stage = "up"
        if angle < 90 and st.session_state.stage == "up":
            st.session_state.stage = "down"
            st.session_state.count += 1
            st.balloons()

    st.image(rgb)
    st.header(f"Счет: {st.session_state.count}")
