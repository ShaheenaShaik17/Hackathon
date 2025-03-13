#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import psutil
from deepface import DeepFace
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Hand and Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Setup Camera
cap = cv2.VideoCapture(0)  # Try changing index (0, 1, 2) if camera isn't detected
pyautogui.FAILSAFE = False

# Get Audio Control for Windows
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Store Authorized Users
authorized_users = ["User1", "User2"]
current_user = None
drawing = np.zeros((480, 640, 3), dtype=np.uint8)
previous_index_tip = None
start_time = None

def set_volume(volume_level):
    """Set Windows volume level (0.0 to 1.0 scale)"""
    volume_level = max(0, min(1, volume_level))  # Ensure range
    volume.SetMasterVolumeLevelScalar(volume_level, None)

def detect_face(frame):
    """Detect face from video frame"""
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            return frame[y:y+h_box, x:x+w_box]
    return None

def authenticate_user(face_img):
    """Identify user via face recognition"""
    global current_user
    try:
        analysis = DeepFace.find(face_img, db_path="./face_db", enforce_detection=False)
        if analysis and len(analysis[0]) > 0:
            recognized_user = analysis[0]['identity'][0].split('/')[-1].split('.')[0]
            if recognized_user in authorized_users:
                current_user = recognized_user
                return recognized_user
    except:
        pass
    return "Unauthorized"

def recognize_shape():
    """Detect shapes drawn in the air"""
    global drawing
    gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            return "Y"  # YouTube
        elif len(approx) == 4:
            return "L"  # Child Lock
        elif len(approx) > 6:
            return "U"  # Unlock
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera not detected!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_img = detect_face(frame)

    # Process Hand Gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if previous_index_tip:
                cv2.line(drawing, previous_index_tip, (x, y), (0, 255, 0), 2)
            previous_index_tip = (x, y)

            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 3:
                shape = recognize_shape()
                if shape:
                    cv2.putText(frame, f'Command: {shape}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    if shape == "Y":
                        print("Launching YouTube")
                        pyautogui.hotkey('win', '1')  # Open first pinned app (customize as needed)
                    elif shape == "L":
                        print("Child Lock Enabled")
                    elif shape == "U" and current_user:
                        print("Unlocking TV for", current_user)
                drawing = np.zeros((480, 640, 3), dtype=np.uint8)
                start_time = None

    # Face Authentication
    if face_img is not None:
        try:
            analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
            age = analysis[0]['age']
            user_status = authenticate_user(face_img)

            if user_status == "Unauthorized":
                cv2.putText(frame, 'Access Denied', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            cv2.putText(frame, f'User: {user_status}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {age}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if age < 12:
                cv2.putText(frame, 'Child Lock Enabled', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except:
            pass

    frame = cv2.addWeighted(frame, 1, drawing, 0.5, 0)
    cv2.imshow('Gesture Control Windows', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




