import cv2
import mediapipe as mp
import numpy as np
import psutil
import pyautogui
import time
import os
import win32gui
import win32con
import win32api
import webbrowser
import ctypes

# Setup for webcam and hand tracking
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera
cap.set(3, wCam)
cap.set(4, hCam)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
AUTHORIZED_FACE = "authorized_face.jpg"
locked = True  # Lock state

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.97, min_tracking_confidence=0.97)
mpDraw = mp.solutions.drawing_utils

# Initialize cooldown timer
last_action_time = time.time()

def set_window_on_top(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 640, 480, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

def register_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.imwrite(AUTHORIZED_FACE, gray[y:y+h, x:x+w])
        print("Face registered.")
        return True
    return False

def authenticate_user(frame):
    if not os.path.exists(AUTHORIZED_FACE):
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def is_media_playing():
    media_processes = ["vlc.exe", "chrome.exe", "msedge.exe", "firefox.exe", "wmplayer.exe"]
    return any(proc.info['name'] in media_processes for proc in psutil.process_iter(attrs=['name']))

def detect_gesture(results, frame):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            pinky_tip = landmarks[20]

            # Strict gesture detection
            if index_tip[1] < wrist[1] and abs(index_tip[0] - wrist[0]) < 0.03:
                return "O"
            
            if index_tip[1] < wrist[1] and pinky_tip[1] < wrist[1] and middle_tip[1] > wrist[1]:
                return "Y"
            
            if index_tip[1] < wrist[1] and thumb_tip[0] < wrist[0] and middle_tip[1] > wrist[1]:
                return "L"
    return None

# Ensure face registration before starting
while not os.path.exists(AUTHORIZED_FACE):
    print("Registering face... Look at the camera.")
    ret, frame = cap.read()
    if ret:
        register_face(frame)
    time.sleep(2)

time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if authenticate_user(frame):
        locked = False
    
    instruction = "Show face to unlock. O=Play/Pause, Y=YouTube, L=Lock"
    if locked:
        instruction = "Locked: Show authorized face to unlock."
    cv2.putText(frame, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if locked else (0, 255, 0), 2)
    
    if not locked:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = detect_gesture(results, frame)
        if gesture and time.time() - last_action_time > 1:
            if gesture == "O":
                pyautogui.press("space")
            elif gesture == "Y":
                webbrowser.open("https://www.youtube.com")
            elif gesture == "L":
                ctypes.windll.user32.LockWorkStation()
                locked = True
            last_action_time = time.time()
    
    small_frame = cv2.resize(frame, (320, 240))
    screen = np.zeros((hCam, wCam, 3), dtype=np.uint8)
    screen[:240, -320:] = small_frame
    cv2.imshow("Gesture Control", screen)
    set_window_on_top("Gesture Control")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
