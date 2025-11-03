"""
AirDraw Cam ✨ Cute Edition with Color Palette 💖
Draw or write in the middle camera window on top of a soft background.

Gestures:
🎨 Draw: index finger up
🧼 Erase: index + middle up
🖐 Clear: open palm (~1s)
💾 Save snapshot: 's'
🚪 Quit: 'q'
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# ---------- Settings ----------
DRAW_THICKNESS = 6
ERASE_RADIUS = 30
VIDEO_FILENAME = "airdraw_output.avi"
FPS = 20.0
CLEAR_HOLD_SECONDS = 1.0
BACKGROUND_IMAGE = "background.png"  # your cute wallpaper
CAM_WINDOW_SIZE = (640, 480)
# ------------------------------

# Color palette (BGR)
COLOR_PALETTE = [
    ((255, 0, 255), "Pink 💗"),
    ((255, 255, 0), "Yellow 💛"),
    ((0, 255, 0), "Green 💚"),
    ((255, 255, 255), "White 🤍"),
    ((255, 0, 0), "Blue 💙"),
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load background
bg = cv2.imread(BACKGROUND_IMAGE)
if bg is None:
    raise FileNotFoundError(f"Background image '{BACKGROUND_IMAGE}' not found.")
bg = cv2.resize(bg, (1280, 720))
h_bg, w_bg, _ = bg.shape

# Camera init
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
prev_x, prev_y = None, None
clear_start_time = None

cam_w, cam_h = CAM_WINDOW_SIZE
canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
draw_color = (255, 0, 255)  # Default: Pink 💗

# Camera position in background
x_offset = (w_bg - cam_w) // 2
y_offset = (h_bg - cam_h) // 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (w_bg, h_bg))

tipIds = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = []
    fingers.append(1 if lm[tipIds[0]].x < lm[tipIds[0]-1].x else 0)
    for id in range(1, 5):
        fingers.append(1 if lm[tipIds[id]].y < lm[tipIds[id]-2].y else 0)
    return fingers

def save_snapshot(full_frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{ts}.png"
    cv2.imwrite(filename, full_frame)
    print(f"[saved] {filename}")

print("AirDraw Cam started 💖  Press 'q' to quit, 's' to save snapshot.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (cam_w, cam_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "none"
        index_x, index_y = None, None

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            h_frame, w_frame, _ = frame.shape
            lm_index = handLms.landmark[8]
            index_x, index_y = int(lm_index.x * w_frame), int(lm_index.y * h_frame)
            cv2.circle(frame, (index_x, index_y), 8, draw_color, cv2.FILLED)

            f_up = fingers_up(handLms)

            if f_up[1] == 1 and f_up[2] == 0:
                gesture = "draw"
            elif f_up[1] == 1 and f_up[2] == 1:
                gesture = "erase"
            elif sum(f_up) == 5:
                gesture = "palm_open"

            # Handle clear gesture
            if gesture == "palm_open":
                if clear_start_time is None:
                    clear_start_time = time.time()
                elif time.time() - clear_start_time > CLEAR_HOLD_SECONDS:
                    canvas = np.zeros_like(canvas)
                    clear_start_time = None
                    print("[canvas cleared]")
            else:
                clear_start_time = None

            # Drawing / erasing
            if gesture == "draw" and index_x is not None:
                # Check if touching color palette area
                if index_y < 40:
                    section_width = w_frame // len(COLOR_PALETTE)
                    color_index = index_x // section_width
                    if 0 <= color_index < len(COLOR_PALETTE):
                        draw_color = COLOR_PALETTE[color_index][0]
                        print(f"[color changed] → {COLOR_PALETTE[color_index][1]}")
                else:
                    if prev_x is None:
                        prev_x, prev_y = index_x, index_y
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), draw_color, DRAW_THICKNESS)
                    prev_x, prev_y = index_x, index_y
            elif gesture == "erase" and index_x is not None:
                cv2.circle(canvas, (index_x, index_y), ERASE_RADIUS, (0, 0, 0), -1)
                prev_x, prev_y = None, None
            else:
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None
            clear_start_time = None

        # Blend drawing
        cam_overlay = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)

        # Add palette bar at top
        section_width = cam_w // len(COLOR_PALETTE)
        for i, (color, name) in enumerate(COLOR_PALETTE):
            x1 = i * section_width
            x2 = x1 + section_width
            cv2.rectangle(cam_overlay, (x1, 0), (x2, 40), color, -1)
            if draw_color == color:
                cv2.rectangle(cam_overlay, (x1, 0), (x2, 40), (0, 0, 0), 2)

        # Insert camera feed into background
        display_frame = bg.copy()
        display_frame[y_offset:y_offset + cam_h, x_offset:x_offset + cam_w] = cam_overlay

        # Add UI text
        cv2.putText(display_frame, f"Mode: {gesture}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display_frame, f"Color: {draw_color}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("AirDraw Cam 🎨", display_frame)
        out.write(display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_snapshot(display_frame)

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Resources released. Video saved as:", VIDEO_FILENAME)
