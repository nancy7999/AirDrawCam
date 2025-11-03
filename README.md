# 🎨 AirDraw Cam – Draw in Air Using Your Hand ✋✨

**AirDraw Cam** is a fun and interactive computer vision project that lets you **draw, doodle, or write in the air** using your hand gestures — just like drawing on Instagram stories!  
The app detects your hand in real time through the webcam and allows you to draw, erase, change colors, and clear the screen — all without touching the keyboard or mouse.

---

## 🌈 Preview
> *(Add your screenshots here once you take them)*  

| Drawing Demo | Gesture Detection |
|---------------|-------------------|
| ![Drawing Screenshot](assets/ss1.png) | ![Gesture Screenshot](assets/ss2.png) |

---

## 💡 Features

✅ **Real-time Hand Tracking** using [MediaPipe](https://developers.google.com/mediapipe)  
✅ **Draw in mid-air** with your index finger  
✅ **Erase** with two fingers (index + middle)  
✅ **Clear canvas** with an open palm ✋  
✅ **Cute Background Overlay** behind your camera feed  
✅ **Color Palette Selector** to switch brush colors  
✅ **Snapshot Save** option (`S` key)  
✅ **Video Recording** of your session (`airdraw_output.avi`)  

---

## 🖐️ Gesture Controls

| Gesture | Action |
|----------|---------|
| 🖕 Index finger up | Draw mode |
| ✌️ Index + Middle up | Erase mode |
| 🖐️ All fingers up (palm open) | Clear the canvas |
| 💾 Press `S` | Save snapshot |
| ❌ Press `Q` | Quit application |

---

## ⚙️ How It Works

1. Uses **MediaPipe Hands** to track 21 hand landmarks in real time.
2. Detects which fingers are up to interpret your gestures.
3. Draws lines, erases, or clears the canvas based on gestures.
4. Combines your **live camera feed** with the **drawing layer** and optional **background**.
5. Records your session and allows saving snapshots.

