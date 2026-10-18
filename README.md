# Hand Gesture Typing (Python, OpenCV, MediaPipe)

Touch a finger joint with your own thumb and that letter types, live, into whatever window is focused.

## Chart
LEFT hand:  wrist=A | index B/C/D | middle E/F/G | ring H/I/J | pinky K/L/M  (tip / mid / base)
RIGHT hand: wrist=N | pinky O/P/Q | ring R/S/T | middle U/V/W | index X/Y/Z  (base / mid / tip)

Control gestures (both hands visible, hold ~0.5s):
- both hands open (5+5) = SPACE
- both fists (0+0) = BACKSPACE
- both thumbs-up (1+1) = ENTER
- fist + thumbs-up = CAPS toggle

A letter fires once per touch. Move the thumb away before touching the same spot again.

## Setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py        # or: python gesture_typing_single_file.py

The hand model (~8 MB) downloads automatically on first run. Click into your text editor before gesturing. Press q in the video window to quit.

## Tuning
- letter_map.py: TOUCH_RATIO (raise if touches don't register), TOUCH_MARGIN (lower = stricter)
- hand_tracker.py: SMOOTHING_ALPHA (lower = smoother, laggier)
- main.py: CONFIRM_FRAMES (frames a touch must hold before it types)
