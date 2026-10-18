"""
Hand Gesture Typing - single-file version (live, webcam).

Touch a finger joint with your own thumb to type a letter.
LEFT hand:  wrist=A | index B/C/D | middle E/F/G | ring H/I/J | pinky K/L/M   (tip/mid/base)
RIGHT hand: wrist=N | pinky O/P/Q | ring R/S/T | middle U/V/W | index X/Y/Z   (base/mid/tip)
Both open=SPACE | both fists=BACKSPACE | both thumbs-up=ENTER | fist+thumbs-up=CAPS

Setup:  pip install opencv-python mediapipe pyautogui
Run:    python gesture_typing_single_file.py   (click into your text editor first)
Keys in the video window: q = quit, s = swap Left/Right hands
"""
import math
import pyautogui
import os
import time
from collections import deque
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Recent mediapipe releases (0.10.31+) removed the old mp.solutions.hands
# API. This uses the current replacement: the Tasks API's HandLandmarker,
# which needs a small model file downloaded once.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model saved to", MODEL_PATH)


class _Point:
    """Simple x/y/z holder so smoothed landmarks look like the raw
    MediaPipe landmark objects (same .x / .y / .z interface)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# EMA smoothing factor: how much weight the newest frame gets.
# Lower = smoother but laggier, higher = snappier but jitterier.
SMOOTHING_ALPHA = 0.45


class HandTracker:
    """Wraps the current MediaPipe Tasks HandLandmarker."""

    def __init__(self, max_hands=2, detection_confidence=0.6, tracking_confidence=0.6):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # Landmark indices for the tip of each finger (same layout as before)
        self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self._start_time = time.time()
        self._last_ts = -1
        self._prev_landmarks = {}  # hand_label -> list of smoothed _Point
        self.swap_labels = False   # press 's' in the video window to flip Left/Right
        self._label_votes = deque(maxlen=15)
        self._frames_without_hands = 0

    def _smooth(self, label, raw_landmarks):
        """Blends this frame's landmarks with the previous frame's for
        that hand, to cut down on small jitter that would otherwise
        make the thumb-touch detection flicker between segments."""
        prev = self._prev_landmarks.get(label)
        smoothed = []
        for i, lm in enumerate(raw_landmarks):
            if prev is not None:
                x = SMOOTHING_ALPHA * lm.x + (1 - SMOOTHING_ALPHA) * prev[i].x
                y = SMOOTHING_ALPHA * lm.y + (1 - SMOOTHING_ALPHA) * prev[i].y
                z = SMOOTHING_ALPHA * lm.z + (1 - SMOOTHING_ALPHA) * prev[i].z
            else:
                x, y, z = lm.x, lm.y, lm.z
            smoothed.append(_Point(x, y, z))
        self._prev_landmarks[label] = smoothed
        return smoothed

    def process(self, frame_bgr):
        """Returns a list of (hand_label, landmarks) for every hand found.
        hand_label is 'Left' or 'Right'. landmarks is a list of 21 points
        each with .x / .y / .z (normalized 0-1), smoothed across frames."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        detected = []
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                detected.append((handedness[0].category_name, landmarks))

        if not detected:
            self._frames_without_hands += 1
            if self._frames_without_hands > 10:
                self._label_votes.clear()
                self._prev_landmarks.clear()
            return []
        self._frames_without_hands = 0

        # MediaPipe's Left/Right label can flicker frame to frame. With two
        # hands, use screen position (mirrored view: your left hand is on the
        # left). With one hand, use a majority vote over recent frames.
        if len(detected) >= 2:
            detected.sort(key=lambda d: d[1][0].x)
            labelled = [("Left", detected[0][1]), ("Right", detected[1][1])]
            self._label_votes.clear()
        else:
            self._label_votes.append(detected[0][0])
            votes = list(self._label_votes)
            voted = max(set(votes), key=votes.count)
            labelled = [(voted, detected[0][1])]

        hands_data = []
        for label, landmarks in labelled:
            if self.swap_labels:
                label = "Right" if label == "Left" else "Left"
            hands_data.append((label, self._smooth(label, landmarks)))
        return hands_data

    def draw(self, frame_bgr, landmarks):
        """Simple dot overlay for each of the 21 hand landmarks."""
        h, w = frame_bgr.shape[:2]
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)

    def count_fingers(self, landmarks, hand_label):
        """Returns (count_of_fingers_up, list_of_5_bools) for one hand."""
        states = []

        if hand_label == "Right":
            thumb_up = landmarks[self.tip_ids[0]].x < landmarks[self.tip_ids[0] - 1].x
        else:
            thumb_up = landmarks[self.tip_ids[0]].x > landmarks[self.tip_ids[0] - 1].x
        states.append(thumb_up)

        for tip_id in self.tip_ids[1:]:
            finger_up = landmarks[tip_id].y < landmarks[tip_id - 2].y
            states.append(finger_up)

        return sum(states), states


# ===== letter map =====

LEFT_ZONE_LETTERS = {
    ("thumb", "wrist"): "A",
    ("index", "tip"): "B", ("index", "mid"): "C", ("index", "base"): "D",
    ("middle", "tip"): "E", ("middle", "mid"): "F", ("middle", "base"): "G",
    ("ring", "tip"): "H", ("ring", "mid"): "I", ("ring", "base"): "J",
    ("pinky", "tip"): "K", ("pinky", "mid"): "L", ("pinky", "base"): "M",
}

RIGHT_ZONE_LETTERS = {
    ("thumb", "wrist"): "N",
    ("pinky", "base"): "O", ("pinky", "mid"): "P", ("pinky", "tip"): "Q",
    ("ring", "base"): "R", ("ring", "mid"): "S", ("ring", "tip"): "T",
    ("middle", "base"): "U", ("middle", "mid"): "V", ("middle", "tip"): "W",
    ("index", "base"): "X", ("index", "mid"): "Y", ("index", "tip"): "Z",
}

# MediaPipe landmark indices per finger: (MCP, PIP, DIP, TIP)
FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# How close (relative to hand size) the thumb tip must get to a zone to count as touching it.
TOUCH_RATIO = 0.25
# Best zone must be at most this fraction of the 2nd-best distance to count as a confident touch.
TOUCH_MARGIN = 0.70


def _mid(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def _point(landmark):
    return (landmark.x, landmark.y, landmark.z)


# MediaPipe's depth (z) is noisier than x/y, so it counts for less.
Z_WEIGHT = 0.5


def _dist(p, q):
    return math.sqrt(
        (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (Z_WEIGHT * (p[2] - q[2])) ** 2
    )


def _segment_points(landmarks, finger):
    mcp, pip, dip, tip = FINGER_LANDMARKS[finger]
    return {
        "base": _mid(landmarks[mcp], landmarks[pip]),
        "mid": _mid(landmarks[pip], landmarks[dip]),
        "tip": _mid(landmarks[dip], landmarks[tip]),
    }


def _zone_points(landmarks):
    """All 13 touch zones for one hand: wrist point + every finger segment midpoint."""
    zones = {("thumb", "wrist"): _point(landmarks[0])}
    for finger in FINGER_LANDMARKS:
        for seg_name, point in _segment_points(landmarks, finger).items():
            zones[(finger, seg_name)] = point
    return zones


def best_zone(landmarks, hand_label):
    """Returns (key, letter, dist, is_confident) for the zone closest to the
    thumb tip. is_confident=False means idle / released."""
    letters = LEFT_ZONE_LETTERS if hand_label == "Left" else RIGHT_ZONE_LETTERS
    thumb_tip = _point(landmarks[4])
    scale = _dist(_point(landmarks[0]), _point(landmarks[9]))
    if scale < 1e-6:
        scale = 1e-6

    ranked = []
    for key, point in _zone_points(landmarks).items():
        d = _dist(thumb_tip, point) / scale
        ranked.append((d, key, letters[key]))
    ranked.sort(key=lambda t: t[0])

    best_dist, best_key, best_letter = ranked[0]
    second_dist = ranked[1][0] if len(ranked) > 1 else best_dist * 10
    is_confident = best_dist < TOUCH_RATIO and best_dist < second_dist * TOUCH_MARGIN
    return best_key, best_letter, best_dist, is_confident


def draw_letter_overlay(frame_bgr, landmarks, hand_label):
    """Draws every zone's letter on the live hand; the touched zone turns orange."""
    import cv2

    h, w = frame_bgr.shape[:2]
    letters = LEFT_ZONE_LETTERS if hand_label == "Left" else RIGHT_ZONE_LETTERS
    active_key, _, _, is_confident = best_zone(landmarks, hand_label)

    for key, point in _zone_points(landmarks).items():
        px, py = int(point[0] * w), int(point[1] * h)
        active = is_confident and key == active_key
        color = (0, 140, 255) if active else (0, 200, 0)
        cv2.circle(frame_bgr, (px, py), 17 if active else 13, color, -1 if active else 2)
        text_color = (255, 255, 255) if active else (0, 0, 0)
        cv2.putText(frame_bgr, letters[key], (px - 7, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def get_combo_action(left_count, right_count):
    """Two-hand control gestures -> 'SPACE' / 'BACKSPACE' / 'ENTER' / 'CAPS' / None."""
    if left_count == 5 and right_count == 5:
        return "SPACE"
    if left_count == 0 and right_count == 0:
        return "BACKSPACE"
    if left_count == 1 and right_count == 1:
        return "ENTER"
    if {left_count, right_count} == {0, 1}:
        return "CAPS"
    return None


# ===== main loop =====
CAM_INDEX = 0                   # change if you have more than one camera
CONFIRM_FRAMES = 4              # frames a NEW touch must stay stable before it fires
TYPE_COOLDOWN_SECONDS = 0.30    # minimum gap between two typed letters (per hand)
COMBO_DWELL_FRAMES = 12         # two-hand control gestures still need a held pose
COMBO_COOLDOWN_SECONDS = 0.6


class ComboDebouncer:
    """SPACE/BACKSPACE/ENTER/CAPS: fire only after the pose is held for
    COMBO_DWELL_FRAMES, then wait before firing again."""

    def __init__(self):
        self.current = None
        self.stable_count = 0
        self.last_fire_time = 0.0

    def update(self, symbol):
        if symbol is not None and symbol == self.current:
            self.stable_count += 1
        else:
            self.current = symbol
            self.stable_count = 0

        now = time.time()
        if (
            symbol is not None
            and self.stable_count == COMBO_DWELL_FRAMES
            and (now - self.last_fire_time) > COMBO_COOLDOWN_SECONDS
        ):
            self.last_fire_time = now
            return symbol
        return None


class TouchTyper:
    """Event-based typing: fires once when a NEW zone touch is confirmed
    (CONFIRM_FRAMES frames, to reject jitter), like pressing a key. Holding
    does not repeat; release (move the thumb away) before the same zone can
    fire again."""

    def __init__(self):
        self.held_key = None
        self.candidate_key = None
        self.candidate_count = 0
        self.last_fire_time = 0.0

    def reset(self):
        self.held_key = None
        self.candidate_key = None
        self.candidate_count = 0

    def update(self, active_key, active_letter, is_confident):
        if not is_confident:
            self.reset()
            return None

        if active_key == self.held_key:
            return None  # still holding the same touch - already fired

        if active_key == self.candidate_key:
            self.candidate_count += 1
        else:
            self.candidate_key = active_key
            self.candidate_count = 1

        now = time.time()
        if (
            self.candidate_count >= CONFIRM_FRAMES
            and (now - self.last_fire_time) > TYPE_COOLDOWN_SECONDS
        ):
            self.held_key = active_key
            self.candidate_key = None
            self.candidate_count = 0
            self.last_fire_time = now
            return active_letter  # fire!
        return None


def main():
    tracker = HandTracker(max_hands=2)
    cap = cv2.VideoCapture(CAM_INDEX)

    typed_text = ""
    caps_on = False

    combo_debouncer = ComboDebouncer()
    typers = {"Left": TouchTyper(), "Right": TouchTyper()}

    print("Live gesture typing started.")
    print("Click into the text box / editor you want to type into BEFORE gesturing.")
    print("Keys in the video window:  q = quit,  s = swap Left/Right hands")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for a natural view of your own hands

        hands_data = tracker.process(frame)
        hands, counts, zone_info = {}, {}, {}
        for label, landmarks in hands_data:
            draw_letter_overlay(frame, landmarks, label)
            hands[label] = landmarks
            count, _ = tracker.count_fingers(landmarks, label)
            counts[label] = count
            zone_info[label] = best_zone(landmarks, label)

        # Two-hand control gestures take priority over spelling a letter
        fired_combo = None
        if "Left" in hands and "Right" in hands:
            action = get_combo_action(counts["Left"], counts["Right"])
            fired_combo = combo_debouncer.update(action)
        else:
            combo_debouncer.update(None)

        fired_letters = []
        for label in ("Left", "Right"):
            if label in hands and fired_combo is None:
                key, letter, _, is_confident = zone_info[label]
                fired = typers[label].update(key, letter, is_confident)
                if fired:
                    fired_letters.append(fired)
            else:
                typers[label].reset()

        # --- apply actions ---
        if fired_combo == "SPACE":
            pyautogui.press("space")
            typed_text += " "
        elif fired_combo == "BACKSPACE":
            pyautogui.press("backspace")
            typed_text = typed_text[:-1]
        elif fired_combo == "ENTER":
            pyautogui.press("enter")
            typed_text += "\n"
        elif fired_combo == "CAPS":
            caps_on = not caps_on

        for letter in fired_letters:
            char = letter.upper() if caps_on else letter.lower()
            pyautogui.typewrite(char, interval=0)
            typed_text += char

        # --- on-screen overlay ---
        status = []
        for label in ("Left", "Right"):
            if label in zone_info:
                _, letter, dist, confident = zone_info[label]
                mark = f"TOUCH {letter}" if confident else "idle"
                status.append(f"{label[0]}: {mark} d={dist:.2f}")
            else:
                status.append(f"{label[0]}: -")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Caps:{caps_on}  Typed: {typed_text[-40:]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "  ".join(status) + f"  (touch if d<{TOUCH_RATIO})  s=swap",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Hand Gesture Typing (q quit, s swap hands)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            tracker.swap_labels = not tracker.swap_labels
            tracker._prev_landmarks.clear()
            for t in typers.values():
                t.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()