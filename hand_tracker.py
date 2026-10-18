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