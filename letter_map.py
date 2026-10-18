"""
Thumb-to-finger-joint typing: touch = type, like pressing a key.

Each hand has 13 touch zones:
  - 4 fingers x 3 segments each (tip / mid / base) = 12 zones -> letters
  - 1 "wrist" zone (bring your thumb down near your own wrist) -> the
    hand's own base letter (A for left, N for right)

Touch a zone with your OWN thumb (same hand) and a letter fires the
instant the touch is confirmed. Holding the touch does NOT repeat it.
Move your thumb away from all zones (release) before the next touch.

LEFT hand:  wrist=A | index B/C/D | middle E/F/G | ring H/I/J | pinky K/L/M   (tip/mid/base)
RIGHT hand: wrist=N | pinky O/P/Q | ring R/S/T | middle U/V/W | index X/Y/Z   (base/mid/tip)

Two-hand control gestures (both hands visible):
    both open (5+5) -> SPACE | both fists (0+0) -> BACKSPACE
    both thumbs-up (1+1) -> ENTER | fist + thumbs-up -> CAPS toggle
"""

import math

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