import time

import cv2
import pyautogui

from hand_tracker import HandTracker
from letter_map import (
    best_zone, get_combo_action, draw_letter_overlay, TOUCH_RATIO,
)

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